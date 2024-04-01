import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, ProbabilisticTensorDictSequential
from torchrl.objectives import distance_loss
from torchrl.objectives.ppo import ClipPPOLoss

"""
Inherit ClipPPOLoss and modify its loss calculation procedures to obtain the naive error for priority calculation.
"abs_error" is the absolute value of the error between target_return and state_value, and has a shape of (batch_size,)
"""


class CustomPERLoss(ClipPPOLoss):
    def __int__(self,
                actor: ProbabilisticTensorDictSequential,
                critic: TensorDictModule,
                *,
                clip_epsilon: float = 0.2,
                entropy_bonus: bool = True,
                samples_mc_entropy: int = 1,
                entropy_coef: float = 0.01,
                critic_coef: float = 1.0,
                loss_critic_type: str = "smooth_l1",
                normalize_advantage: bool = True,
                gamma: float = None,
                separate_losses: bool = False,
                **kwargs,
                ):
        super(CustomPERLoss, self).__int__(
            actor,
            critic,
            entropy_bonus=entropy_bonus,
            samples_mc_entropy=samples_mc_entropy,
            entropy_coef=entropy_coef,
            critic_coef=critic_coef,
            loss_critic_type=loss_critic_type,
            normalize_advantage=normalize_advantage,
            gamma=gamma,
            separate_losses=separate_losses,
            **kwargs,
        )
        self.register_buffer("clip_epsilon", torch.tensor(clip_epsilon))

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone(False)
        advantage = tensordict.get(self.tensor_keys.advantage, None)
        if advantage is None:
            self.value_estimator(
                tensordict,
                params=self._cached_critic_params_detached,
                target_params=self.target_critic_params,
            )
            advantage = tensordict.get(self.tensor_keys.advantage)
        if self.normalize_advantage and advantage.numel() > 1:
            loc = advantage.mean().item()
            scale = advantage.std().clamp_min(1e-6).item()
            advantage = (advantage - loc) / scale

        log_weight, dist = self._log_weight(tensordict)
        neg_loss = (log_weight.exp() * advantage).mean()
        td_out = TensorDict({"loss_objective": -neg_loss.mean()}, [])
        if self.entropy_bonus:
            entropy = self.get_entropy_bonus(dist)
            td_out.set("entropy", entropy.mean().detach())  # for logging
            td_out.set("loss_entropy", -self.entropy_coef * entropy.mean())
        if self.critic_coef:
            loss_critic_temp, abs_error = self.loss_critic(tensordict)
            loss_critic = loss_critic_temp.mean()
            td_out.set("loss_critic", loss_critic.mean())
        return td_out, abs_error

    def loss_critic(self, tensordict: TensorDictBase):
        if self.separate_losses:
            tensordict = tensordict.detach()
        try:
            target_return = tensordict.get(self.tensor_keys.value_target)
        except KeyError:
            raise KeyError(
                f"the key {self.tensor_keys.value_target} was not found in the input tensordict. "
                f"Make sure you provided the right key and the value_target (i.e. the target "
                f"return) has been retrieved accordingly. Advantage classes such as GAE, "
                f"TDLambdaEstimate and TDEstimate all return a 'value_target' entry that "
                f"can be used for the value loss."
            )

        state_value_td = self.critic(
            tensordict,
            params=self.critic_params,
        )

        try:
            state_value = state_value_td.get(self.tensor_keys.value)
        except KeyError:
            raise KeyError(
                f"the key {self.tensor_keys.value} was not found in the input tensordict. "
                f"Make sure that the value_key passed to PPO is accurate."
            )

        loss_value = distance_loss(
            target_return,
            state_value,
            loss_function=self.loss_critic_type,
        )

        abs_error = self.abs_error(target_return, state_value)

        return self.critic_coef * loss_value, abs_error

    def abs_error(self,
                  data: torch.Tensor,
                  label: torch.Tensor,
                  ):
        error_temp = data - label

        return error_temp.abs().mean(1).flatten()
