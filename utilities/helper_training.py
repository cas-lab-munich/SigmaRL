import torch

# Tensordict modules
from tensordict.tensordict import TensorDictBase, TensorDict

# Data collection
from torchrl.collectors import SyncDataCollector

# Env
from torchrl.envs import TransformedEnv
from torchrl.envs.utils import (
    set_exploration_type,
)
from torchrl.envs.libs.vmas import VmasEnv

# Utils
from matplotlib import pyplot as plt
from typing import Callable, Optional

from matplotlib import pyplot as plt
import json
import os
import re

def get_model_name(parameters):
    # model_name = f"nags{parameters.n_agents}_it{parameters.n_iters}_fpb{parameters.frames_per_batch}_tfrms{parameters.total_frames}_neps{parameters.num_epochs}_mnbsz{parameters.minibatch_size}_lr{parameters.lr}_mgn{parameters.max_grad_norm}_clp{parameters.clip_epsilon}_gm{parameters.gamma}_lmbda{parameters.lmbda}_etp{parameters.entropy_eps}_mstp{parameters.max_steps}_nenvs{parameters.num_vmas_envs}"
    if "road" in parameters.scenario_name:
        model_name = f"reward{parameters.episode_reward_mean_current:.2f}"
    elif "path" in parameters.scenario_name:
        model_name = f"reward{parameters.episode_reward_mean_current:.2f}"
    elif "obstacle_avoidance" in parameters.scenario_name:
        model_name = f"reward{parameters.episode_reward_mean_current:.2f}"
    else:
        raise ValueError(f"Required scenario ('{parameters.scenario_name}') is not found.")

    return model_name

##################################################
## Custom Classes
##################################################
class VmasEnvCustom(VmasEnv):
    def _reset(
        self, tensordict: Optional[TensorDictBase] = None, **kwargs
    ) -> TensorDictBase:
        # print("Custoim _reset()")
        if tensordict is not None and "_reset" in tensordict.keys():
            _reset = tensordict.get("_reset")
            envs_to_reset = _reset.squeeze(-1)
            
            self._env.scenario.training_info = tensordict # TODO
            
            if envs_to_reset.all():
                self._env.reset(return_observations=False)
            else:
                for env_index, to_reset in enumerate(envs_to_reset):
                    if to_reset:
                        self._env.reset_at(env_index, return_observations=False)
        else:
            self._env.reset(return_observations=False)

        obs, dones, infos = self._env.get_from_scenario(
            get_observations=True,
            get_infos=True,
            get_rewards=False,
            get_dones=True,
        )
        dones = self.read_done(dones)

        agent_tds = []
        for i in range(self.n_agents):
            agent_obs = self.read_obs(obs[i])
            agent_info = self.read_info(infos[i])

            agent_td = TensorDict(
                source={
                    "observation": agent_obs,
                },
                batch_size=self.batch_size,
                device=self.device,
            )
            if agent_info is not None:
                agent_td.set("info", agent_info)
            agent_tds.append(agent_td)

        agent_tds = torch.stack(agent_tds, dim=1)
        if not self.het_specs:
            agent_tds = agent_tds.to_tensordict()
        tensordict_out = TensorDict(
            source={"agents": agent_tds, "done": dones, "terminated": dones.clone()},
            batch_size=self.batch_size,
            device=self.device,
        )

        return tensordict_out
    
    
class TransformedEnvCustom(TransformedEnv):
    def rollout(
        self,
        max_steps: int,
        policy: Optional[Callable[[TensorDictBase], TensorDictBase]] = None,
        callback: Optional[Callable[[TensorDictBase], TensorDictBase]] = None,
        auto_reset: bool = True,
        auto_cast_to_device: bool = False,
        break_when_any_done: bool = True,
        return_contiguous: bool = True,
        tensordict: Optional[TensorDictBase] = None,
        out=None,
    ):
        """Executes a rollout in the environment.

        The function will stop as soon as one of the contained environments
        returns done=True.

        Args:
            max_steps (int): maximum number of steps to be executed. The actual number of steps can be smaller if
                the environment reaches a done state before max_steps have been executed.
            policy (callable, optional): callable to be called to compute the desired action. If no policy is provided,
                actions will be called using :obj:`env.rand_step()`
                default = None
            callback (callable, optional): function to be called at each iteration with the given TensorDict.
            auto_reset (bool, optional): if ``True``, resets automatically the environment
                if it is in a done state when the rollout is initiated.
                Default is ``True``.
            auto_cast_to_device (bool, optional): if ``True``, the device of the tensordict is automatically cast to the
                policy device before the policy is used. Default is ``False``.
            break_when_any_done (bool): breaks if any of the done state is True. If False, a reset() is
                called on the sub-envs that are done. Default is True.
            return_contiguous (bool): if False, a LazyStackedTensorDict will be returned. Default is True.
            tensordict (TensorDict, optional): if auto_reset is False, an initial
                tensordict must be provided.

        Returns:
            TensorDict object containing the resulting trajectory.

        The data returned will be marked with a "time" dimension name for the last
        dimension of the tensordict (at the ``env.ndim`` index).
        """
        print("[DEBUG] new env.rollout")
        try:
            policy_device = next(policy.parameters()).device
        except (StopIteration, AttributeError):
            policy_device = self.device

        env_device = self.device

        if auto_reset:
            if tensordict is not None:
                raise RuntimeError(
                    "tensordict cannot be provided when auto_reset is True"
                )
            tensordict = self.reset()
        elif tensordict is None:
            raise RuntimeError("tensordict must be provided when auto_reset is False")
        if policy is None:

            policy = self.rand_action

        kwargs = {
            "tensordict": tensordict,
            "auto_cast_to_device": auto_cast_to_device,
            "max_steps": max_steps,
            "policy": policy,
            "policy_device": policy_device,
            "env_device": env_device,
            "callback": callback,
        }
        if break_when_any_done:
            tensordicts = self._rollout_stop_early(**kwargs)
        else:
            tensordicts = self._rollout_nonstop(**kwargs)
        batch_size = self.batch_size if tensordict is None else tensordict.batch_size
        out_td = torch.stack(tensordicts, len(batch_size), out=out)
        if return_contiguous:
            out_td = out_td.contiguous()
        out_td.refine_names(..., "time")
        return out_td

class SyncDataCollectorCustom(SyncDataCollector):
    # Redefine `rollout`
    @torch.no_grad()
    def rollout(self) -> TensorDictBase:
        """Computes a rollout in the environment using the provided policy.

        Returns:
            TensorDictBase containing the computed rollout.

        """
        # print("[DEBUG] new rollout")
        if self.reset_at_each_iter:
            self._tensordict.update(self.env.reset())

        # self._tensordict.fill_(("collector", "step_count"), 0)
        self._tensordict_out.fill_(("collector", "traj_ids"), -1)
        tensordicts = []

        prediction_horizon = 1  # Set the number of actions you want to generate per step

        with set_exploration_type(self.exploration_type):
            for t in range(self.frames_per_batch):
                if (
                    self.init_random_frames is not None
                    and self._frames < self.init_random_frames
                ):
                    self.env.rand_action(self._tensordict)
                else:
                    for _ in range(prediction_horizon):
                        self.policy(self._tensordict)

                tensordict, tensordict_ = self.env.step_and_maybe_reset(
                    self._tensordict
                )
                self._tensordict = tensordict_.set(
                    "collector", tensordict.get("collector").clone(False)
                )
                tensordicts.append(
                    tensordict.to(self.storing_device, non_blocking=True)
                )

                self._update_traj_ids(tensordict)
                if (
                    self.interruptor is not None
                    and self.interruptor.collection_stopped()
                ):
                    try:
                        torch.stack(
                            tensordicts,
                            self._tensordict_out.ndim - 1,
                            out=self._tensordict_out[: t + 1],
                        )
                    except RuntimeError:
                        with self._tensordict_out.unlock_():
                            torch.stack(
                                tensordicts,
                                self._tensordict_out.ndim - 1,
                                out=self._tensordict_out[: t + 1],
                            )
                    break
            else:
                try:
                    self._tensordict_out = torch.stack(
                        tensordicts,
                        self._tensordict_out.ndim - 1,
                        out=self._tensordict_out,
                    )
                except RuntimeError:
                    with self._tensordict_out.unlock_():
                        self._tensordict_out = torch.stack(
                            tensordicts,
                            self._tensordict_out.ndim - 1,
                            out=self._tensordict_out,
                        )
        return self._tensordict_out

class Parameters():
    def __init__(self,
                # General parameters
                n_agents: int = 4,       # Number of agents
                dt: float = 0.05,           # [s] sample time
                device: str = "cpu",         # Tensor device
                scenario_name: str = "road_traffic",
                
                # Training parameters
                n_iters: int = 250,            # Number of iterations
                frames_per_batch: int = 2**12,
                num_epochs: int = 60,
                minibatch_size: int = 2**9,
                lr: float = 2e-4,               # Learning rate
                lr_min: float = 1e-5,           # Minimum learning rate (used for scheduling of learning rate)
                max_grad_norm: float = 1.0,
                clip_epsilon: float = 0.2,
                gamma: float = 0.99,
                lmbda: float = 0.9,
                entropy_eps: float = 1e-4,
                max_steps: int = 2**7,
                total_frames: int = None,
                num_vmas_envs: int = None,      # Number of vectorized environments
                training_strategy: str = "4",
                
                episode_reward_mean_current: float = 0.00,  # Achieved mean episode reward (total/n_agents)
                episode_reward_intermidiate: float = -1e3, # A arbitrary, small initial value
                
                # Observation
                is_partial_observation: bool = True,
                is_global_coordinate_sys: bool = False,      # Global or local coordinate system
                n_points_short_term: int = 3,            # Number of points that build a short-term reference path
                n_nearing_agents_observed: int = 2,      # Number of nearing agents to be observed (consider limited sensor range)
                n_nearing_obstacles_observed: int = 4,   # Number of nearing obstacles to be observed (consider limited sensor range)
                is_observe_corners: bool = False,            # If True, corners of agents/obstacles will be observed; otherwise, the center point and rotation angle.

                is_testing_mode: bool = False,               # In testing mode, collisions do not terminate the current simulation
                is_visualize_short_term_path: bool = True,
                
                
                # Save/Load
                is_save_intermidiate_model: bool = True,
                is_load_model: bool = False,
                mode_name: str = None,
                where_to_save: str = "outputs/",
                is_continue_train: bool = False,             # Whether to continue training after loading an offline model
                is_save_eval_results: bool = True,
                is_load_out_td: bool = False,
                
                is_real_time_rendering: bool = False,        # Simulation will be paused at each time step for a certain duration to enable real-time rendering
                is_prb: bool = False,

                ############################################
                ## Only for path-tracking scenario
                ############################################
                is_mixed_scenario_training: bool = True,    # Whether to use mixed scenarios durining training
                is_use_intermediate_goals: bool = False,     # If True, intermediate goals will be used, serving as reward shaping; otherwise, only a final goal will be used
                path_tracking_type: str = "sine",             # For path-tracking scenarios
                is_dynamic_goal_reward: bool = False,        # TODO Adjust the goal reward based on how well agents achieve their goals
                
                ############################################
                ## Only for obstacle-avoidance scenario
                ############################################
                obstacle_type: str = "static",                  # For obstacle-avoidance scenarios
                ):
        
        self.n_agents = n_agents
        self.dt = dt
        
        self.device = device
        self.scenario_name = scenario_name
        
        # Sampling
        self.n_iters = n_iters
        self.frames_per_batch = frames_per_batch
        
        if (frames_per_batch is not None) and (n_iters is not None):
            self.total_frames = frames_per_batch * n_iters

        # Training
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.lr = lr
        self.lr_min = lr_min
        self.max_grad_norm = max_grad_norm
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_eps = entropy_eps
        self.max_steps = max_steps
        self.training_strategy = training_strategy
        
        # Scenario parameters
        self.is_mixed_scenario_training = is_mixed_scenario_training
        
        if (frames_per_batch is not None) and (max_steps is not None):
            self.num_vmas_envs = frames_per_batch // max_steps # Number of vectorized envs. frames_per_batch should be divisible by this number,

        self.is_save_intermidiate_model = is_save_intermidiate_model
        self.is_load_model = is_load_model        
        
        self.episode_reward_mean_current = episode_reward_mean_current
        self.episode_reward_intermidiate = episode_reward_intermidiate
        self.where_to_save = where_to_save
        self.is_continue_train = is_continue_train

        # Observation
        self.is_partial_observation = is_partial_observation
        self.is_global_coordinate_sys = is_global_coordinate_sys
        self.n_points_short_term = n_points_short_term
        self.is_use_intermediate_goals = is_use_intermediate_goals
        self.n_nearing_agents_observed = n_nearing_agents_observed
        self.n_nearing_obstacles_observed = n_nearing_obstacles_observed
        self.is_observe_corners = is_observe_corners
        
        self.is_testing_mode = is_testing_mode
        self.is_visualize_short_term_path = is_visualize_short_term_path
        
        self.path_tracking_type = path_tracking_type
        self.is_dynamic_goal_reward = is_dynamic_goal_reward
        
        self.obstacle_type = obstacle_type

        self.is_save_eval_results = is_save_eval_results
        self.is_load_out_td = is_load_out_td
            
        self.is_real_time_rendering = is_real_time_rendering

        self.is_prb = is_prb
        
        if (mode_name is None) and (scenario_name is not None):
            self.mode_name = get_model_name(self)
            
            
    def to_dict(self):
        # Create a dictionary representation of the instance
        return self.__dict__

    @classmethod
    def from_dict(cls, dict_data):
        # Create an instance of the class from a dictionary
        return cls(**dict_data)

class SaveData():
    def __init__(self, parameters: Parameters, episode_reward_mean_list: [] = None):
        self.parameters = parameters
        self.episode_reward_mean_list = episode_reward_mean_list
    def to_dict(self):
        return {
            'parameters': self.parameters.to_dict(),  # Convert Parameters instance to dict
            'episode_reward_mean_list': self.episode_reward_mean_list
        }
    @classmethod
    def from_dict(cls, dict_data):
        parameters = Parameters.from_dict(dict_data['parameters'])  # Convert dict back to Parameters instance
        return cls(parameters, dict_data['episode_reward_mean_list'])



##################################################
## Helper Functions
##################################################
def get_path_to_save_model(parameters: Parameters):
    parameters.mode_name = get_model_name(parameters=parameters)
    
    PATH_POLICY = parameters.where_to_save + parameters.mode_name + "_policy.pth"
    PATH_CRITIC = parameters.where_to_save + parameters.mode_name + "_critic.pth"
    PATH_FIG = parameters.where_to_save + parameters.mode_name + "_training_process.pdf"
    PATH_JSON = parameters.where_to_save + parameters.mode_name + "_data.json"
    
    return PATH_POLICY, PATH_CRITIC, PATH_FIG, PATH_JSON

def delete_model_with_lower_mean_reward(parameters:Parameters):
    # Regular expression pattern to match and capture the float number
    pattern = r'reward(-?[0-9]*\.?[0-9]+)_'

    # Iterate over files in the directory
    for file_name in os.listdir(parameters.where_to_save):
        match = re.search(pattern, file_name)
        if match:
            # Get the achieved mean episode reward of the saved model
            episode_reward_mean = float(match.group(1))
            if episode_reward_mean < parameters.episode_reward_intermidiate:
                # Delete the saved model if its performance is worse
                os.remove(os.path.join(parameters.where_to_save, file_name))

def find_the_hightest_reward_among_all_models(path):
    """This function returns the hightest reward of the models stored in folder `parameters.where_to_save`"""
    # Initialize variables to track the highest reward and corresponding model
    highest_reward = float('-inf')
    
    pattern = r'reward(-?[0-9]*\.?[0-9]+)_'
    # Iterate through the files in the directory
    for filename in os.listdir(path):
        match = re.search(pattern, filename)
        if match:
            # Extract the reward and convert it to float
            episode_reward_mean = float(match.group(1))
            
            # Check if this reward is higher than the current highest
            if episode_reward_mean > highest_reward:
                highest_reward = episode_reward_mean # Update
                 
    return highest_reward


def save(parameters: Parameters, save_data: SaveData, policy=None, critic=None):
    # Get paths
    PATH_POLICY, PATH_CRITIC, PATH_FIG, PATH_JSON = get_path_to_save_model(parameters=parameters)
    
    # Save parameters and mean episode reward list
    json_object = json.dumps(save_data.to_dict(), indent=4) # Serializing json
    with open(PATH_JSON, "w") as outfile: # Writing to sample.json
        outfile.write(json_object)
    # Example to how to open the saved json file
    # with open('save_data.json', 'r') as file:
    #     data = json.load(file)
    #     loaded_parameters = Parameters.from_dict(data)

    # Save figure
    plt.clf()  # Clear the current figure to avoid drawing on the same figure in the next iteration
    plt.plot(save_data.episode_reward_mean_list)
    plt.xlabel("Training iterations")
    plt.ylabel("Episode reward mean")
    plt.tight_layout() # Set the layout to be tight to minimize white space !!! deprecated
    plt.savefig(PATH_FIG, format="pdf", bbox_inches="tight")
    # plt.savefig(PATH_FIG, format="pdf")

    # Save models
    if (policy != None) & (critic != None): 
        # Delete models with lower mean episode reward
        delete_model_with_lower_mean_reward(parameters=parameters)
        # Save current models
        torch.save(policy.state_dict(), PATH_POLICY)
        torch.save(critic.state_dict(), PATH_CRITIC)
        
    print(f"Saved model: {parameters.episode_reward_mean_current:.2f}.")
