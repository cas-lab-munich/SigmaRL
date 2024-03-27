# Adapted from https://pytorch.org/rl/tutorials/multiagent_ppo.html

# Torch
import torch

# Enable anomaly detection
# torch.autograd.set_detect_anomaly(True)

# Tensordict modules
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

# Data collection
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Env
from torchrl.envs import RewardSum
from torchrl.envs.utils import (
    check_env_specs,
)
# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

# Loss
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# Utils
from tqdm import tqdm

import os, sys

import matplotlib.pyplot as plt

# Scientific plotting
import scienceplots # Do not remove (https://github.com/garrettj403/SciencePlots)
plt.rcParams.update({'figure.dpi': '100'}) # Avoid DPI problem (https://github.com/garrettj403/SciencePlots/issues/60)
plt.style.use(['science','ieee']) # The science + ieee styles for IEEE papers (can also be one of 'ieee' and 'science' )
# print(plt.style.available) # List all available style

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

# Import custom classes
from utilities.helper_training import Parameters, SaveData, VmasEnvCustom, SyncDataCollectorCustom, TransformedEnvCustom, get_path_to_save_model, find_the_hightest_reward_among_all_models, save
from utilities.evaluation import evaluate_outputs

from scenarios.car_like_robots_road_traffic_1 import ScenarioRoadTraffic
from scenarios.car_like_robots_path_tracking import ScenarioPathTracking
from scenarios.car_like_robots_obstacle_avoidance import ScenarioObstacleAvoidance 


# Reproducibility
torch.manual_seed(0)


def multiagent_ppo_cavs(parameters: Parameters):
    if "road_traffic" in parameters.scenario_name:
        scenario = ScenarioRoadTraffic()
    elif "path_tracking" in parameters.scenario_name:
        scenario = ScenarioPathTracking()
    elif "obstacle_avoidance" in parameters.scenario_name:
        scenario = ScenarioObstacleAvoidance()
    else:
        raise ValueError(f"The given scenario '{parameters.scenario_name}' is not found. Current implementation includes 'car_like_robots_road_traffic' and 'car_like_robots_path_tracking'.")
    
    scenario.parameters = parameters
    
    env = VmasEnvCustom(
        scenario=scenario,
        num_envs=parameters.num_vmas_envs,
        continuous_actions=True,  # VMAS supports both continuous and discrete actions
        max_steps=parameters.max_steps,
        device=parameters.device,
        # Scenario kwargs
        n_agents=parameters.n_agents,  # These are custom kwargs that change for each VMAS scenario, see the VMAS repo to know more.
    )


    
    save_data = SaveData(
        parameters=parameters,
        episode_reward_mean_list=[],
    )


    # print("env.full_action_spec:", env.full_action_spec, "\n")
    # print("env.full_reward_spec:", env.full_reward_spec, "\n")
    # print("env.full_done_spec:", env.full_done_spec, "\n")
    # print("env.observation_spec:", env.observation_spec, "\n")

    # print("env.action_keys:", env.action_keys, "\n")
    # print("env.reward_keys:", env.reward_keys, "\n")
    # print("env.done_keys:", env.done_keys, "\n")

    env = TransformedEnvCustom(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )

    check_env_specs(env)

    # n_rollout_steps = 5
    # rollout = env.rollout(n_rollout_steps)
    # print("rollout of three steps:", rollout, "\n")
    # print("Shape of the rollout TensorDict:", rollout.batch_size, "\n")

    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[
                -1
            ],  # n_obs_per_agent
            n_agent_outputs=(2 * env.action_spec.shape[-1]),  # 2 * n_actions_per_agents
            n_agents=env.n_agents,
            centralised=False,  # the policies are decentralised (ie each agent will act from its observation)
            share_params=True, # sharing parameters means that agents will all share the same policy, which will allow them to benefit from each other’s experiences, resulting in faster training. On the other hand, it will make them behaviorally homogenous, as they will share the same model
            device=parameters.device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ),
        NormalParamExtractor(),  # this will just separate the last dimension into two outputs: a `loc` and a non-negative `scale``, used as parameters for a normal distribution (mean and standard deviation)
    )


    print("policy_net:", policy_net, "\n")

    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")], # represents the parameters of the policy distribution for each agent
    )

    # Use a probabilistic actor allows for exploration
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.unbatched_action_spec,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": env.unbatched_action_spec[env.action_key].space.low,
            "max": env.unbatched_action_spec[env.action_key].space.high,
        },
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"), # log probability favors numerical stability and gradient calculation
    )  # we'll need the log-prob for the PPO loss

    mappo = True  # IPPO (Independent PPO) if False

    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1], # Number of observations
        # n_agent_inputs=
        #                 env.observation_spec["agents", "info", "pri"].shape[-1]
        #                 + env.observation_spec["agents", "info", "pos"].shape[-1]
        #                 + env.observation_spec["agents", "info", "rot"].shape[-1]
        #                 + env.observation_spec["agents", "info", "vel"].shape[-1]
        #                 + env.observation_spec["agents", "info", "act_vel"].shape[-1]
        #                 + env.observation_spec["agents", "info", "act_steer"].shape[-1]
        #                 + env.observation_spec["agents", "info", "ref"].shape[-1]  # TODO Check if refefrence paths are needed for the critic
        #                 # + env.observation_spec["agents", "info", "distance_ref"].shape[-1]
        #                 # + env.observation_spec["agents", "info", "distance_left_b"].shape[-1]
        #                 # + env.observation_spec["agents", "info", "distance_right_b"].shape[-1]
        #                 ,
        n_agent_outputs=1,  # 1 value per agent
        n_agents=env.n_agents,
        centralised=mappo, # If `centralised` is True (which may help overcome the non-stationary problem in MARL), each agent will use the inputs of all agents to compute its output (n_agent_inputs * n_agents will be the number of inputs for one agent). Otherwise, each agent will only use its data as input.
        share_params=True, # If `share_params` is True, the same MLP will be used to make the forward pass for all agents (homogeneous policies). Otherwise, each agent will use a different MLP to process its input (heterogeneous policies).
        device=parameters.device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )
    # print(critic_net)
    critic = TensorDictModule(
        module=critic_net,
        in_keys=[("agents", "observation")], # Note that the critic in PPO only takes the same inputs (observations) as the actor
        # in_keys=
        # [
        #     ("agents", "info", "pri"),
        #     ("agents", "info", "pos"),
        #     ("agents", "info", "rot"),
        #     ("agents", "info", "vel"),
        #     ("agents", "info", "act_steer"),
        #     ("agents", "info", "act_vel"),
        #     ("agents", "info", "ref"),
        #     # ("agents", "info", "distance_ref"),
        #     # ("agents", "info", "distance_left_b"),
        #     # ("agents", "info", "distance_right_b")
        # ], # Different observations for the critic
        out_keys=[("agents", "state_value")],
    )

    print("critic_net:", critic_net, "\n")
    # print("Running policy:", policy(env.reset()), "\n")
    # print("Running value:", critic(env.reset()), "\n")


    # Check if the directory defined to store the model exists and create it if not
    if not os.path.exists(parameters.where_to_save):
        os.makedirs(parameters.where_to_save)
        print(f"A new directory ({parameters.where_to_save}) to save the trained models has been created.")
        
    # Specify a path
     
    
    # Load an existing model or train a new model?
    if parameters.is_load_model:
        # Load the model with the hightest reward in the folder `parameters.where_to_save`
        highest_reward = find_the_hightest_reward_among_all_models(parameters=parameters)
        parameters.episode_reward_mean_current = highest_reward # Update the parameter so that the right filename will be returned later on 
        if highest_reward is not float('-inf'):
            print("Offline model exists and will be loaded.")
            PATH_POLICY, PATH_CRITIC, PATH_FIG, PATH_JSON = get_path_to_save_model(parameters=parameters)
            # Load the saved model state dictionaries
            policy.load_state_dict(torch.load(PATH_POLICY))
        else:
            raise ValueError("There is no model stored in '{parameters.where_to_save}', or the model names stored here are not following the right pattern.")

        if not parameters.is_continue_train:
            print("Training will not continue.")
            return env, policy, parameters
        else:
            print("Training will continue with the loaded model.")
            critic.load_state_dict(torch.load(PATH_CRITIC))

    # collector = SyncDataCollector(
    #     env,
    #     policy,
    #     device=parameters.device,
    #     storing_device=parameters.device,
    #     frames_per_batch=parameters.frames_per_batch,
    #     total_frames=total_frames,
    # )

    collector = SyncDataCollectorCustom(
        env,
        policy,
        device=parameters.device,
        storing_device=parameters.device,
        frames_per_batch=parameters.frames_per_batch,
        total_frames=parameters.total_frames,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            parameters.frames_per_batch, device=parameters.device
        ),  # We store the frames_per_batch collected at each iteration
        sampler=SamplerWithoutReplacement(),
        batch_size=parameters.minibatch_size,  # We will sample minibatches of this size
    )


    loss_module = ClipPPOLoss(
        actor=policy,
        critic=critic,
        clip_epsilon=parameters.clip_epsilon,
        entropy_coef=parameters.entropy_eps,
        normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
    )
    loss_module.set_keys(  # We have to tell the loss where to find the keys
        reward=env.reward_key,
        action=env.action_key,
        sample_log_prob=("agents", "sample_log_prob"),
        value=("agents", "state_value"),
        # These last 2 keys will be expanded to match the reward shape
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )


    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=parameters.gamma, lmbda=parameters.lmbda
    )  # We build GAE
    GAE = loss_module.value_estimator # Generalized Advantage Estimation 

    optim = torch.optim.Adam(loss_module.parameters(), parameters.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, parameters.gamma=0.1)


    pbar = tqdm(total=parameters.n_iters, desc="episode_reward_mean = 0")

    episode_reward_mean_list = []
    for tensordict_data in collector:
        tensordict_data.set(
            ("next", "agents", "done"),
            tensordict_data.get(("next", "done"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
        )
        tensordict_data.set(
            ("next", "agents", "terminated"),
            tensordict_data.get(("next", "terminated"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
        )
        # We need to expand the done and terminated to match the reward shape (this is expected by the value estimator)

        with torch.no_grad():
            GAE(
                tensordict_data,
                params=loss_module.critic_params,
                target_params=loss_module.target_critic_params,
            )  # Compute GAE and add it to the data

        data_view = tensordict_data.reshape(-1)  # Flatten the batch size to shuffle data
        replay_buffer.extend(data_view)

        for _ in range(parameters.num_epochs):
            # print("[DEBUG] for _ in range(parameters.num_epochs):")
            for _ in range(parameters.frames_per_batch // parameters.minibatch_size):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                
                # print(loss_value)
                
                assert not loss_value.isnan().any()
                assert not loss_value.isinf().any()

                loss_value.backward()

                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), parameters.max_grad_norm
                )  # Optional

                # print("[DEBUG] optim.step()")
                optim.step()
                optim.zero_grad()

        collector.update_policy_weights_()

        # Logging
        done = tensordict_data.get(("next", "agents", "done"))
        episode_reward_mean = (
            tensordict_data.get(("next", "agents", "episode_reward"))[done].mean().item()
        )
        episode_reward_mean_list.append(episode_reward_mean)
        pbar.set_description(f"episode_reward_mean = {episode_reward_mean:.3f}", refresh=False)

        env.test1 = pbar.n
        env.test2 = episode_reward_mean
        
        if parameters.is_save_intermidiate_model:
            # Update the current mean episode reward
            parameters.episode_reward_mean_current = episode_reward_mean
            save_data.episode_reward_mean_list = episode_reward_mean_list

            if episode_reward_mean > parameters.episode_reward_intermidiate:
                # Save the model if it improves the mean episode reward sufficiently enough
                save(parameters=parameters, save_data=save_data, policy=policy, critic=critic)
                # Update the episode reward of the saved model
                parameters.episode_reward_intermidiate = episode_reward_mean
            else:
                # Save only the mean episode reward list and parameters
                parameters.episode_reward_mean_current = parameters.episode_reward_intermidiate
                save(parameters=parameters, save_data=save_data, policy=None, critic=None)

            # print("Fig saved.")

        # Learning rate schedule
        for param_group in optim.param_groups:
            # Linear decay to lr_min
            lr_decay = (parameters.lr - parameters.lr_min) * (1 - (pbar.n / parameters.n_iters))
            param_group['lr'] = parameters.lr_min + lr_decay
            if (pbar.n % 10 == 0):
                print(f"Learning rate updated to {param_group['lr']}.")
                
        pbar.update()
        
    # Save the final model
    if not parameters.is_save_intermidiate_model:
        # Update the current mean episode reward
        parameters.episode_reward_mean_current = episode_reward_mean
        save_data.episode_reward_mean_list = episode_reward_mean_list

        save(parameters=parameters, save_data=save_data, policy=policy, critic=critic)
        print("Final model saved.")
        
    print(f"All files have been saved under {parameters.where_to_save + parameters.mode_name}.")
    # plt.show()
    
    return env, policy, parameters


if __name__ == "__main__":
    scenario_name = "car_like_robots_road_traffic" # car_like_robots_road_traffic, car_like_robots_path_tracking, car_like_robots_obstacle_avoidance
    
    parameters = Parameters(
        n_agents=10,
        dt=0.05, # [s] sample time 
        device="cpu" if not torch.backends.cuda.is_built() else "cuda:0",  # The divice where learning is run
        scenario_name=scenario_name,
        
        # Training parameters
        n_iters=500, # Number of sampling and training iterations (on-policy: rollouts are collected during sampling phase, which will be immediately used in the training phase of the same iteration),
        frames_per_batch=2**12, # Number of team frames collected per training iteration 
                                # num_envs = frames_per_batch / max_steps
                                # total_frames = frames_per_batch * n_iters
                                # sub_batch_size = frames_per_batch // minibatch_size
        num_epochs=60, # Optimization steps per batch of data collected,
        minibatch_size=2**9, # Size of the mini-batches in each optimization step (2**9 - 2**12?),
        lr=2e-4, # Learning rate,
        lr_min=1e-5, # Learning rate,
        max_grad_norm=1.0, # Maximum norm for the gradients,
        clip_epsilon=0.2, # clip value for PPO loss,
        gamma=0.99, # discount factor (empirical formula: 0.1 = gamma^t, where t is the number of future steps that you want your agents to predict {0.96 -> 56 steps, 0.98 -> 114 steps, 0.99 -> 229 steps, 0.995 -> 459 steps})
        lmbda=0.9, # lambda for generalised advantage estimation,
        entropy_eps=1e-4, # coefficient of the entropy term in the PPO loss,
        max_steps=2**7, # Episode steps before done
        training_strategy='1', # One of {'1', '2', '3', '4'}
        
        is_save_intermidiate_model=True, # Is this is true, the model with the hightest mean episode reward will be saved,
        
        episode_reward_mean_current=0.00,
        
        is_load_model=False, # Load offline model if available. The offline model in `where_to_save` whose name contains `episode_reward_mean_current` will be loaded
        is_continue_train=False, # If offline models are loaded, whether to continue to train the model
        mode_name=None, 
        episode_reward_intermidiate=-1e3, # The initial value should be samll enough
        
        where_to_save=f"outputs/{scenario_name}_ppo/mixed_training_0327_whole_map/", # folder where to save the trained models, fig, data, etc.

        # Scenario parameters
        is_partial_observation=True,
        is_global_coordinate_sys=False,
        n_points_short_term=3,
        is_use_intermediate_goals=False,
        n_nearing_agents_observed=2,
        n_nearing_obstacles_observed=4,
        
        is_testing_mode=False,
        is_visualize_short_term_path=True,
        
        is_save_eval_results=True,
        
        ############################################
        # For car_like_robots_path_tracking only
        ############################################
        path_tracking_type='sine', # [relevant to path-tracking scenarios] should be one of 'line', 'turning', 'circle', 'sine', and 'horizontal_8'
        obstacle_type="static", # [relevant for obstacle-avoidance scenarios] should be one of "static" and "dynamic"
        is_mixed_scenario_training = True,
        is_dynamic_goal_reward=False, # set to True if the goal reward is dynamically adjusted based on the performance of agents' history trajectories 

        ############################################
        # For car_like_robots_obstacle_avoidance only
        ############################################
        is_observe_corners=False,
    )
        
    env, policy, parameters = multiagent_ppo_cavs(parameters=parameters)

    # Evaluate the model
    with torch.no_grad():
        out_td = env.rollout(
            max_steps=parameters.max_steps*4,
            policy=policy,
            callback=lambda env, _: env.render(),
            auto_cast_to_device=True,
            break_when_any_done=True,
        )
    # evaluate_outputs(out_td=out_td, parameters=parameters, agent_width=env.scenario.world.agents[0].shape.width)
    
