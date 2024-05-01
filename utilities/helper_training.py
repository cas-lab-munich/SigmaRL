import torch

# Tensordict modules
from tensordict.tensordict import TensorDictBase, TensorDict

# Data collection
from torchrl.collectors import SyncDataCollector

# Env
from torchrl.envs import TransformedEnv
from torchrl.envs.utils import (
    set_exploration_type,
    _terminated_or_truncated,
    step_mdp,
)
from torchrl.envs.libs.vmas import VmasEnv

from vmas.simulator.utils import (
    X,
    Y,
)

# Utils
from matplotlib import pyplot as plt
from typing import Callable, Optional, Tuple, Callable, Optional, Union
from ctypes import byref

from matplotlib import pyplot as plt
import json
import os
import re

def get_model_name(parameters):
    # model_name = f"nags{parameters.n_agents}_it{parameters.n_iters}_fpb{parameters.frames_per_batch}_tfrms{parameters.total_frames}_neps{parameters.num_epochs}_mnbsz{parameters.minibatch_size}_lr{parameters.lr}_mgn{parameters.max_grad_norm}_clp{parameters.clip_epsilon}_gm{parameters.gamma}_lmbda{parameters.lmbda}_etp{parameters.entropy_eps}_mstp{parameters.max_steps}_nenvs{parameters.num_vmas_envs}"
    model_name = f"reward{parameters.episode_reward_mean_current:.2f}"

    return model_name

##################################################
## Custom Classes
##################################################  
class TransformedEnvCustom(TransformedEnv):
    """
    Slightly modify the function `rollout`, `_rollout_stop_early`, and `_rollout_nonstop` to enable returning a frame list to save evaluation video
    """
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
        is_save_simulation_video: bool = False,
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
        # print("[DEBUG] new env.rollout")
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
            "is_save_simulation_video": is_save_simulation_video,
        }
        if break_when_any_done:
            if is_save_simulation_video:
                tensordicts, frame_list = self._rollout_stop_early(**kwargs)
            else:
                tensordicts = self._rollout_stop_early(**kwargs)
        else:
            if is_save_simulation_video:
                tensordicts, frame_list = self._rollout_nonstop(**kwargs)
            else:
                tensordicts = self._rollout_nonstop(**kwargs)
                
        batch_size = self.batch_size if tensordict is None else tensordict.batch_size
        out_td = torch.stack(tensordicts, len(batch_size), out=out)
        if return_contiguous:
            out_td = out_td.contiguous()
        out_td.refine_names(..., "time")
        
        if is_save_simulation_video:
            return out_td, frame_list
        else:
            return out_td
        
    def _rollout_stop_early(
        self,
        *,
        tensordict,
        auto_cast_to_device,
        max_steps,
        policy,
        policy_device,
        env_device,
        callback,
        is_save_simulation_video,
    ):
        tensordicts = []
        
        if is_save_simulation_video:
            frame_list = []
            
        for i in range(max_steps):
            if auto_cast_to_device:
                tensordict = tensordict.to(policy_device, non_blocking=True)
            tensordict = policy(tensordict)
            if auto_cast_to_device:
                tensordict = tensordict.to(env_device, non_blocking=True)
            tensordict = self.step(tensordict)
            tensordicts.append(tensordict.clone(False))

            if i == max_steps - 1:
                # we don't truncated as one could potentially continue the run
                break
            tensordict = step_mdp(
                tensordict,
                keep_other=True,
                exclude_action=False,
                exclude_reward=True,
                reward_keys=self.reward_keys,
                action_keys=self.action_keys,
                done_keys=self.done_keys,
            )
            # done and truncated are in done_keys
            # We read if any key is done.
            any_done = _terminated_or_truncated(
                tensordict,
                full_done_spec=self.output_spec["full_done_spec"],
                key=None,
            )
            if any_done:
                break

            if callback is not None:
                if is_save_simulation_video:
                    frame = callback(self, tensordict)
                    frame_list.append(frame)
                else:
                    callback(self, tensordict)
                
        if is_save_simulation_video:
            return tensordicts, frame_list
        else:
            return tensordicts

    def _rollout_nonstop(
        self,
        *,
        tensordict,
        auto_cast_to_device,
        max_steps,
        policy,
        policy_device,
        env_device,
        callback,
        is_save_simulation_video,
    ):
        tensordicts = []
        tensordict_ = tensordict

        if is_save_simulation_video:
            frame_list = []
            
        for i in range(max_steps):
            if auto_cast_to_device:
                tensordict_ = tensordict_.to(policy_device, non_blocking=True)
            tensordict_ = policy(tensordict_)
            if auto_cast_to_device:
                tensordict_ = tensordict_.to(env_device, non_blocking=True)
            tensordict, tensordict_ = self.step_and_maybe_reset(tensordict_)
            tensordicts.append(tensordict)
            if i == max_steps - 1:
                # we don't truncated as one could potentially continue the run
                break

            if callback is not None:
                if is_save_simulation_video:
                    frame = callback(self, tensordict)
                    frame_list.append(frame)
                else:
                    callback(self, tensordict)
                    
        if is_save_simulation_video:
            return tensordicts, frame_list
        else:
            return tensordicts
        
class Parameters():
    def __init__(self,
                # General parameters
                n_agents: int = 4,          # Number of agents
                dt: float = 0.05,           # [s] sample time
                device: str = "cpu",        # Tensor device
                scenario_name: str = "road_traffic",    # Scenario name
                
                # Training parameters
                n_iters: int = 250,             # Number of training iterations
                frames_per_batch: int = 2**12, # Number of team frames collected per training iteration 
                                            # num_envs = frames_per_batch / max_steps
                                            # total_frames = frames_per_batch * n_iters
                                            # sub_batch_size = frames_per_batch // minibatch_size
                num_epochs: int = 60,       # Optimization steps per batch of data collected
                minibatch_size: int = 2**9,     # Size of the mini-batches in each optimization step (2**9 - 2**12?)
                lr: float = 2e-4,               # Learning rate
                lr_min: float = 1e-5,           # Minimum learning rate (used for scheduling of learning rate)
                max_grad_norm: float = 1.0,     # Maximum norm for the gradients
                clip_epsilon: float = 0.2,      # Clip value for PPO loss
                gamma: float = 0.99,            # Discount factor from 0 to 1. A greater value corresponds to a better farsight
                lmbda: float = 0.9,             # lambda for generalised advantage estimation
                entropy_eps: float = 1e-4,      # Coefficient of the entropy term in the PPO loss
                max_steps: int = 2**7,          # Episode steps before done
                total_frames: int = None,       # Total frame for one training, equals `frames_per_batch * n_iters`
                num_vmas_envs: int = None,      # Number of vectorized environments
                training_strategy: str = "4",  # One of {'1', '2', '3', '4'}. 
                                            # 1 for vanilla
                                            # 2 for vanilla with prioritized replay buffer
                                            # 3 for vanilla with challenging initial state buffer
                                            # 4 for mixed training
                episode_reward_mean_current: float = 0.00,  # Achieved mean episode reward (total/n_agents)
                episode_reward_intermediate: float = -1e3, # A arbitrary, small initial value
                
                is_prb: bool = False,       # # Whether to enable prioritized replay buffer
                scenario_probabilities = [1.0, 0.0, 0.0], # Probabilities of training agents in intersection, merge-in, or merge-out scenario
                
                # Observation
                n_points_short_term: int = 3,            # Number of points that build a short-term reference path

                is_partial_observation: bool = True,     # Whether to enable partial observation
                n_nearing_agents_observed: int = 2,      # Number of nearing agents to be observed (consider limited sensor range)

                # Parameters for ablation studies
                is_ego_view: bool = True,                           # Ego view or bird view
                is_apply_mask: bool = True,                         # Whether to mask distant agents
                is_observe_distance_to_agents: bool = True,         # Whether to observe the distance to other agents
                is_observe_distance_to_boundaries: bool = True,     # Whether to observe points on lanelet boundaries or observe the distance to labelet boundaries
                is_observe_distance_to_center_line: bool = True,    # Whether to observe the distance to reference path
                is_observe_vertices: bool = True,                         # Whether to observe the vertices of other agents (or center point)
                
                is_add_noise: bool = True,                          # Whether to add noise to observations
                is_observe_ref_path_other_agents: bool = False,     # Whether to observe the reference paths of other agents
                is_use_mtv_distance: bool = True,           # Whether to use MTV-based (Minimum Translation Vector) distance or c2c-based (center-to-center) distance.
                
                # Visu
                is_visualize_short_term_path: bool = True,  # Whether to visualize short-term reference paths
                is_visualize_lane_boundary: bool = False,   # Whether to visualize lane boundary
                is_real_time_rendering: bool = False,       # Simulation will be paused at each time step for a certain duration to enable real-time rendering
                is_visualize_extra_info: bool = True,       # Whether to render extra information such time and time step
                render_title: str = "",                     # The title to be rendered

                # Save/Load
                is_save_intermediate_model: bool = True,    # Whether to save intermediate model (also called checkpoint) with the hightest episode reward
                is_load_model: bool = False,                # Whether to load saved model
                is_load_final_model: bool = False,          # Whether to load the final model (last iteration)
                mode_name: str = None,
                where_to_save: str = "outputs/",            # Define where to save files such as intermediate models
                is_continue_train: bool = False,            # Whether to continue training after loading an offline model
                is_save_eval_results: bool = True,          # Whether to save evaluation results such as figures and evaluation outputs
                is_load_out_td: bool = False,               # Whether to load evaluation outputs
                
                is_testing_mode: bool = False,              # In testing mode, collisions do not terminate the current simulation
                is_save_simulation_video: bool = False,     # Whether to save simulation videos
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
        
        if (frames_per_batch is not None) and (max_steps is not None):
            self.num_vmas_envs = frames_per_batch // max_steps # Number of vectorized envs. frames_per_batch should be divisible by this number,

        self.is_save_intermediate_model = is_save_intermediate_model
        self.is_load_model = is_load_model        
        self.is_load_final_model = is_load_final_model        
        
        self.episode_reward_mean_current = episode_reward_mean_current
        self.episode_reward_intermediate = episode_reward_intermediate
        self.where_to_save = where_to_save
        self.is_continue_train = is_continue_train

        # Observation
        self.is_partial_observation = is_partial_observation
        self.n_points_short_term = n_points_short_term
        self.n_nearing_agents_observed = n_nearing_agents_observed
        self.is_observe_distance_to_agents = is_observe_distance_to_agents
        
        self.is_testing_mode = is_testing_mode
        self.is_save_simulation_video = is_save_simulation_video
        self.is_visualize_short_term_path = is_visualize_short_term_path
        self.is_visualize_lane_boundary = is_visualize_lane_boundary
        
        self.is_ego_view = is_ego_view
        self.is_apply_mask = is_apply_mask
        self.is_use_mtv_distance = is_use_mtv_distance
        self.is_observe_distance_to_boundaries = is_observe_distance_to_boundaries
        self.is_observe_distance_to_center_line = is_observe_distance_to_center_line
        self.is_observe_vertices = is_observe_vertices
        self.is_add_noise = is_add_noise 
        self.is_observe_ref_path_other_agents = is_observe_ref_path_other_agents 

        self.is_save_eval_results = is_save_eval_results
        self.is_load_out_td = is_load_out_td
            
        self.is_real_time_rendering = is_real_time_rendering
        self.is_visualize_extra_info = is_visualize_extra_info
        self.render_title = render_title

        self.is_prb = is_prb
        self.scenario_probabilities = scenario_probabilities
        
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

def delete_files_with_lower_mean_reward(parameters:Parameters):
    # Regular expression pattern to match and capture the float number
    pattern = r'reward(-?[0-9]*\.?[0-9]+)_'

    # Iterate over files in the directory
    for file_name in os.listdir(parameters.where_to_save):
        match = re.search(pattern, file_name)
        if match:
            # Get the achieved mean episode reward of the saved model
            episode_reward_mean = float(match.group(1))
            if episode_reward_mean < parameters.episode_reward_intermediate:
                # Delete the saved model if its performance is worse
                os.remove(os.path.join(parameters.where_to_save, file_name))

def find_the_highest_reward_among_all_models(path):
    """This function returns the highest reward of the models stored in folder `parameters.where_to_save`"""
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
        # Save current models
        torch.save(policy.state_dict(), PATH_POLICY)
        torch.save(critic.state_dict(), PATH_CRITIC)
        # Delete files with lower mean episode reward
        delete_files_with_lower_mean_reward(parameters=parameters)

    print(f"Saved model: {parameters.episode_reward_mean_current:.2f}.")

def compute_td_error(tensordict_data: TensorDict, gamma = 0.9):
    """
    Computes TD error.
    
    Args:
        gamma: discount factor
    """
    current_state_values = tensordict_data["agents"]["state_value"]
    next_rewards = tensordict_data.get(("next", "agents", "reward"))
    next_state_values = tensordict_data.get(("next", "agents", "state_value"))
    td_error = next_rewards + gamma * next_state_values - current_state_values # See Eq. (2) of Section B EXPERIMENTAL DETAILS of paper https://doi.org/10.48550/arXiv.1511.05952
    td_error = td_error.abs() # Magnitude is more interesting than the actual TD error
    
    td_error_average_over_agents = td_error.mean(dim=-2) # Cooperative agents
    
    # Normalize TD error to [-1, 1] (priorities must be positive)
    td_min = td_error_average_over_agents.min()
    td_max = td_error_average_over_agents.max()
    td_error_range = td_max - td_min
    td_error_range = max(td_error_range, 1e-3) # For numerical stability
    
    td_error_average_over_agents = (td_error_average_over_agents - td_min) / td_error_range
    td_error_average_over_agents = torch.clamp(td_error_average_over_agents, 1e-3, 1) # For numerical stability
    
    return td_error_average_over_agents
