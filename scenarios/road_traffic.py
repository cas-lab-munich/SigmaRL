import os
import sys
import time
from termcolor import colored, cprint

# !Important: Add project root to system path if you want to run this file directly
script_dir = os.path.dirname(__file__) # Directory of the current script
project_root = os.path.dirname(script_dir) # Project root directory
if project_root not in sys.path:
    sys.path.append(project_root)
    
import torch
from torch import Tensor
from typing import Dict

# Enable anomaly detection
# torch.autograd.set_detect_anomaly(True)

import matplotlib.pyplot as plt

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, World
from vmas.simulator.scenario import BaseScenario
# from vmas.simulator.dynamics.kinematic_bicycle import KinematicBicycle

from utilities.kinematic_bicycle import KinematicBicycle
from utilities.colors import Color, colors

from utilities.helper_training import Parameters

from utilities.helper_scenario import Distances, Normalizers, Observations, Penalties, ReferencePathsAgentRelated, ReferencePathsMapRelated, Rewards, Thresholds, Collisions, Timer, Constants, CircularBuffer, StateBuffer, InitialStateBuffer, Noise, Evaluation, exponential_decreasing_fcn, get_distances_between_agents, get_perpendicular_distances, get_rectangle_vertices, get_short_term_reference_path, interX, angle_eliminate_two_pi, transform_from_global_to_local_coordinate

from utilities.map_manager import MapManager

from utilities.constants import SCENARIOS

## Simulation parameters 
n_agents = 8                    # The number of agents
dt = 0.05                       # Sample time in [s]
max_steps = 1000                # Maximum simulation steps
is_real_time_rendering = True   # Simulation will be paused at each time step for real-time rendering
agent_max_speed = 1.0           # Maximum allowed speed in [m/s]
agent_max_steering_angle = 35   # Maximum allowed steering angle in degree
agent_mass = 0.5                # The mass of each agent in [kg]

## Geometry
agent_width = 0.08              # The width of the agent in [m]
agent_length = 0.16             # The length of the agent in [m]
wheelbase_front = agent_length / 2                  # Front wheelbase in [m]
wheelbase_rear = agent_length - wheelbase_front     # Rear wheelbase in [m]
lane_width = 0.15               # The (rough) width of each lane in [m]

## Reward
r_p_normalizer = 100    # Rewards and renalties must be normalized to range [-1, 1]

reward_progress = 10 / r_p_normalizer   # Reward for moving along reference paths
reward_vel = 5 / r_p_normalizer         # Reward for moving in high velocities. 
reward_reach_goal = 0 / r_p_normalizer  # Goal-reaching reward

## Penalty
penalty_deviate_from_ref_path = -2 / r_p_normalizer      # Penalty for deviating from reference paths
threshold_deviate_from_ref_path = (lane_width - agent_width) / 2 # Use for penalizing of deviating from reference path
penalty_near_boundary = -20 / r_p_normalizer              # Penalty for being too close to lanelet boundaries
penalty_near_other_agents = -20 / r_p_normalizer          # Penalty for being too close to other agents
penalty_collide_with_agents = -100 / r_p_normalizer       # Penalty for colliding with other agents 
penalty_collide_with_boundaries = -100 / r_p_normalizer   # Penalty for colliding with lanelet boundaries
penalty_change_steering = -2 / r_p_normalizer          # Penalty for changing steering too quick
penalty_time = 5 / r_p_normalizer                      # Penalty for losing time

threshold_reach_goal = agent_width / 2  # Threshold less than which agents are considered at their goal positions

threshold_change_steering = 10 # Threshold above which agents will be penalized for changing steering too quick [degree]

threshold_near_boundary_high = (lane_width - agent_width) / 2 * 0.9    # Threshold beneath which agents will started be 
                                                                        # penalized for being too close to lanelet boundaries
threshold_near_boundary_low = 0 # Threshold above which agents will be penalized for being too close to lanelet boundaries 

threshold_near_other_agents_c2c_high = agent_length + agent_width     # Threshold beneath which agents will started be 
                                                        # penalized for being too close to other agents (for center-to-center distance)
threshold_near_other_agents_c2c_low = (agent_length + agent_width) / 2   # Threshold above which agents will be penalized (for center-to-center distance, 
                                                        # if a c2c distance is less than the half of the agent width, they are colliding, which will be penalized by another penalty)

threshold_near_other_agents_MTV_high = agent_length  # Threshold beneath which agents will be penalized for 
                                                    # being too close to other agents (for MTV-based distance)
threshold_near_other_agents_MTV_low = 0             # Threshold above which agents will be penalized for
                                                    # being too close to other agents (for MTV-based distance)
                                                    
threshold_no_reward_if_too_close_to_boundaries = agent_width / 10
threshold_no_reward_if_too_close_to_other_agents = agent_width / 6

## Visualization
resolution_factor = 200     # Default 200

is_testing_mode = False             # In testing mode, collisions do not lead to the termination of the simulation 
is_visualize_short_term_path = True
is_visualize_extra_info = False
render_title = "Multi-Agent Reinforcement Learning for Connected and Automated Vehicles"

# Reference path
n_points_short_term = 3             # The number of points on short-term reference paths
n_points_nearing_boundary = 5       # The number of points on nearing boundaries to be observed
sample_interval_ref_path = 2                 # Integer, sample interval from the long-term reference path for the short-term reference paths 
max_ref_path_points = 200           # The estimated maximum points on the reference path

## Observation
is_partial_observation = True       # Set to True if each agent can only observe a subset of other agents, i.e., limitations on sensor range are considered
                                    # Note that this also reduces the observation size, which may facilitate training
n_nearing_agents_observed = 3       # The number of most nearing agents to be observed by each agent. This parameter will be used if `is_partial_observation = True`

noise_level = 0.2 * agent_width     # Noise will be generated by the standary normal distribution. This parameter controls the noise level

n_stored_steps = 5      # The number of steps to store (include the current step). At least one
n_observed_steps = 1    # The number of steps to observe (include the current step). At least one, and at most `n_stored_steps`

# Training parameters
scenario_type = "CPM_mixed"  # One of {"CPM_entire", "CPM_mixed", "intersection_1", "design you own map and name it here"}

buffer_size = 100               # Used only when `is_challenging_initial_state_buffer` is True
n_steps_before_recording = 10   # The states of agents at time step `current_time_step - n_steps_before_recording` before collisions will be recorded and used later when resetting the envs
n_steps_stored = n_steps_before_recording # Store previous `n_steps_stored` steps of states
probability_record = 1.0            # Probability of recording a collision-event into the buffer
probability_use_recording = 0.2     # Probability of using an recording when resetting an env

cpm_scenario_probabilities = [1.0, 0.0, 0.0] # 1 for intersection, 2 for merge-in, 3 for merge-out scenario

is_ego_view = True                  # Global coordinate system (bird view) or local coordinate system (ego view)
is_apply_mask = True
is_observe_vertices = True
is_observe_distance_to_agents = False
is_observe_distance_to_boundaries = True # Whether to observe points on lanelet boundaries or the distance to lanelet boundaries
is_observe_distance_to_center_line = True

is_observe_ref_path_other_agents = False
is_use_mtv_distance = False
is_add_noise = True


class ScenarioRoadTraffic(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # print("[DEBUG] make_world() road_traffic")
        # device = torch.device("mps") # For mac with m chip to use GPU acceleration (however, seems not be fully supported by VMAS)
        # Specify parameters if not given
        if not hasattr(self, "parameters"):
            self.parameters = Parameters(
                n_agents=n_agents,
                is_partial_observation=is_partial_observation,
                is_testing_mode=is_testing_mode,
                is_visualize_short_term_path=is_visualize_short_term_path,
                max_steps=max_steps,
                scenario_type=scenario_type,
                n_nearing_agents_observed=n_nearing_agents_observed,
                is_real_time_rendering=is_real_time_rendering,
                n_points_short_term=n_points_short_term,
                dt=dt,

                is_ego_view=is_ego_view,
                is_apply_mask=is_apply_mask,
                is_observe_vertices=is_observe_vertices,
                is_observe_distance_to_agents=is_observe_distance_to_agents,
                is_observe_distance_to_boundaries=is_observe_distance_to_boundaries,
                is_observe_distance_to_center_line=is_observe_distance_to_center_line,
                
                is_use_mtv_distance=is_use_mtv_distance,
                cpm_scenario_probabilities=cpm_scenario_probabilities,
                is_add_noise=is_add_noise,
                is_observe_ref_path_other_agents=is_observe_ref_path_other_agents,
                is_visualize_extra_info=is_visualize_extra_info,
                render_title=render_title,
            )

        self.shared_reward = kwargs.get("shared_reward", False)
        
        width = kwargs.get("width", agent_width)
        l_f = kwargs.get("l_f", wheelbase_front)    # Front wheelbase
        l_r = kwargs.get("l_r", wheelbase_rear)     # Rear wheelbase
        max_steering_angle = kwargs.get("max_steering_angle", torch.deg2rad(torch.tensor(agent_max_steering_angle, device=device, dtype=torch.float32)))
        max_speed = kwargs.get("max_speed", agent_max_speed)
        
        world_x_dim = SCENARIOS[self.parameters.scenario_type]["world_x_dim"]
        world_y_dim = SCENARIOS[self.parameters.scenario_type]["world_y_dim"]
        
        self.render_origin = [
            world_x_dim / 2,
            world_y_dim / 2,
        ]

        self.viewer_size = (int(world_x_dim * resolution_factor), int(world_y_dim * resolution_factor))
        self.viewer_zoom = SCENARIOS[self.parameters.scenario_type]["viewer_zoom"]
        
        # Logs
        if self.parameters.is_testing_mode:
            print(colored(f"[INFO] Testing mode", "red"))
        print(colored(f"[INFO] Scenario type: {self.parameters.scenario_type}", "red"))
        if self.parameters.is_prb:
            print(colored("[INFO] Enable prioritized replay buffer", "red"))
        if self.parameters.is_challenging_initial_state_buffer:
            print(colored("[INFO] Enable challenging initial state buffer", "red"))
            
        self.parameters.n_nearing_agents_observed = min(self.parameters.n_nearing_agents_observed, self.parameters.n_agents - 1)
        self.n_agents = self.parameters.n_agents
        
        # Timer for the first env
        self.timer = Timer(
            start=time.time(),
            end=0,
            step=torch.zeros(batch_dim, device=device, dtype=torch.int32), # Each environment has its own time step
            step_duration=torch.zeros(self.parameters.max_steps, device=device, dtype=torch.float32),
            step_begin=time.time(),
            render_begin=0,
        )
        
        # Make world
        world = World(
            batch_dim,
            device,
            x_semidim=torch.tensor(world_x_dim, device=device, dtype=torch.float32),
            y_semidim=torch.tensor(world_y_dim, device=device, dtype=torch.float32),
            dt=self.parameters.dt
        )
        
        # Get map data
        self.map = MapManager(
            scenario_type = self.parameters.scenario_type, 
            device = device,
        )
        
        cprint("[INFO] Map parsed.", "blue")
        # Determine the maximum number of points on the reference path
        if "CPM_mixed" in self.parameters.scenario_type:
            # Mixed scenarios including intersection, merge in, and merge out
            max_ref_path_points = max([
                ref_p["center_line"].shape[0] for ref_p in self.map.parser.reference_paths_intersection + self.map.parser.reference_paths_merge_in + self.map.parser.reference_paths_merge_out
            ]) + self.parameters.n_points_short_term * sample_interval_ref_path + 2 # Append a smaller buffer
        else:
            # Single scenario
            max_ref_path_points = max([
                ref_p["center_line"].shape[0] for ref_p in self.map.parser.reference_paths
            ]) + self.parameters.n_points_short_term * sample_interval_ref_path + 2 # Append a smaller buffer
            
        # Get all reference paths
        self.ref_paths_map_related = ReferencePathsMapRelated(
            long_term_all=self.map.parser.reference_paths,
            long_term_intersection=self.map.parser.reference_paths_intersection,
            long_term_merge_in=self.map.parser.reference_paths_merge_in,
            long_term_merge_out=self.map.parser.reference_paths_merge_out,
            point_extended_all=torch.zeros((len(self.map.parser.reference_paths), self.parameters.n_points_short_term * sample_interval_ref_path, 2), device=device, dtype=torch.float32), # Not interesting, may be useful in the future
            point_extended_intersection=torch.zeros((len(self.map.parser.reference_paths_intersection), self.parameters.n_points_short_term * sample_interval_ref_path, 2), device=device, dtype=torch.float32),
            point_extended_merge_in=torch.zeros((len(self.map.parser.reference_paths_merge_in), self.parameters.n_points_short_term * sample_interval_ref_path, 2), device=device, dtype=torch.float32),
            point_extended_merge_out=torch.zeros((len(self.map.parser.reference_paths_merge_out), self.parameters.n_points_short_term * sample_interval_ref_path, 2), device=device, dtype=torch.float32),
            sample_interval=torch.tensor(sample_interval_ref_path, device=device, dtype=torch.int32),
        )
        
        # Extended the reference path by several points along the last vector of the center line 
        idx_broadcasting_entend = torch.arange(1, self.parameters.n_points_short_term * sample_interval_ref_path + 1, device=device, dtype=torch.int32).unsqueeze(1)
        for idx, i_path in enumerate(self.map.parser.reference_paths):
            center_line_i = i_path["center_line"]
            direction = center_line_i[-1] - center_line_i[-2]
            self.ref_paths_map_related.point_extended_all[idx, :] = center_line_i[-1] + idx_broadcasting_entend * direction
        for idx, i_path in enumerate(self.map.parser.reference_paths_intersection):
            center_line_i = i_path["center_line"]
            direction = center_line_i[-1] - center_line_i[-2]
            self.ref_paths_map_related.point_extended_intersection[idx, :] = center_line_i[-1] + idx_broadcasting_entend * direction            
        for idx, i_path in enumerate(self.map.parser.reference_paths_merge_in):
            center_line_i = i_path["center_line"]
            direction = center_line_i[-1] - center_line_i[-2]
            self.ref_paths_map_related.point_extended_merge_in[idx, :] = center_line_i[-1] + idx_broadcasting_entend * direction
        for idx, i_path in enumerate(self.map.parser.reference_paths_merge_out):
            center_line_i = i_path["center_line"]
            direction = center_line_i[-1] - center_line_i[-2]
            self.ref_paths_map_related.point_extended_merge_out[idx, :] = center_line_i[-1] + idx_broadcasting_entend * direction
        
        # Initialize agent-specific reference paths, which will be determined in `reset_world_at` function
        self.ref_paths_agent_related = ReferencePathsAgentRelated(
            long_term=torch.zeros((batch_dim, self.n_agents, max_ref_path_points, 2), device=device, dtype=torch.float32), # Long-term reference paths of agents
            long_term_vec_normalized=torch.zeros((batch_dim, self.n_agents, max_ref_path_points, 2), device=device, dtype=torch.float32),
            left_boundary=torch.zeros((batch_dim, self.n_agents, max_ref_path_points, 2), device=device, dtype=torch.float32),
            right_boundary=torch.zeros((batch_dim, self.n_agents, max_ref_path_points, 2), device=device, dtype=torch.float32),
            entry=torch.zeros((batch_dim, self.n_agents, 2, 2), device=device, dtype=torch.float32),
            exit=torch.zeros((batch_dim, self.n_agents, 2, 2), device=device, dtype=torch.float32),
            is_loop=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.bool),
            n_points_long_term=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32),
            n_points_left_b=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32),
            n_points_right_b=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32),
            short_term=torch.zeros((batch_dim, self.n_agents, self.parameters.n_points_short_term, 2), device=device, dtype=torch.float32), # Short-term reference path
            short_term_indices = torch.zeros((batch_dim, self.n_agents, self.parameters.n_points_short_term), device=device, dtype=torch.int32),
            n_points_nearing_boundary=torch.tensor(n_points_nearing_boundary, device=device, dtype=torch.int32),
            nearing_points_left_boundary=torch.zeros((batch_dim, self.n_agents, n_points_nearing_boundary, 2), device=device, dtype=torch.float32), # Nearing left boundary
            nearing_points_right_boundary=torch.zeros((batch_dim, self.n_agents, n_points_nearing_boundary, 2), device=device, dtype=torch.float32), # Nearing right boundary
            scenario_id=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32), # Which scenarios agents are (1 for intersection, 2 for merge-in, 3 for merge-out)
            path_id=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32), # Which paths agents are
            point_id=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32), # Which points agents are
        )        
        
        # The shape of each agent is considered a rectangle with 4 vertices. 
        # The first vertex is repeated at the end to close the shape.
        self.vertices = torch.zeros((batch_dim, self.n_agents, 5, 2), device=device, dtype=torch.float32) 
 
        weighting_ref_directions = torch.linspace(1, 0.2, steps=self.parameters.n_points_short_term, device=device, dtype=torch.float32)
        weighting_ref_directions /= weighting_ref_directions.sum()
        self.rewards = Rewards(
            progress=torch.tensor(reward_progress, device=device, dtype=torch.float32),
            weighting_ref_directions=weighting_ref_directions, # Progress in the weighted directions (directions indicating by closer short-term reference points have higher weights)
            higth_v=torch.tensor(reward_vel, device=device, dtype=torch.float32),
            reach_goal=torch.tensor(reward_reach_goal, device=device, dtype=torch.float32),
        )
        self.rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)

        self.penalties = Penalties(
            deviate_from_ref_path=torch.tensor(penalty_deviate_from_ref_path, device=device, dtype=torch.float32),
            near_boundary=torch.tensor(penalty_near_boundary, device=device, dtype=torch.float32),
            near_other_agents=torch.tensor(penalty_near_other_agents, device=device, dtype=torch.float32),
            collide_with_agents=torch.tensor(penalty_collide_with_agents, device=device, dtype=torch.float32),
            collide_with_boundaries=torch.tensor(penalty_collide_with_boundaries, device=device, dtype=torch.float32),
            change_steering=torch.tensor(penalty_change_steering, device=device, dtype=torch.float32),
            time=torch.tensor(penalty_time, device=device, dtype=torch.float32),
        )
        
        self.observations = Observations(
            n_nearing_agents=torch.tensor(self.parameters.n_nearing_agents_observed, device=device, dtype=torch.int32),
            noise_level=torch.tensor(noise_level, device=device, dtype=torch.float32),
            n_stored_steps=torch.tensor(n_stored_steps, device=device, dtype=torch.int32),
            n_observed_steps=torch.tensor(n_observed_steps, device=device, dtype=torch.int32),
            nearing_agents_indices=torch.zeros((batch_dim, self.n_agents, self.parameters.n_nearing_agents_observed), device=device, dtype=torch.int32)
        )
        assert self.observations.n_stored_steps >= 1, "The number of stored steps should be at least 1."
        assert self.observations.n_observed_steps >= 1, "The number of observed steps should be at least 1."
        assert self.observations.n_stored_steps >= self.observations.n_observed_steps, "The number of stored steps should be greater or equal than the number of observed steps."
        
        if self.parameters.is_ego_view:
            self.observations.past_pos = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.n_agents, 2), device=device, dtype=torch.float32))
            self.observations.past_rot = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.n_agents), device=device, dtype=torch.float32))
            self.observations.past_vertices = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.n_agents, 4, 2), device=device, dtype=torch.float32))
            self.observations.past_vel = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.n_agents, 2), device=device, dtype=torch.float32))
            self.observations.past_short_term_ref_points = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.n_agents, self.parameters.n_points_short_term, 2), device=device, dtype=torch.float32))
            self.observations.past_left_boundary = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.n_agents, self.ref_paths_agent_related.n_points_nearing_boundary, 2), device=device, dtype=torch.float32))
            self.observations.past_right_boundary = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.n_agents, self.ref_paths_agent_related.n_points_nearing_boundary, 2), device=device, dtype=torch.float32))
        else:
            # Bird view
            self.observations.past_pos = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, 2), device=device, dtype=torch.float32))
            self.observations.past_rot = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents), device=device, dtype=torch.float32))
            self.observations.past_vertices = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, 4, 2), device=device, dtype=torch.float32))
            self.observations.past_vel = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, 2), device=device, dtype=torch.float32))
            self.observations.past_short_term_ref_points = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.parameters.n_points_short_term, 2), device=device, dtype=torch.float32))
            self.observations.past_left_boundary = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.ref_paths_agent_related.n_points_nearing_boundary, 2), device=device, dtype=torch.float32))
            self.observations.past_right_boundary = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.ref_paths_agent_related.n_points_nearing_boundary, 2), device=device, dtype=torch.float32))

        self.observations.past_action_vel = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents), device=device, dtype=torch.float32))
        self.observations.past_action_steering = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents), device=device, dtype=torch.float32))
        self.observations.past_distance_to_ref_path = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents), device=device, dtype=torch.float32))
        self.observations.past_distance_to_boundaries = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents), device=device, dtype=torch.float32))
        self.observations.past_distance_to_left_boundary = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents), device=device, dtype=torch.float32))
        self.observations.past_distance_to_right_boundary = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents), device=device, dtype=torch.float32))
        self.observations.past_distance_to_agents = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.n_agents), device=device, dtype=torch.float32))
        self.observations.past_lengths = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents), device=device, dtype=torch.float32))
        self.observations.past_widths = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents), device=device, dtype=torch.float32))

        self.normalizers = Normalizers(
            pos=torch.tensor([agent_length * 10, agent_length * 10], device=device, dtype=torch.float32),
            pos_world=torch.tensor([world_x_dim, world_y_dim], device=device, dtype=torch.float32),
            v=torch.tensor(max_speed, device=device, dtype=torch.float32),
            rot=torch.tensor(2 * torch.pi, device=device, dtype=torch.float32),
            action_steering=max_steering_angle,
            action_vel=torch.tensor(max_speed, device=device, dtype=torch.float32),
            distance_lanelet=torch.tensor(lane_width * 3, device=device, dtype=torch.float32),
            distance_ref=torch.tensor(lane_width * 3, device=device, dtype=torch.float32),
            distance_agent=torch.tensor(agent_length * 10, device=device, dtype=torch.float32),
        )
        
        # Distances to boundaries and reference path, and also the closest point on the reference paths of agents
        if self.parameters.is_use_mtv_distance:
            distance_type = "MTV" # One of {"c2c", "MTV"}
        else:
            distance_type = "c2c" # One of {"c2c", "MTV"}
        # print(colored("[INFO] Distance type: ", "black"), colored(distance_type, "blue"))
            
        self.distances = Distances(
            type = distance_type, # Type of distances between agents
            agents=torch.zeros(world.batch_dim, self.n_agents, self.n_agents, dtype=torch.float32),
            left_boundaries=torch.zeros((batch_dim, self.n_agents, 1 + 4), device=device, dtype=torch.float32), # The first entry for the center, the last 4 entries for the four vertices
            right_boundaries=torch.zeros((batch_dim, self.n_agents, 1 + 4), device=device, dtype=torch.float32),
            boundaries=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.float32),
            ref_paths=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.float32),
            closest_point_on_ref_path=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32),
            closest_point_on_left_b=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32),
            closest_point_on_right_b=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32),
        )

        self.thresholds = Thresholds(
            reach_goal=torch.tensor(threshold_reach_goal, device=device, dtype=torch.float32),
            deviate_from_ref_path=torch.tensor(threshold_deviate_from_ref_path, device=device, dtype=torch.float32),
            near_boundary_low=torch.tensor(threshold_near_boundary_low, device=device, dtype=torch.float32),
            near_boundary_high=torch.tensor(threshold_near_boundary_high, device=device, dtype=torch.float32),
            near_other_agents_low=torch.tensor(
                threshold_near_other_agents_c2c_low if self.distances.type == "c2c" else threshold_near_other_agents_MTV_low, 
                device=device,
                dtype=torch.float32
            ),
            near_other_agents_high=torch.tensor(
                threshold_near_other_agents_c2c_high if self.distances.type == "c2c" else threshold_near_other_agents_MTV_high, 
                device=device,
                dtype=torch.float32
            ),
            change_steering=torch.tensor(threshold_change_steering, device=device, dtype=torch.float32).deg2rad(),
            no_reward_if_too_close_to_boundaries=torch.tensor(threshold_no_reward_if_too_close_to_boundaries, device=device, dtype=torch.float32),
            no_reward_if_too_close_to_other_agents=torch.tensor(threshold_no_reward_if_too_close_to_other_agents, device=device, dtype=torch.float32),
            distance_mask_agents=agent_length * 5,
        )
        
        # Create agents
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                shape=Box(length=l_f+l_r, width=width),
                color=colors[i],
                collide=False,
                render_action=False,
                u_range=[max_speed, max_steering_angle],    # Control command serves as velocity command 
                u_multiplier=[1, 1],
                max_speed=max_speed,
                dynamics=KinematicBicycle(                  # Use the kinematic bicycle model for each agent
                    world, 
                    width=width, 
                    l_f=l_f, 
                    l_r=l_r, 
                    max_steering_angle=max_steering_angle, 
                    integration="rk4"                       # one of {"euler", "rk4"}
                )
            )                            
            world.add_agent(agent)
        
        self.constants = Constants(
            env_idx_broadcasting=torch.arange(batch_dim, device=device, dtype=torch.int32).unsqueeze(-1),
            empty_action_vel=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.float32),
            empty_action_steering=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.float32),
            mask_pos=torch.tensor(1, device=device, dtype=torch.float32),
            mask_zero=torch.tensor(0, device=device, dtype=torch.float32),
            mask_one=torch.tensor(1, device=device, dtype=torch.float32),
            reset_agent_min_distance=torch.tensor((l_f+l_r) ** 2 + width ** 2, device=device, dtype=torch.float32).sqrt() * 1.2,
        )
        
        # Initialize collision matrix
        self.collisions = Collisions(
            with_agents=torch.zeros((world.batch_dim, self.n_agents, self.n_agents), device=device, dtype=torch.bool),
            with_lanelets=torch.zeros((world.batch_dim, self.n_agents), device=device, dtype=torch.bool),
            with_entry_segments=torch.zeros((world.batch_dim, self.n_agents), device=device, dtype=torch.bool),
            with_exit_segments=torch.zeros((world.batch_dim, self.n_agents), device=device, dtype=torch.bool),
        )
        
        if self.parameters.is_challenging_initial_state_buffer:
            self.initial_state_buffer = InitialStateBuffer(
                # Used only when self.parameters.is_challenging_initial_state_buffer is True
                probability_record=torch.tensor(probability_record, device=device, dtype=torch.float32),
                probability_use_recording=torch.tensor(probability_use_recording, device=device, dtype=torch.float32),
                buffer=torch.zeros((buffer_size, self.n_agents, 8), device=device, dtype=torch.float32), # [pos_x, pos_y, rot, vel_x, vel_y, scenario_id, path_id, point_id]
            )

        # Store the states of agents at previous several time steps
        self.state_buffer = StateBuffer(
            buffer=torch.zeros((n_steps_stored, batch_dim, self.n_agents, 8), device=device, dtype=torch.float32), # [pos_x, pos_y, rot, vel_x, vel_y, scenario_id, path_id, point_id],
        )
        
        self.evaluation = Evaluation(
            pos_traj=torch.zeros((batch_dim, self.n_agents, self.parameters.max_steps, 2), device=device, dtype=torch.float32),
            v_traj=torch.zeros((batch_dim, self.n_agents, self.parameters.max_steps, 2), device=device, dtype=torch.float32),
            rot_traj=torch.zeros((batch_dim, self.n_agents, self.parameters.max_steps), device=device, dtype=torch.float32),
            deviation_from_ref_path=torch.zeros((batch_dim, self.n_agents, self.parameters.max_steps), device=device, dtype=torch.float32),
            path_tracking_error_mean=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.float32),
        )
        
        return world

    def reset_world_at(self, env_index: int = None, agent_index: int = None):
        # print(f"[DEBUG] reset_world_at(): env_index = {env_index}")
        """
        This function resets the world at the specified env_index.

        Args:
        :param env_index: index of the environment to reset. If None a vectorized reset should be performed

        """
        agents = self.world.agents

        is_reset_single_agent = agent_index is not None
        
        if is_reset_single_agent:
            assert env_index is not None

        for env_i in [env_index] if env_index is not None else range(self.world.batch_dim):
            # Begining of a new simulation (only record for the first env)
            if env_i == 0:
                self.timer.step_duration[:] = 0
                self.timer.start = time.time()
                self.timer.step_begin = time.time()
                self.timer.end = 0
                
            if not is_reset_single_agent:
                # Each time step of a simulation
                self.timer.step[env_i] = 0

            # Get the center line and boundaries of the long-term reference path for each agent
            if self.parameters.scenario_type == "CPM_mixed":
                if is_reset_single_agent:
                    scenario_id = self.ref_paths_agent_related.scenario_id[env_i, agent_index] # Keep the same scenario
                else:
                    scenario_id = torch.multinomial(
                        torch.tensor(self.parameters.cpm_scenario_probabilities, device=self.world.device, dtype=torch.float32), 
                        1, 
                        replacement=True
                    ).item() + 1 # A random interger {1, 2, 3}
                    self.ref_paths_agent_related.scenario_id[env_i, :] = scenario_id
                if scenario_id == 1:
                    # Intersection scenario
                    ref_paths_scenario = self.ref_paths_map_related.long_term_intersection
                    extended_points = self.ref_paths_map_related.point_extended_intersection
                elif scenario_id == 2:
                    # Merge-in scenario
                    ref_paths_scenario = self.ref_paths_map_related.long_term_merge_in
                    extended_points = self.ref_paths_map_related.point_extended_merge_in
                elif scenario_id == 3:
                    # Merge-out scenario
                    ref_paths_scenario = self.ref_paths_map_related.long_term_merge_out
                    extended_points = self.ref_paths_map_related.point_extended_merge_out
            else:
                ref_paths_scenario = self.ref_paths_map_related.long_term_all
                extended_points = self.ref_paths_map_related.point_extended_all
                self.ref_paths_agent_related.scenario_id[env_i, :] = 0 # 0 for others, 1 for intersection, 2 for merge-in, 3 for merge-out scenario
            
            if (self.parameters.is_challenging_initial_state_buffer and 
                (torch.rand(1) < self.initial_state_buffer.probability_use_recording) and 
                (self.initial_state_buffer.valid_size >= 1)):
                # Use initial state buffer
                is_use_state_buffer = True
                initial_state = self.initial_state_buffer.get_random()
                self.ref_paths_agent_related.scenario_id[env_i] = initial_state[:, self.initial_state_buffer.idx_scenario] # Update
                self.ref_paths_agent_related.path_id[env_i] = initial_state[:, self.initial_state_buffer.idx_path] # Update
                self.ref_paths_agent_related.point_id[env_i] = initial_state[:, self.initial_state_buffer.idx_point] # Update
                # print(colored(f"[LOG] Reset with path ids: {initial_state[:, -2]}", "red"))
            else:
                is_use_state_buffer = False
                
            for i_agent in range(self.n_agents) if not is_reset_single_agent else agent_index.unsqueeze(0):
                if is_use_state_buffer:
                    path_id = initial_state[i_agent, self.initial_state_buffer.idx_path].int()
                    ref_path = ref_paths_scenario[path_id]
                
                    agents[i_agent].set_pos(initial_state[i_agent, 0:2], batch_index=env_i)
                    agents[i_agent].set_rot(initial_state[i_agent, 2], batch_index=env_i)
                    agents[i_agent].set_vel(initial_state[i_agent, 3:5], batch_index=env_i)
                    
                else:
                    is_feasible_initial_position_found = False
                    random_count = 0
                    
                    # Ramdomly generate initial states for each agent
                    while not is_feasible_initial_position_found:
                        if random_count >= 20:
                            cprint(f"Reset agent(s): random_count = {random_count}.", "grey")
                        random_count += 1
                        path_id = torch.randint(0, len(ref_paths_scenario), (1,)).item() # Select randomly a path
                        self.ref_paths_agent_related.path_id[env_i, i_agent] = path_id # Update
                        ref_path = ref_paths_scenario[path_id]
                        
                        num_points = ref_path["center_line"].shape[0]
                        
                        if self.parameters.scenario_type == "CPM_mixed":
                            # In the mixed scenarios of the CPM case, we aovid using the beginning part of a path, making agents encounter each other more frequently. Additionally, We avoid initializing agents to be at a very end of a path.
                            start_point_idx = 6
                            end_point_idx = int(num_points / 2)
                        else:
                            start_point_idx = 3  # Do not set to an overly small value to make sure agents are fully inside its lane 
                            end_point_idx = num_points - 3
                            
                        random_point_id = torch.randint(start_point_idx, end_point_idx, (1,)).item()
                        
                        self.ref_paths_agent_related.point_id[env_i, i_agent] = random_point_id # Update
                        position_start = ref_path["center_line"][random_point_id]
                        agents[i_agent].set_pos(position_start, batch_index=env_i)

                        # Check if the initial position is feasible
                        if not is_reset_single_agent:
                            if i_agent == 0:
                                # The initial position of the first agent is always feasible
                                is_feasible_initial_position_found = True
                                continue
                            else:
                                positions = torch.stack([self.world.agents[i].state.pos[env_i] for i in range(i_agent+1)])
                        else:
                            # Check if the initial position of the agent to be reset is collision-free with other agents
                            positions = torch.stack([self.world.agents[i].state.pos[env_i] for i in range(self.n_agents)])
                                            
                        diff_sq = (positions[i_agent, :] - positions) ** 2 # Calculate pairwise squared differences in positions
                        initial_mutual_distances_sq = torch.sum(diff_sq, dim=-1)
                        initial_mutual_distances_sq[i_agent] = torch.max(initial_mutual_distances_sq) + 1 # Set self-to-self distance to a sufficiently high value
                        min_distance_sq = torch.min(initial_mutual_distances_sq)
                        
                        is_feasible_initial_position_found = min_distance_sq >= (self.constants.reset_agent_min_distance ** 2)

                    rot_start = ref_path["center_line_yaw"][random_point_id]
                    vel_start_abs = torch.rand(1, dtype=torch.float32, device=self.world.device) * agents[i_agent].max_speed # Random initial velocity
                    vel_start = torch.hstack([vel_start_abs * torch.cos(rot_start), vel_start_abs * torch.sin(rot_start)])

                    agents[i_agent].set_rot(rot_start, batch_index=env_i)
                    agents[i_agent].set_vel(vel_start, batch_index=env_i)
                
                # Long-term reference paths for agents
                n_points_long_term = ref_path["center_line"].shape[0]
                
                self.ref_paths_agent_related.long_term[env_i, i_agent, 0:n_points_long_term, :] = ref_path["center_line"]
                self.ref_paths_agent_related.long_term[env_i, i_agent, n_points_long_term:(n_points_long_term+self.parameters.n_points_short_term * self.ref_paths_map_related.sample_interval), :] = extended_points[path_id, :, :]
                self.ref_paths_agent_related.long_term[env_i, i_agent, (n_points_long_term+self.parameters.n_points_short_term * self.ref_paths_map_related.sample_interval):, :] = extended_points[path_id, -1, :]
                self.ref_paths_agent_related.n_points_long_term[env_i, i_agent] = n_points_long_term
                
                self.ref_paths_agent_related.long_term_vec_normalized[env_i, i_agent, 0:n_points_long_term-1, :] = ref_path["center_line_vec_normalized"]
                self.ref_paths_agent_related.long_term_vec_normalized[env_i, i_agent, (n_points_long_term-1):(n_points_long_term-1+self.parameters.n_points_short_term * self.ref_paths_map_related.sample_interval), :] = ref_path["center_line_vec_normalized"][-1, :]

                n_points_left_b = ref_path["left_boundary_shared"].shape[0]
                self.ref_paths_agent_related.left_boundary[env_i, i_agent, 0:n_points_left_b, :] = ref_path["left_boundary_shared"]
                self.ref_paths_agent_related.left_boundary[env_i, i_agent, n_points_left_b:, :] = ref_path["left_boundary_shared"][-1, :]
                self.ref_paths_agent_related.n_points_left_b[env_i, i_agent] = n_points_left_b
                
                n_points_right_b = ref_path["right_boundary_shared"].shape[0]
                self.ref_paths_agent_related.right_boundary[env_i, i_agent, 0:n_points_right_b, :] = ref_path["right_boundary_shared"]
                self.ref_paths_agent_related.right_boundary[env_i, i_agent, n_points_right_b:, :] = ref_path["right_boundary_shared"][-1, :]
                self.ref_paths_agent_related.n_points_right_b[env_i, i_agent] = n_points_right_b

                self.ref_paths_agent_related.entry[env_i, i_agent, 0, :] = ref_path["left_boundary_shared"][0, :]
                self.ref_paths_agent_related.entry[env_i, i_agent, 1, :] = ref_path["right_boundary_shared"][0, :]

                self.ref_paths_agent_related.exit[env_i, i_agent, 0, :] = ref_path["left_boundary_shared"][-1, :]
                self.ref_paths_agent_related.exit[env_i, i_agent, 1, :] = ref_path["right_boundary_shared"][-1, :]
                
                self.ref_paths_agent_related.is_loop[env_i, i_agent] = ref_path["is_loop"]

            # The operations below can be done for all envs in parallel
            if env_index is None:
                if env_i == (self.world.batch_dim - 1):
                    env_j = slice(None) # `slice(None)` is equivalent to `:`
                else:
                    continue
            else:
                env_j = env_i
                
            for i_agent in range(self.n_agents) if not is_reset_single_agent else agent_index.unsqueeze(0):
                # Distance from the center of gravity (CG) of the agent to its reference path
                self.distances.ref_paths[env_j, i_agent], self.distances.closest_point_on_ref_path[env_j, i_agent] = get_perpendicular_distances(
                    point=agents[i_agent].state.pos[env_j, :],
                    polyline=self.ref_paths_agent_related.long_term[env_j, i_agent],
                    n_points_long_term=self.ref_paths_agent_related.n_points_long_term[env_j, i_agent],
                )
                # Distances from CG to left boundary
                center_2_left_b, self.distances.closest_point_on_left_b[env_j, i_agent] = get_perpendicular_distances(
                    point=agents[i_agent].state.pos[env_j, :],
                    polyline=self.ref_paths_agent_related.left_boundary[env_j, i_agent],
                    n_points_long_term=self.ref_paths_agent_related.n_points_left_b[env_j, i_agent],
                )
                self.distances.left_boundaries[env_j, i_agent, 0] = center_2_left_b - (agents[i_agent].shape.width / 2)
                # Distances from CG to right boundary
                center_2_right_b, self.distances.closest_point_on_right_b[env_j, i_agent] = get_perpendicular_distances(
                    point=agents[i_agent].state.pos[env_j, :],
                    polyline=self.ref_paths_agent_related.right_boundary[env_j, i_agent],
                    n_points_long_term=self.ref_paths_agent_related.n_points_right_b[env_j, i_agent],
                )
                self.distances.right_boundaries[env_j, i_agent, 0] = center_2_right_b - (agents[i_agent].shape.width / 2)
                # Calculate the positions of the four vertices of the agents
                self.vertices[env_j, i_agent] = get_rectangle_vertices(
                    center=agents[i_agent].state.pos[env_j, :],
                    yaw=agents[i_agent].state.rot[env_j, :], 
                    width=agents[i_agent].shape.width, 
                    length=agents[i_agent].shape.length,
                    is_close_shape=True
                )
                # Distances from the four vertices of the agent to its left and right lanelet boundary
                for c_i in range(4):
                    self.distances.left_boundaries[env_j, i_agent, c_i + 1], _ = get_perpendicular_distances(
                        point=self.vertices[env_j, i_agent, c_i, :],
                        polyline=self.ref_paths_agent_related.left_boundary[env_j, i_agent],
                        n_points_long_term=self.ref_paths_agent_related.n_points_left_b[env_j, i_agent],
                    )
                    self.distances.right_boundaries[env_j, i_agent, c_i + 1], _ = get_perpendicular_distances(
                        point=self.vertices[env_j, i_agent, c_i, :],
                        polyline=self.ref_paths_agent_related.right_boundary[env_j, i_agent],
                        n_points_long_term=self.ref_paths_agent_related.n_points_right_b[env_j, i_agent],
                    )
                # Distance from agent to its left/right lanelet boundary is defined as the minimum distance among five distances (four vertices, CG)
                self.distances.boundaries[env_j, i_agent], _ = torch.min(
                    torch.hstack(
                        (
                            self.distances.left_boundaries[env_j, i_agent],
                            self.distances.right_boundaries[env_j, i_agent]
                        )
                    ),
                    dim=-1
                )
        
                # Get the short-term reference paths
                self.ref_paths_agent_related.short_term[env_j, i_agent], _ = get_short_term_reference_path(
                    polyline=self.ref_paths_agent_related.long_term[env_j, i_agent],
                    index_closest_point=self.distances.closest_point_on_ref_path[env_j, i_agent],
                    n_points_to_return=self.parameters.n_points_short_term, 
                    device=self.world.device,
                    is_polyline_a_loop=self.ref_paths_agent_related.is_loop[env_j, i_agent],
                    n_points_long_term=self.ref_paths_agent_related.n_points_long_term[env_j, i_agent],
                    sample_interval=self.ref_paths_map_related.sample_interval,
                    n_points_shift=1,
                )

                if not self.parameters.is_observe_distance_to_boundaries:
                    # Get nearing points on boundaries
                    self.ref_paths_agent_related.nearing_points_left_boundary[env_j, i_agent], _ = get_short_term_reference_path(
                        polyline=self.ref_paths_agent_related.left_boundary[env_j, i_agent],
                        index_closest_point=self.distances.closest_point_on_left_b[env_j, i_agent],
                        n_points_to_return=self.ref_paths_agent_related.n_points_nearing_boundary,
                        device=self.world.device,
                        is_polyline_a_loop=self.ref_paths_agent_related.is_loop[env_j, i_agent],
                        n_points_long_term=self.ref_paths_agent_related.n_points_long_term[env_j, i_agent],
                        sample_interval=1,
                        n_points_shift=1,
                    )
                    self.ref_paths_agent_related.nearing_points_right_boundary[env_j, i_agent], _ = get_short_term_reference_path(
                        polyline=self.ref_paths_agent_related.right_boundary[env_j, i_agent],
                        index_closest_point=self.distances.closest_point_on_right_b[env_j, i_agent],
                        n_points_to_return=self.ref_paths_agent_related.n_points_nearing_boundary,
                        device=self.world.device,
                        is_polyline_a_loop=self.ref_paths_agent_related.is_loop[env_j, i_agent],
                        n_points_long_term=self.ref_paths_agent_related.n_points_long_term[env_j, i_agent],
                        sample_interval=1,
                        n_points_shift=1,
                    )

            # Compute mutual distances between agents 
            # TODO Enable the possibility of computing the mutual distances of agents in a single env
            mutual_distances = get_distances_between_agents(self=self, distance_type=self.distances.type, is_set_diagonal=True)
            # Reset mutual distances of all envs
            self.distances.agents[env_j, :, :] = mutual_distances[env_j, :, :]

            # Reset the collision matrix
            self.collisions.with_agents[env_j, :, :] = False
            self.collisions.with_lanelets[env_j, :] = False
            self.collisions.with_entry_segments[env_j, :] = False
            self.collisions.with_exit_segments[env_j, :] = False
            
        # Reset the state buffer
        self.state_buffer.reset() 
        state_add = torch.cat(
            (
                torch.stack([a.state.pos for a in agents], dim=1),
                torch.stack([a.state.rot for a in agents], dim=1),
                torch.stack([a.state.vel for a in agents], dim=1),
                self.ref_paths_agent_related.scenario_id[:].unsqueeze(-1),
                self.ref_paths_agent_related.path_id[:].unsqueeze(-1),
                self.ref_paths_agent_related.point_id[:].unsqueeze(-1),
            ),
            dim=-1
        )
        self.state_buffer.add(state_add) # Add new state

    def process_action(self, agent: Agent):
        # print("[DEBUG] process_action()")
        if hasattr(agent, "dynamics") and hasattr(agent.dynamics, "process_force"):
            agent.dynamics.process_force()
            assert not agent.action.u.isnan().any()
            assert not agent.action.u.isinf().any()
        else:
            # The agent does not have a dynamics property, or it does not have a process_force method
            pass

    def reward(self, agent: Agent):
        # print("[DEBUG] reward()")
        # Initialize
        self.rew[:] = 0
        
        # Get the index of the current agent
        agent_index = self.world.agents.index(agent)

        # If rewards are shared among agents
        if self.shared_reward:
            # TODO Support shared reward
            raise NotImplementedError
            
        # [update] mutual distances between agents, vertices of each agent, and collision matrices
        if agent_index == 0: # Avoid repeated computations
            # Timer
            self.timer.step_duration[self.timer.step] = time.time() - self.timer.step_begin                
            self.timer.step_begin = time.time() # Set to the current time as the begin of the current time step
            self.timer.step += 1 # Increment step by 1
            # print(self.timer.step)

            # Update distances between agents
            self.distances.agents = get_distances_between_agents(self=self, distance_type=self.distances.type, is_set_diagonal=True)
            self.collisions.with_agents[:] = False   # Reset
            self.collisions.with_lanelets[:] = False # Reset
            self.collisions.with_entry_segments[:] = False # Reset
            self.collisions.with_exit_segments[:] = False # Reset

            for a_i in range(self.n_agents):
                self.vertices[:, a_i] = get_rectangle_vertices(
                    center=self.world.agents[a_i].state.pos,
                    yaw=self.world.agents[a_i].state.rot,
                    width=self.world.agents[a_i].shape.width,
                    length=self.world.agents[a_i].shape.length,
                    is_close_shape=True,
                )
                # Update the collision matrices
                if self.distances.type == "c2c":
                    for a_j in range(a_i+1, self.n_agents):
                        # Check for collisions between agents using the interX function
                        collision_batch_index = interX(self.vertices[:, a_i], self.vertices[:, a_j], False)
                        self.collisions.with_agents[torch.nonzero(collision_batch_index), a_i, a_j] = True
                        self.collisions.with_agents[torch.nonzero(collision_batch_index), a_j, a_i] = True
                elif self.distances.type == "MTV":
                    # Two agents collide if their MTV-based distance is zero
                    self.collisions.with_agents[:] = self.distances.agents == 0

                # Check for collisions between agents and lanelet boundaries
                collision_with_left_boundary = interX(
                    L1=self.vertices[:, a_i], 
                    L2=self.ref_paths_agent_related.left_boundary[:, a_i], 
                    is_return_points=False,
                ) # [batch_dim]
                collision_with_right_boundary = interX(
                    L1=self.vertices[:, a_i], 
                    L2=self.ref_paths_agent_related.right_boundary[:, a_i],
                    is_return_points=False,
                ) # [batch_dim]
                self.collisions.with_lanelets[(collision_with_left_boundary | collision_with_right_boundary), a_i] = True

                # Check for collisions with entry or exit segments (only need if agents' reference paths are not a loop)
                if not self.ref_paths_agent_related.is_loop[:, a_i].any():
                    self.collisions.with_entry_segments[:, a_i] = interX(
                        L1=self.vertices[:, a_i],
                        L2=self.ref_paths_agent_related.entry[:, a_i],
                        is_return_points=False,
                    )
                    self.collisions.with_exit_segments[:, a_i] = interX(
                        L1=self.vertices[:, a_i],
                        L2=self.ref_paths_agent_related.exit[:, a_i],
                        is_return_points=False,
                    )
                    
        # Distance from the center of gravity (CG) of the agent to its reference path
        self.distances.ref_paths[:, agent_index], self.distances.closest_point_on_ref_path[:, agent_index] = get_perpendicular_distances(
            point=agent.state.pos, 
            polyline=self.ref_paths_agent_related.long_term[:, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_long_term[:, agent_index],
        )
        # Distances from CG to left boundary
        center_2_left_b, self.distances.closest_point_on_left_b[:, agent_index] = get_perpendicular_distances(
            point=agent.state.pos[:, :], 
            polyline=self.ref_paths_agent_related.left_boundary[:, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_left_b[:, agent_index],
        )
        self.distances.left_boundaries[:, agent_index, 0] = center_2_left_b - (agent.shape.width / 2)
        # Distances from CG to right boundary
        center_2_right_b, self.distances.closest_point_on_right_b[:, agent_index] = get_perpendicular_distances(
            point=agent.state.pos[:, :], 
            polyline=self.ref_paths_agent_related.right_boundary[:, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_right_b[:, agent_index],
        )
        self.distances.right_boundaries[:, agent_index, 0] = center_2_right_b - (agent.shape.width / 2)
        # Distances from the four vertices of the agent to its left and right lanelet boundary
        for c_i in range(4):
            self.distances.left_boundaries[:, agent_index, c_i + 1], _ = get_perpendicular_distances(
                point=self.vertices[:, agent_index, c_i, :],
                polyline=self.ref_paths_agent_related.left_boundary[:, agent_index],
                n_points_long_term=self.ref_paths_agent_related.n_points_left_b[:, agent_index],
            )
            self.distances.right_boundaries[:, agent_index, c_i + 1], _ = get_perpendicular_distances(
                point=self.vertices[:, agent_index, c_i, :],
                polyline=self.ref_paths_agent_related.right_boundary[:, agent_index],
                n_points_long_term=self.ref_paths_agent_related.n_points_right_b[:, agent_index],
            )
        # Distance from agent to its left/right lanelet boundary is defined as the minimum distance among five distances (four vertices, CG)
        self.distances.boundaries[:, agent_index], _ = torch.min(
            torch.hstack(
                (
                    self.distances.left_boundaries[:, agent_index],
                    self.distances.right_boundaries[:, agent_index]
                )
            ),
            dim=-1
        )
            
        ##################################################
        ## [reward] forward movement
        ##################################################
        latest_state = self.state_buffer.get_latest(n=1)
        move_vec = (agent.state.pos - latest_state[:, agent_index, 0:2]).unsqueeze(1) # Vector of the current movement
        
        ref_points_vecs = self.ref_paths_agent_related.short_term[:, agent_index] - latest_state[:, agent_index, 0:2].unsqueeze(1) # Vectors from the previous position to the points on the short-term reference path
        move_projected = torch.sum(move_vec * ref_points_vecs, dim=-1)
        move_projected_weighted = torch.matmul(move_projected, self.rewards.weighting_ref_directions) # Put more weights on nearing reference points

        reward_movement = move_projected_weighted / (agent.max_speed * self.world.dt) * self.rewards.progress
        self.rew += reward_movement # Relative to the maximum possible movement
        
        ##################################################
        ## [reward] high velocity
        ##################################################   
        v_proj = torch.sum(agent.state.vel.unsqueeze(1) * ref_points_vecs, dim=-1).mean(-1)
        factor_moving_direction = torch.where(v_proj>0, 1, 2) # Get penalty if move in negative direction
        
        reward_vel = factor_moving_direction * v_proj / agent.max_speed * self.rewards.higth_v
        self.rew += reward_vel

        ##################################################
        ## [reward] reach goal
        ##################################################
        reward_goal = self.collisions.with_exit_segments[:, agent_index] * self.rewards.reach_goal
        self.rew += reward_goal

        ##################################################
        ## [penalty] close to lanelet boundaries
        ##################################################        
        penalty_close_to_lanelets = exponential_decreasing_fcn(
            x=self.distances.boundaries[:, agent_index],
            x0=self.thresholds.near_boundary_low, 
            x1=self.thresholds.near_boundary_high,
        ) * self.penalties.near_boundary
        self.rew += penalty_close_to_lanelets

        ##################################################
        ## [penalty] close to other agents
        ##################################################
        mutual_distance_exp_fcn = exponential_decreasing_fcn(
            x=self.distances.agents[:, agent_index, :], 
            x0=self.thresholds.near_other_agents_low, 
            x1=self.thresholds.near_other_agents_high
        )
        penalty_close_to_agents = torch.sum(mutual_distance_exp_fcn, dim=1) * self.penalties.near_other_agents
        self.rew += penalty_close_to_agents

        ##################################################
        ## [penalty] deviating from reference path
        ##################################################
        self.rew += self.distances.ref_paths[:, agent_index] / self.thresholds.deviate_from_ref_path * self.penalties.deviate_from_ref_path

        ##################################################
        ## [penalty] changing steering too quick
        ##################################################
        steering_current = self.observations.past_action_steering.get_latest(n=1)[:, agent_index]
        steering_past = self.observations.past_action_steering.get_latest(n=2)[:, agent_index]

        steering_change = torch.clamp(
            (steering_current - steering_past).abs() * self.normalizers.action_steering - self.thresholds.change_steering, # Not forget to denormalize
            min=0,
        )
        steering_change_reward_factor = steering_change / (2 * agent.u_range[1] - 2 * self.thresholds.change_steering)
        penalty_change_steering = steering_change_reward_factor * self.penalties.change_steering
        self.rew += penalty_change_steering

        # ##################################################
        # ## [penalty] colliding with other agents
        # ##################################################
        is_collide_with_agents = self.collisions.with_agents[:, agent_index]        
        penalty_collide_other_agents = is_collide_with_agents.any(dim=-1) * self.penalties.collide_with_agents
        self.rew += penalty_collide_other_agents

        ##################################################
        ## [penalty] colliding with lanelet boundaries
        ##################################################
        is_collide_with_lanelets = self.collisions.with_lanelets[:, agent_index]
        penalty_collide_lanelet = is_collide_with_lanelets * self.penalties.collide_with_boundaries
        self.rew += penalty_collide_lanelet

        ##################################################
        ## [penalty/reward] time
        ##################################################
        # Get time reward if moving in positive direction; otherwise get time penalty
        time_reward = torch.where(v_proj>0, 1, -1) * agent.state.vel.norm(dim=-1) / agent.max_speed * self.penalties.time
        self.rew += time_reward

        # [update] previous positions and short-term reference paths
        if agent_index == (self.n_agents - 1): # Avoid repeated updating
            state_add = torch.cat(
                (
                    torch.stack([a.state.pos for a in self.world.agents], dim=1),
                    torch.stack([a.state.rot for a in self.world.agents], dim=1),
                    torch.stack([a.state.vel for a in self.world.agents], dim=1),
                    self.ref_paths_agent_related.scenario_id[:].unsqueeze(-1),
                    self.ref_paths_agent_related.path_id[:].unsqueeze(-1),
                    self.ref_paths_agent_related.point_id[:].unsqueeze(-1),
                ),
                dim=-1
            )
            self.state_buffer.add(state_add)
        
        self.ref_paths_agent_related.short_term[:, agent_index], _ = get_short_term_reference_path(
            polyline=self.ref_paths_agent_related.long_term[:, agent_index], 
            index_closest_point=self.distances.closest_point_on_ref_path[:, agent_index],
            n_points_to_return=self.parameters.n_points_short_term, 
            device=self.world.device,
            is_polyline_a_loop=self.ref_paths_agent_related.is_loop[:, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_long_term[:, agent_index],
            sample_interval=self.ref_paths_map_related.sample_interval,
        )

        if not self.parameters.is_observe_distance_to_boundaries:
            # Get nearing points on boundaries
            self.ref_paths_agent_related.nearing_points_left_boundary[:, agent_index], _ = get_short_term_reference_path(
                polyline=self.ref_paths_agent_related.left_boundary[:, agent_index],
                index_closest_point=self.distances.closest_point_on_left_b[:, agent_index],
                n_points_to_return=self.ref_paths_agent_related.n_points_nearing_boundary,
                device=self.world.device,
                is_polyline_a_loop=self.ref_paths_agent_related.is_loop[:, agent_index],
                n_points_long_term=self.ref_paths_agent_related.n_points_long_term[:, agent_index],
                sample_interval=1,
                n_points_shift=-2,
            )
            self.ref_paths_agent_related.nearing_points_right_boundary[:, agent_index], _ = get_short_term_reference_path(
                polyline=self.ref_paths_agent_related.right_boundary[:, agent_index],
                index_closest_point=self.distances.closest_point_on_right_b[:, agent_index],
                n_points_to_return=self.ref_paths_agent_related.n_points_nearing_boundary,
                device=self.world.device,
                is_polyline_a_loop=self.ref_paths_agent_related.is_loop[:, agent_index],
                n_points_long_term=self.ref_paths_agent_related.n_points_long_term[:, agent_index],
                sample_interval=1,
                n_points_shift=-2,
            )

        assert not self.rew.isnan().any(), "Rewards contain nan."
        assert not self.rew.isinf().any(), "Rewards contain inf."
        
        
        # Clamed the reward to avoid abs(reward) being too large
        rew_clamed = torch.clamp(self.rew, min=-1, max=1)

        return rew_clamed


    def observation(self, agent: Agent):
        # print("[DEBUG] observation()")
        """
        Generate an observation for the given agent in all envs.

        Args:
            agent: The agent for which the observation is to be generated.

        Returns:
            The observation for the given agent in all envs.
        """
        agent_index = self.world.agents.index(agent)
        
        positions_global = torch.stack([a.state.pos for a in self.world.agents], dim=0).transpose(0, 1)
        rotations_global = torch.stack([a.state.rot for a in self.world.agents], dim=0).transpose(0, 1).squeeze(-1)
        
        lengths_global = torch.tensor(
            [a.shape.length for a in self.world.agents],
            device=self.world.device,
            dtype=torch.float32
        ).repeat(
            self.world.batch_dim,
            1
        )

        widths_global = torch.tensor(
            [a.shape.width for a in self.world.agents],
            device=self.world.device,
            dtype=torch.float32
        ).repeat(
            self.world.batch_dim,
            1
        )
        
        if agent_index == 0: # Avoid repeated computations            
            # Add new observation & normalize
            self.observations.past_distance_to_agents.add(self.distances.agents / self.normalizers.distance_lanelet)
            self.observations.past_distance_to_ref_path.add(self.distances.ref_paths / self.normalizers.distance_lanelet)
            self.observations.past_distance_to_left_boundary.add(torch.min(self.distances.left_boundaries, dim=-1)[0] / self.normalizers.distance_lanelet)
            self.observations.past_distance_to_right_boundary.add(torch.min(self.distances.right_boundaries, dim=-1)[0] / self.normalizers.distance_lanelet)
            self.observations.past_distance_to_boundaries.add(self.distances.boundaries / self.normalizers.distance_lanelet)
            self.observations.past_lengths.add(lengths_global / self.normalizers.distance_agent)  # Use distance to agents as the normalizer
            self.observations.past_widths.add(widths_global / self.normalizers.distance_agent)

            if self.parameters.is_ego_view:
                pos_i_others = torch.zeros((self.world.batch_dim, self.n_agents, self.n_agents, 2), device=self.world.device, dtype=torch.float32) # Positions of other agents relative to agent i
                rot_i_others = torch.zeros((self.world.batch_dim, self.n_agents, self.n_agents), device=self.world.device, dtype=torch.float32) # Rotations of other agents relative to agent i
                vel_i_others = torch.zeros((self.world.batch_dim, self.n_agents, self.n_agents, 2), device=self.world.device, dtype=torch.float32) # Velocities of other agents relative to agent i
                ref_i_others = torch.zeros_like((self.observations.past_short_term_ref_points.get_latest())) # Reference paths of other agents relative to agent i
                l_b_i_others = torch.zeros_like((self.observations.past_left_boundary.get_latest())) # Left boundaries of other agents relative to agent i
                r_b_i_others = torch.zeros_like((self.observations.past_right_boundary.get_latest())) # Right boundaries of other agents relative to agent i
                ver_i_others = torch.zeros_like((self.observations.past_vertices.get_latest())) # Vertices of other agents relative to agent i

                for a_i in range(self.n_agents):
                    pos_i = self.world.agents[a_i].state.pos
                    rot_i = self.world.agents[a_i].state.rot

                    # Store new observation - position
                    pos_i_others[:, a_i] = transform_from_global_to_local_coordinate(
                        pos_i=pos_i,
                        pos_j=positions_global,
                        rot_i=rot_i,
                    )

                    # Store new observation - rotation
                    rot_i_others[:, a_i] = rotations_global - rot_i
                    
                    for a_j in range(self.n_agents):                        
                        # Store new observation - velocities
                        rot_rel = rot_i_others[:, a_i, a_j].unsqueeze(1)
                        vel_abs = torch.norm(self.world.agents[a_j].state.vel, dim=1).unsqueeze(1) # TODO Check if relative velocities here are better
                        vel_i_others[:, a_i, a_j] = torch.hstack(
                            (
                                vel_abs * torch.cos(rot_rel), 
                                vel_abs * torch.sin(rot_rel)
                            )
                        )
                        
                        # Store new observation - reference paths
                        ref_i_others[:, a_i, a_j] = transform_from_global_to_local_coordinate(
                            pos_i=pos_i,
                            pos_j=self.ref_paths_agent_related.short_term[:, a_j],
                            rot_i=rot_i,
                        )
                        
                        # Store new observation - left boundary
                        if not self.parameters.is_observe_distance_to_boundaries:
                            l_b_i_others[:, a_i, a_j] = transform_from_global_to_local_coordinate(
                                pos_i=pos_i,
                                pos_j=self.ref_paths_agent_related.nearing_points_left_boundary[:, a_j],
                                rot_i=rot_i,
                            )
                            
                            # Store new observation - right boundary
                            r_b_i_others[:, a_i, a_j] = transform_from_global_to_local_coordinate(
                                pos_i=pos_i,
                                pos_j=self.ref_paths_agent_related.nearing_points_right_boundary[:, a_j],
                                rot_i=rot_i,
                            )
                        
                        # Store new observation - vertices
                        ver_i_others[:, a_i, a_j] = transform_from_global_to_local_coordinate(
                            pos_i=pos_i,
                            pos_j=self.vertices[:, a_j, 0:4, :],
                            rot_i=rot_i,
                        )
                # Add new observations & normalize
                self.observations.past_pos.add(pos_i_others / (self.normalizers.pos if self.parameters.is_ego_view else self.normalizers.pos_world))
                self.observations.past_rot.add(rot_i_others / self.normalizers.rot)
                self.observations.past_vel.add(vel_i_others / self.normalizers.v)
                self.observations.past_short_term_ref_points.add(ref_i_others / (self.normalizers.pos if self.parameters.is_ego_view else self.normalizers.pos_world))
                self.observations.past_left_boundary.add(l_b_i_others / (self.normalizers.pos if self.parameters.is_ego_view else self.normalizers.pos_world))
                self.observations.past_right_boundary.add(r_b_i_others / (self.normalizers.pos if self.parameters.is_ego_view else self.normalizers.pos_world))
                self.observations.past_vertices.add(ver_i_others / (self.normalizers.pos if self.parameters.is_ego_view else self.normalizers.pos_world))
                
            else: # Global coordinate system
                # Store new observations
                self.observations.past_pos.add(positions_global / (self.normalizers.pos if self.parameters.is_ego_view else self.normalizers.pos_world))
                self.observations.past_vel.add(torch.stack([a.state.vel for a in self.world.agents], dim=1) / self.normalizers.v)
                self.observations.past_rot.add(rotations_global[:] / self.normalizers.rot)
                self.observations.past_vertices.add(self.vertices[:, :, 0:4, :] / (self.normalizers.pos if self.parameters.is_ego_view else self.normalizers.pos_world))
                self.observations.past_short_term_ref_points.add(self.ref_paths_agent_related.short_term[:] / (self.normalizers.pos if self.parameters.is_ego_view else self.normalizers.pos_world))
                self.observations.past_left_boundary.add(self.ref_paths_agent_related.nearing_points_left_boundary  / (self.normalizers.pos if self.parameters.is_ego_view else self.normalizers.pos_world))
                self.observations.past_right_boundary.add(self.ref_paths_agent_related.nearing_points_right_boundary  / (self.normalizers.pos if self.parameters.is_ego_view else self.normalizers.pos_world))

            # Add new observation - actions & normalize
            if agent.action.u is None:
                self.observations.past_action_vel.add(self.constants.empty_action_vel)
                self.observations.past_action_steering.add(self.constants.empty_action_steering)
            else:
                self.observations.past_action_vel.add(torch.stack([a.action.u[:, 0] for a in self.world.agents], dim=1) / self.normalizers.action_vel)
                self.observations.past_action_steering.add(torch.stack([a.action.u[:, 1] for a in self.world.agents], dim=1) / self.normalizers.action_steering)
                
            if self.parameters.is_apply_mask:
                # Determine the current lanelet IDs of all agents of all envs for later use 
                self.map.determine_current_lanelet(positions_global)

        ##################################################
        ## Observation of other agents
        ##################################################        
        if self.parameters.is_partial_observation:
            # Each agent observes only a fixed number of nearest agents
            nearing_agents_distances, nearing_agents_indices = torch.topk(self.distances.agents[:, agent_index], k=self.observations.n_nearing_agents, largest=False)

            if self.parameters.is_apply_mask:
                # Two kinds of agents will be masked by ego agents:
                # 1. By distance: agents that are distant to the ego agents
                # 2. By lanelet relation: agents whose lanelets are not the neighboring lanelets or the same lanelets of the ego agents
                masked_agents_by_distance = (nearing_agents_distances >= self.thresholds.distance_mask_agents)
                # print(f"masked_agents_by_distance = {masked_agents_by_distance}")
                if len(self.map.parser.neighboring_lanelets_idx) != 0:
                    # Mask agents by lanelets
                    masked_agents_by_lanelets = self.map.determine_masked_agents_by_lanelets(agent_index, nearing_agents_indices)
                else:
                    masked_agents_by_lanelets = torch.zeros((self.world.batch_dim, self.parameters.n_nearing_agents_observed), device=self.world.device, dtype=torch.bool)
                    
                masked_agents = masked_agents_by_distance | masked_agents_by_lanelets               
                    
            else:
                # Otherwise no agents will be masked
                masked_agents = torch.zeros((self.world.batch_dim, self.parameters.n_nearing_agents_observed), device=self.world.device, dtype=torch.bool)
            
            indexing_tuple_1 = (self.constants.env_idx_broadcasting,) + \
                            ((agent_index,) if self.parameters.is_ego_view else ()) + \
                            (nearing_agents_indices,)
            
            # Positions of nearing agents
            obs_pos_other_agents = self.observations.past_pos.get_latest()[indexing_tuple_1] # [batch_size, n_nearing_agents, 2]
            obs_pos_other_agents[masked_agents] = self.constants.mask_one # Position mask

            # Rotations of nearing agents
            obs_rot_other_agents = self.observations.past_rot.get_latest()[indexing_tuple_1] # [batch_size, n_nearing_agents]
            obs_rot_other_agents[masked_agents] = self.constants.mask_zero # Rotation mask

            # Lengths and widths of nearing agents
            obs_lengths_other_agents = self.observations.past_lengths.get_latest()[self.constants.env_idx_broadcasting, nearing_agents_indices]
            obs_widths_other_agents = self.observations.past_widths.get_latest()[self.constants.env_idx_broadcasting, nearing_agents_indices]

            # Velocities of nearing agents
            obs_vel_other_agents = self.observations.past_vel.get_latest()[indexing_tuple_1] # [batch_size, n_nearing_agents]
            obs_vel_other_agents[masked_agents] = self.constants.mask_zero # Velocity mask
            
            # Reference paths of nearing agents
            obs_ref_path_other_agents = self.observations.past_short_term_ref_points.get_latest()[indexing_tuple_1] # [batch_size, n_nearing_agents, n_points_short_term, 2]
            obs_ref_path_other_agents[masked_agents] = self.constants.mask_one # Reference-path mask

            # vertices of nearing agents
            obs_vertices_other_agents = self.observations.past_vertices.get_latest()[indexing_tuple_1] # [batch_size, n_nearing_agents, 4, 2]
            obs_vertices_other_agents[masked_agents] = self.constants.mask_one # Reference-path mask
            
            # Distances to nearing agents
            obs_distance_other_agents = self.observations.past_distance_to_agents.get_latest()[self.constants.env_idx_broadcasting, agent_index, nearing_agents_indices] # [batch_size, n_nearing_agents]
            obs_distance_other_agents[masked_agents] = self.constants.mask_one # Distance mask

        else:
            indexing_tuple_2 = (self.constants.env_idx_broadcasting.squeeze(-1),) + \
                            ((agent_index,) if self.parameters.is_ego_view else ())

            obs_pos_other_agents = self.observations.past_pos.get_latest()[indexing_tuple_2] # [batch_size, n_agents, 2]            
            obs_rot_other_agents = self.observations.past_rot.get_latest()[indexing_tuple_2] # [batch_size, n_agents, (n_agents)]
            obs_vel_other_agents = self.observations.past_vel.get_latest()[indexing_tuple_2] # [batch_size, n_agents, 2]
            obs_ref_path_other_agents = self.observations.past_short_term_ref_points.get_latest()[indexing_tuple_2] # [batch_size, n_agents, n_points_short_term, 2]
            obs_vertices_other_agents = self.observations.past_vertices.get_latest()[indexing_tuple_2] # [batch_size, n_agents, 4, 2]
            obs_distance_other_agents = self.observations.past_distance_to_agents.get_latest()[indexing_tuple_2] # [batch_size, n_agents]
            obs_distance_other_agents[indexing_tuple_2] = 0 # Reset self-self distance to zero
            obs_lengths_other_agents = self.observations.past_lengths.get_latest()[indexing_tuple_2]
            obs_widths_other_agents = self.observations.past_widths.get_latest()[indexing_tuple_2]

        # Flatten the last dimensions to combine all features into a single dimension
        obs_pos_other_agents_flat = obs_pos_other_agents.reshape(self.world.batch_dim, self.observations.n_nearing_agents, -1)
        obs_rot_other_agents_flat = obs_rot_other_agents.reshape(self.world.batch_dim, self.observations.n_nearing_agents, -1)
        obs_vel_other_agents_flat = obs_vel_other_agents.reshape(self.world.batch_dim, self.observations.n_nearing_agents, -1)
        obs_ref_path_other_agents_flat = obs_ref_path_other_agents.reshape(self.world.batch_dim, self.observations.n_nearing_agents, -1)
        obs_vertices_other_agents_flat = obs_vertices_other_agents.reshape(self.world.batch_dim, self.observations.n_nearing_agents, -1)
        obs_distance_other_agents_flat = obs_distance_other_agents.reshape(self.world.batch_dim, self.observations.n_nearing_agents, -1)
        obs_lengths_other_agents_flat = obs_lengths_other_agents.reshape(self.world.batch_dim, self.observations.n_nearing_agents, -1)
        obs_widths_other_agents_flat = obs_widths_other_agents.reshape(self.world.batch_dim, self.observations.n_nearing_agents, -1)

        # Observation of other agents
        obs_others_list = [
            obs_vertices_other_agents_flat if self.parameters.is_observe_vertices else    # [other] vertices
            torch.cat(
                [
                    obs_pos_other_agents_flat,                                      # [others] positions
                    obs_rot_other_agents_flat,                                      # [others] rotations
                    obs_lengths_other_agents_flat,                                  # [others] lengths
                    obs_widths_other_agents_flat,                                   # [others] widths
                ], dim=-1
            ),  
            obs_vel_other_agents_flat,                                              # [others] velocities
            obs_distance_other_agents_flat if self.parameters.is_observe_distance_to_agents else None, # [others] mutual distances
            obs_ref_path_other_agents_flat if self.parameters.is_observe_ref_path_other_agents else None,               # [others] reference paths
        ]
        obs_others_list = [o for o in obs_others_list if o is not None] # Filter out None values
        obs_other_agents = torch.cat(obs_others_list, dim=-1).reshape(self.world.batch_dim, -1) # [batch_size, -1]        

        indexing_tuple_3 = (self.constants.env_idx_broadcasting,) + \
                            (agent_index,) + \
                            ((agent_index,) if self.parameters.is_ego_view else ())        
        indexing_tuple_vel = (self.constants.env_idx_broadcasting,) + \
                            (agent_index,) + \
                            ((agent_index, 0) if self.parameters.is_ego_view else ()) # In local coordinate system, only the first component is interesting, as the second is always 0
        # All observations
        obs_list = [
            None if self.parameters.is_ego_view else self.observations.past_pos.get_latest()[indexing_tuple_3].reshape(self.world.batch_dim, -1), # [own] position,
            None if self.parameters.is_ego_view else self.observations.past_rot.get_latest()[indexing_tuple_3].reshape(self.world.batch_dim, -1), # [own] rotation,
            self.observations.past_vel.get_latest()[indexing_tuple_vel].reshape(self.world.batch_dim, -1),                  # [own] velocity
            self.observations.past_short_term_ref_points.get_latest()[indexing_tuple_3].reshape(self.world.batch_dim, -1),       # [own] short-term reference path
            self.observations.past_distance_to_ref_path.get_latest()[:, agent_index].reshape(self.world.batch_dim, -1) if self.parameters.is_observe_distance_to_center_line else None, # [own] distances to reference paths
            self.observations.past_distance_to_left_boundary.get_latest()[:, agent_index].reshape(self.world.batch_dim, -1) if self.parameters.is_observe_distance_to_boundaries else self.observations.past_left_boundary.get_latest()[indexing_tuple_3].reshape(self.world.batch_dim, -1), # [own] left boundaries 
            self.observations.past_distance_to_right_boundary.get_latest()[:, agent_index].reshape(self.world.batch_dim, -1) if self.parameters.is_observe_distance_to_boundaries else self.observations.past_right_boundary.get_latest()[indexing_tuple_3].reshape(self.world.batch_dim, -1), # [own] right boundaries 
            obs_other_agents, # [others]
        ]
        obs_list = [o for o in obs_list if o is not None] # Filter out None values
        obs = torch.hstack(obs_list)
        
        assert not obs.isnan().any(), "Observations contain nan."
        assert not obs.isinf().any(), "Observations contain inf."
        # assert not (obs.abs() > 2).any(), "Observations contain values greater than 1."
        
        if self.parameters.is_add_noise:
            return obs + (0.02 * torch.rand_like(obs, device=self.world.device, dtype=torch.float32))
        else:
            return obs
    
    def done(self):
        # print("[DEBUG] done()")
        is_collision_with_agents = self.collisions.with_agents.view(self.world.batch_dim,-1).any(dim=-1) # [batch_dim]
        is_collision_with_lanelets = self.collisions.with_lanelets.any(dim=-1)
        is_leaving_entry_segment = self.collisions.with_entry_segments.any(dim=-1) & (self.timer.step >= 20)
        is_any_agents_leaving_exit_segment = self.collisions.with_exit_segments.any(dim=-1)
        is_max_steps_reached = self.timer.step == (self.parameters.max_steps - 1)
        
        if self.parameters.is_challenging_initial_state_buffer: # Record challenging initial states
            if torch.rand(1) > (1 - self.initial_state_buffer.probability_record): # Only a certain probability to record
                for env_collide in torch.where(is_collision_with_agents)[0]:
                    self.initial_state_buffer.add(self.state_buffer.get_latest(n=n_steps_stored)[env_collide])
                    # print(colored(f"[LOG] Record states with path ids: {self.ref_paths_agent_related.path_id[env_collide]}.", "blue"))
        
        if self.parameters.is_testing_mode:
            is_done = is_max_steps_reached # In test mode, we only reset the whole env if the maximum time steps are reached
            
            # Reset single agent
            agents_reset = (
                self.collisions.with_agents.any(dim=-1) |
                self.collisions.with_lanelets |
                self.collisions.with_entry_segments |
                self.collisions.with_exit_segments
            )
            agents_reset_indices = torch.where(agents_reset)
            for env_idx, agent_idx in zip(agents_reset_indices[0], agents_reset_indices[1]):
                if not is_done[env_idx]:
                    self.reset_world_at(env_index=env_idx, agent_index=agent_idx)
        else:
            is_done = is_max_steps_reached | is_collision_with_agents | is_collision_with_lanelets
            if self.parameters.scenario_type != "CPM_entire":  # This part only applies to the map that have loop-shaped paths 
                # Reset the whole system only when collisions occur. Reset a single agents if it leaves an entry or an exit
                
                # Reset single agnet
                agents_reset = (
                    self.collisions.with_entry_segments |
                    self.collisions.with_exit_segments
                )
                agents_reset_indices = torch.where(agents_reset)
                for env_idx, agent_idx in zip(agents_reset_indices[0], agents_reset_indices[1]):
                    if not is_done[env_idx]:
                        # Skip envs with done flag since later they will be reset anyway
                        self.reset_world_at(env_index=env_idx, agent_index=agent_idx)
                        # print(f"Reset agent {agent_idx} in env {env_idx}")
            else:
                # Reset the whole system once collisions occur. There is no entry or exit in this scenario.
                assert(not is_leaving_entry_segment.any())
                assert(not is_any_agents_leaving_exit_segment.any())

            assert not (is_collision_with_agents & (self.timer.step == 0)).any()
            assert not (is_collision_with_lanelets & (self.timer.step == 0)).any()
            assert not (is_leaving_entry_segment & (self.timer.step == 0)).any()
            assert not (is_max_steps_reached & (self.timer.step == 0)).any()
            assert not (is_any_agents_leaving_exit_segment & (self.timer.step == 0)).any()
            
        # Logs
        # if is_collision_with_agents.any():
        #     print("Collide with other agents.")
        # if is_collision_with_lanelets.any():
        #     print("Collide with lanelet.")
        # if is_leaving_entry_segment.any():
        #     print("At least one agent is leaving its entry segment.")
        # if is_max_steps_reached.any():
        #     print("The number of the maximum steps is reached.")
        # if is_any_agents_leaving_exit_segment.any():
        #     print("At least one agent is leaving its exit segment.")            

        return is_done


    def info(self, agent: Agent) -> Dict[str, Tensor]:
        """
        This function computes the info dict for "agent" in a vectorized way
        The returned dict should have a key for each info of interest and the corresponding value should
        be a tensor of shape (n_envs, info_size)

        Implementors can access the world at "self.world"

        To increase performance, tensors created should have the device set, like:
        torch.tensor(..., device=self.world.device)

        :param agent: Agent batch to compute info of
        :return: info: A dict with a key for each info of interest, and a tensor value  of shape (n_envs, info_size)
        """
        agent_index = self.world.agents.index(agent) # Index of the current agent

        is_action_empty = agent.action.u is None

        is_collision_with_agents = self.collisions.with_agents[:, agent_index].any(dim=-1) # [batch_dim]
        is_collision_with_lanelets = self.collisions.with_lanelets.any(dim=-1)

        info = {
            "pos": agent.state.pos / self.normalizers.pos_world,
            "rot": angle_eliminate_two_pi(agent.state.rot) / self.normalizers.rot,
            "vel": agent.state.vel / self.normalizers.v,
            "act_vel": (agent.action.u[:, 0] / self.normalizers.action_vel) if not is_action_empty else self.constants.empty_action_vel[:, agent_index],
            "act_steer": (agent.action.u[:, 1] / self.normalizers.action_steering) if not is_action_empty else self.constants.empty_action_steering[:, agent_index],
            "ref": (self.ref_paths_agent_related.short_term[:, agent_index] / self.normalizers.pos_world).reshape(self.world.batch_dim, -1),
            "distance_ref": self.distances.ref_paths[:, agent_index] / self.normalizers.distance_ref,
            "distance_left_b": self.distances.left_boundaries[:, agent_index].min(dim=-1)[0] / self.normalizers.distance_lanelet,
            "distance_right_b": self.distances.right_boundaries[:, agent_index].min(dim=-1)[0] / self.normalizers.distance_lanelet,
            "is_collision_with_agents": is_collision_with_agents,
            "is_collision_with_lanelets": is_collision_with_lanelets,
        }
        
        return info
    
    
    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering

        if self.parameters.is_real_time_rendering:
            if self.timer.step[0] == 0:
                pause_duration = 0 # Not sure how long should the simulation be paused at time step 0, so rather 0
            else:
                pause_duration = self.world.dt - (time.time() - self.timer.render_begin)
            if pause_duration > 0:
                time.sleep(pause_duration)
            # print(f"Paused for {pause_duration} sec.")
            
            self.timer.render_begin = time.time() # Update
        geoms = []
        
        # Visualize all lanelets
        for i in range(len(self.map.parser.lanelets_all)):
            lanelet = self.map.parser.lanelets_all[i]
            
            geom = rendering.PolyLine(
                v = lanelet["left_boundary"],
                close=False,
            )
            xform = rendering.Transform()
            geom.add_attr(xform)            
            geom.set_color(*Color.black100)
            geoms.append(geom)
            
            geom = rendering.PolyLine(
                v = lanelet["right_boundary"],
                close=False,
            )
            xform = rendering.Transform()
            geom.add_attr(xform)            
            geom.set_color(*Color.black100)
            geoms.append(geom)
        
        if self.parameters.is_visualize_extra_info:
            hight_a = -0.10
            hight_b = -0.20
            hight_c = -0.30
            
            # Title
            geom = rendering.TextLine(
                text=self.parameters.render_title,
                x=0.05 * resolution_factor,
                y=(self.world.y_semidim + hight_a) * resolution_factor,
                font_size=14,
            )
            xform = rendering.Transform()
            geom.add_attr(xform)  
            geoms.append(geom)

            # Time and time step
            geom = rendering.TextLine(
                text=f"t: {self.timer.step[0]*self.parameters.dt:.2f} sec",
                x=0.05 * resolution_factor,
                y=(self.world.y_semidim + hight_b) * resolution_factor,
                font_size=14,
            )
            xform = rendering.Transform()
            geom.add_attr(xform)  
            geoms.append(geom)
            
            geom = rendering.TextLine(
                text=f"n: {self.timer.step[0]}",
                x=0.05 * resolution_factor,
                y=(self.world.y_semidim + hight_c) * resolution_factor,
                font_size=14,
            )
            xform = rendering.Transform()
            geom.add_attr(xform)  
            geoms.append(geom)
            
            # Mean velocity
            # mean_vel = torch.vstack([a.state.vel for a in self.world.agents]).norm(dim=-1).mean()
            # geom = rendering.TextLine(
            #     text=f"Mean velocity: {mean_vel:.2f} m/s",
            #     x=1.68 * resolution_factor,
            #     y=(self.world.y_semidim + hight_b) * resolution_factor,
            #     font_size=14,
            # )
            # xform = rendering.Transform()
            # geom.add_attr(xform)  
            # geoms.append(geom)

            # Mean deviation from lane center line
            # mean_deviation_from_center_line = self.distances.ref_paths[0].mean()
            # geom = rendering.TextLine(
            #     text=f"Mean deviation: {mean_deviation_from_center_line:.2f} m",
            #     x=3.15 * resolution_factor,
            #     y=(self.world.y_semidim + hight_b) * resolution_factor,
            #     font_size=14,
            # )
            # xform = rendering.Transform()
            # geom.add_attr(xform)  
            # geoms.append(geom)
            
        for agent_i in range(self.n_agents):
            # # Visualize goal
            # if not self.ref_paths_agent_related.is_loop[env_index, agent_i]:
            #     circle = rendering.make_circle(radius=self.thresholds.reach_goal, filled=True)
            #     xform = rendering.Transform()
            #     circle.add_attr(xform)
            #     xform.set_translation(
            #         self.ref_paths_agent_related.long_term[env_index, agent_i, -1, 0], 
            #         self.ref_paths_agent_related.long_term[env_index, agent_i, -1, 1]
            #     )
            #     circle.set_color(*colors[agent_i])
            #     geoms.append(circle)

            # Visualize short-term reference paths of agents
            # if self.parameters.is_visualize_short_term_path & (agent_i == 0):
            if self.parameters.is_visualize_short_term_path:
                geom = rendering.PolyLine(
                    v = self.ref_paths_agent_related.short_term[env_index, agent_i],
                    close=False,
                )
                xform = rendering.Transform()
                geom.add_attr(xform)            
                geom.set_color(*colors[agent_i])
                geoms.append(geom)
                
                for i_p in self.ref_paths_agent_related.short_term[env_index, agent_i]:
                    circle = rendering.make_circle(radius=0.01, filled=True)
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(
                        i_p[0], 
                        i_p[1]
                    )
                    circle.set_color(*colors[agent_i])
                    geoms.append(circle)
            
            # Visualize nearing points on boundaries
            if not self.parameters.is_observe_distance_to_boundaries:
                # Left boundary
                geom = rendering.PolyLine(
                    v = self.ref_paths_agent_related.nearing_points_left_boundary[env_index, agent_i],
                    close=False,
                )
                xform = rendering.Transform()
                geom.add_attr(xform)            
                geom.set_color(*colors[agent_i])
                geoms.append(geom)
                
                for i_p in self.ref_paths_agent_related.nearing_points_left_boundary[env_index, agent_i]:
                    circle = rendering.make_circle(radius=0.01, filled=True)
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(
                        i_p[0], 
                        i_p[1]
                    )
                    circle.set_color(*colors[agent_i])
                    geoms.append(circle)
                    
                # Right boundary
                geom = rendering.PolyLine(
                    v = self.ref_paths_agent_related.nearing_points_right_boundary[env_index, agent_i],
                    close=False,
                )
                xform = rendering.Transform()
                geom.add_attr(xform)            
                geom.set_color(*colors[agent_i])
                geoms.append(geom)
                
                for i_p in self.ref_paths_agent_related.nearing_points_right_boundary[env_index, agent_i]:
                    circle = rendering.make_circle(radius=0.01, filled=True)
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(
                        i_p[0], 
                        i_p[1]
                    )
                    circle.set_color(*colors[agent_i])
                    geoms.append(circle)
            
            # Agent IDs
            geom = rendering.TextLine(
                text=f"{agent_i}",
                x=(self.world.agents[agent_i].state.pos[env_index, 0] / self.world.x_semidim) * self.viewer_size[0],
                y=(self.world.agents[agent_i].state.pos[env_index, 1] / self.world.y_semidim) * self.viewer_size[1],
                # x=(self.world.agents[agent_i].state.pos[env_index, 0] - self.render_origin[0] + self.world.x_semidim / 2) * resolution_factor / self.viewer_zoom,
                # y=(self.world.agents[agent_i].state.pos[env_index, 1] - self.render_origin[1] + self.world.y_semidim / 2) * resolution_factor / self.viewer_zoom,
                font_size=14,
            )
            xform = rendering.Transform()
            geom.add_attr(xform)  
            geoms.append(geom)
                
            # Lanelet boundaries of agents' reference path
            if self.parameters.is_visualize_lane_boundary:
                if agent_i == 0:
                    # Left boundary
                    geom = rendering.PolyLine(
                        v = self.ref_paths_agent_related.left_boundary[env_index, agent_i],
                        close=False,
                    )
                    xform = rendering.Transform()
                    geom.add_attr(xform)            
                    geom.set_color(*colors[agent_i])
                    geoms.append(geom)
                    # Right boundary
                    geom = rendering.PolyLine(
                        v = self.ref_paths_agent_related.right_boundary[env_index, agent_i],
                        close=False,
                    )
                    xform = rendering.Transform()
                    geom.add_attr(xform)            
                    geom.set_color(*colors[agent_i])
                    geoms.append(geom)
                    # Entry
                    geom = rendering.PolyLine(
                        v = self.ref_paths_agent_related.entry[env_index, agent_i],
                        close=False,
                    )
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.set_color(*colors[agent_i])
                    geoms.append(geom)
                    # Exit
                    geom = rendering.PolyLine(
                        v = self.ref_paths_agent_related.exit[env_index, agent_i],
                        close=False,
                    )
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.set_color(*colors[agent_i])
                    geoms.append(geom)
            
        return geoms


if __name__ == "__main__":
    scenario = ScenarioRoadTraffic()
    render_interactively(
        scenario=scenario, control_two_agents=False, shared_reward=False,
    )
    