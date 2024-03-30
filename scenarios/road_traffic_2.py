import os
import sys
import time
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

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World, Line
# from vmas.simulator.dynamics.kinematic_bicycle import KinematicBicycle
from vmas.simulator.scenario import BaseScenario
from utilities.colors import Color

import matplotlib.pyplot as plt

from utilities.helper_training import Parameters


from utilities.helper_scenario import Distances, Normalizers, Observations, Penalties, ReferencePathsAgentRelated, ReferencePathsMapRelated, Rewards, Thresholds, Collisions, Timer, Constants, StateBuffer, InitialStateBuffer, Prioritization, Noise, exponential_decreasing_fcn, get_distances_between_agents, get_perpendicular_distances, get_rectangle_corners, get_short_term_reference_path, interX, angle_eliminate_two_pi, transform_from_global_to_local_coordinate

from utilities.kinematic_bicycle import KinematicBicycle

# Get road data
from utilities.get_cpm_lab_map import get_map_data
from utilities.get_reference_paths import get_reference_paths

## Simulation parameters 
n_agents = 10                    # The number of agents
dt = 0.05                        # Sample time in [s]
max_steps = 1000                # Maximum simulation steps
is_real_time_rendering = True   # Simulation will be paused at each time step for real-time rendering
agent_max_speed = 1.0           # Maximum allowed speed in [m/s]
agent_max_steering_angle = 35   # Maximum allowed steering angle in degree
agent_mass = 0.5                # The mass of each agent in [kg]

## Geometry
world_x_dim = 4.5               # The x-dimension of the world in [m]
world_y_dim = 4.0               # The y-dimension of the world in [m]
agent_width = 0.08              # The width of the agent in [m]
agent_length = 0.16             # The length of the agent in [m]
wheelbase_front = agent_length / 2                              # Front wheelbase in [m]
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

threshold_near_other_agents_c2c_high = agent_length     # Threshold beneath which agents will started be 
                                                        # penalized for being too close to other agents (for center-to-center distance)
threshold_near_other_agents_c2c_low = agent_width / 2   # Threshold above which agents will be penalized (for center-to-center distance, 
                                                        # if a c2c distance is less than the half of the agent width, they are colliding, which will be penalized by another penalty)

threshold_near_other_agents_MTV_high = agent_length  # Threshold beneath which agents will be penalized for 
                                                    # being too close to other agents (for MTV-based distance)
threshold_near_other_agents_MTV_low = 0             # Threshold above which agents will be penalized for
                                                    # being too close to other agents (for MTV-based distance)
                                                    
threshold_no_reward_if_too_close_to_boundaries = agent_width / 10
threshold_no_reward_if_too_close_to_other_agents = agent_width / 6


## Visualization
viewer_size = (world_x_dim * 200, world_y_dim * 200) # TODO Check if we can use a fix camera view in vmas
viewer_zoom = 1
is_testing_mode = False             # In testing mode, collisions do not lead to the termination of the simulation 
is_visualize_short_term_path = True

# Reference path
n_points_short_term = 3             # The number of points on short-term reference paths
sample_interval = 2                 # Integer, sample interval from the long-term reference path for the short-term reference paths 
max_ref_path_points = 200           # The estimated maximum points on the reference path

## Observation
is_partial_observation = True         # Set to True if each agent can only observe a subset of other agents, i.e., limitations on sensor range are considered
                                    # Note that this also reduces the observation size, which may facilitate training
n_nearing_agents_observed = 3       # The number of most nearing agents to be observed by each agent. This parameter will be used if `is_partial_observation = True`
is_global_coordinate_sys = True    # Global coordinate system (top-down view) or local coordinate system (ego view)
is_add_noise = False                # TODO Add noise to observations to avoid overfitting
noise_level = 0.2 * agent_width     # Noise will be generated by the standary normal distribution. This parameter controls the noise level

n_stored_steps = 5      # The number of steps to store (include the current step). At least one
n_observed_steps = 1    # The number of steps to observe (include the current step). At least one, and at most `n_stored_steps`

# Training parameters
training_strategy = "2" # One of {"1", "2", "3", "4"}
                        # "1": Train in a single, comprehensive scenario
                        # "2": Train in a single, comprehensive scenario with prioritized replay buffer
                        # "3": Train in a single, comprehensive scenario with challenging initial state buffer
                        # "4": Training in mixed scenarios
buffer_size = 100 # Used only when training_strategy == "3"
n_steps_before_recording = 10 # The states of agents at time step `current_time_step - n_steps_before_recording` before collisions will be recorded and used later when resetting the envs
n_steps_stored = n_steps_before_recording # Store previous `n_steps_stored` steps of states
probability_record = 1.0 # Probability of recording a collision-event into the buffer
probability_use_recording = 0.1 # Probability of using an recording when resetting an env

colors = [
    Color.blue100, Color.purple100, Color.violet100, Color.bordeaux100, Color.red100, Color.orange100, Color.maygreen100, Color.green100, Color.turquoise100, Color.petrol100, Color.yellow100, Color.magenta100, Color.black100,
    Color.blue50, Color.purple50, Color.violet50, Color.bordeaux50, Color.red50, Color.orange50, Color.maygreen50, Color.green50, Color.turquoise50, Color.petrol50, Color.yellow50, Color.magenta50, Color.black50,
] # Each agent will get a different color

class ScenarioRoadTraffic(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        print("[DEBUG] make_world() road_traffic")
        # device = torch.device("mps") # For mac with m chip to use GPU acceleration (however, seems not be fully supported by VMAS)
        self.shared_reward = kwargs.get("shared_reward", False)
        
        width = kwargs.get("width", agent_width)
        l_f = kwargs.get("l_f", wheelbase_front)    # Front wheelbase
        l_r = kwargs.get("l_r", wheelbase_rear)     # Rear wheelbase
        max_steering_angle = kwargs.get("max_steering_angle", torch.deg2rad(torch.tensor(agent_max_steering_angle, device=device, dtype=torch.float32)))
        max_speed = kwargs.get("max_speed", agent_max_speed)
        
        self.viewer_size = viewer_size
        self.viewer_zoom = viewer_zoom
        
        self.viewer_bound = torch.tensor(
            # [-0.1, world_x_dim + 0.1, -0.1, world_y_dim + 0.1],
            [0, world_x_dim, 0, world_y_dim],
            device=device,
            dtype=torch.float32,
        )

        # Specify parameters if not given
        if not hasattr(self, "parameters"):
            self.parameters = Parameters(
                n_agents=n_agents,
                is_partial_observation=is_partial_observation,
                is_testing_mode=is_testing_mode,
                is_visualize_short_term_path=is_visualize_short_term_path,
                max_steps=max_steps,
                is_global_coordinate_sys=is_global_coordinate_sys,
                training_strategy=training_strategy,
                n_nearing_agents_observed=n_nearing_agents_observed,
                is_real_time_rendering=is_real_time_rendering,
                n_points_short_term=n_points_short_term,
                dt=dt,
            )
            
        # Parameter adjustment to meet simulation requirements
        if self.parameters.training_strategy == "4":
            self.parameters.n_agents = 4
            print(f"\033[91mChange the number of agents to {self.parameters.n_agents}, as the training strategy is 'train in mixed scenarios'.\033[0m") # Print in red

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
        self.map_data = get_map_data(device=device)
        # Long-term reference path
        reference_paths_all, reference_paths_intersection, reference_paths_merge_in, reference_paths_merge_out = get_reference_paths(self.n_agents, self.map_data) 

        # Determine the maximum number of points on the reference path
        if self.parameters.training_strategy in ("1", "2", "3"):
            # Train in one single, comprehensive scenario
            max_ref_path_points = max([
                ref_p["center_line"].shape[0] for ref_p in reference_paths_all
            ]) + self.parameters.n_points_short_term * sample_interval + 2 # Append a smaller buffer
        else:
            # Train in mixed scenarios 
            max_ref_path_points = max([
                ref_p["center_line"].shape[0] for ref_p in reference_paths_intersection + reference_paths_merge_in + reference_paths_merge_out
            ]) + self.parameters.n_points_short_term * sample_interval + 2 # Append a smaller buffer
            
        # Get all reference paths
        self.ref_paths_map_related = ReferencePathsMapRelated(
            long_term_all=reference_paths_all,
            long_term_intersection=reference_paths_intersection,
            long_term_merge_in=reference_paths_merge_in,
            long_term_merge_out=reference_paths_merge_out,
            point_extended_all=torch.zeros((len(reference_paths_all), self.parameters.n_points_short_term * sample_interval, 2), device=device, dtype=torch.float32), # Not interesting, may be useful in the future
            point_extended_intersection=torch.zeros((len(reference_paths_intersection), self.parameters.n_points_short_term * sample_interval, 2), device=device, dtype=torch.float32),
            point_extended_merge_in=torch.zeros((len(reference_paths_merge_in), self.parameters.n_points_short_term * sample_interval, 2), device=device, dtype=torch.float32),
            point_extended_merge_out=torch.zeros((len(reference_paths_merge_out), self.parameters.n_points_short_term * sample_interval, 2), device=device, dtype=torch.float32),
            sample_interval=torch.tensor(sample_interval, device=device, dtype=torch.int32),
        )
        
        # Extended the reference path by several points along the last vector of the center line 
        # TODO Check is this is necessary
        idx_broadcasting_entend = torch.arange(1, self.parameters.n_points_short_term * sample_interval + 1, device=device, dtype=torch.int32).unsqueeze(1)
        for idx, i_path in enumerate(reference_paths_all):
            center_line_i = i_path["center_line"]
            direction = center_line_i[-1] - center_line_i[-2]
            self.ref_paths_map_related.point_extended_all[idx, :] = center_line_i[-1] + idx_broadcasting_entend * direction
        for idx, i_path in enumerate(reference_paths_intersection):
            center_line_i = i_path["center_line"]
            direction = center_line_i[-1] - center_line_i[-2]
            self.ref_paths_map_related.point_extended_intersection[idx, :] = center_line_i[-1] + idx_broadcasting_entend * direction            
        for idx, i_path in enumerate(reference_paths_merge_in):
            center_line_i = i_path["center_line"]
            direction = center_line_i[-1] - center_line_i[-2]
            self.ref_paths_map_related.point_extended_merge_in[idx, :] = center_line_i[-1] + idx_broadcasting_entend * direction
        for idx, i_path in enumerate(reference_paths_merge_out):
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
            n_points_short_term=torch.tensor(self.parameters.n_points_short_term, device=device, dtype=torch.int32),
            scenario_id=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32), # Which scenarios agents are (1 for intersection, 2 for merge-in, 3 for merge-out)
            path_id=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32), # Which paths agents are
            point_id=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32), # Which points agents are
        )
        
        # The shape of each car-like robots is considered a rectangle with 4 corners. 
        # The first corner is repeated at the end to close the shape. 
        self.corners = torch.zeros((batch_dim, self.n_agents, 5, 2), device=device, dtype=torch.float32) 
 
        weighting_ref_directions = torch.linspace(1, 0.2, steps=self.ref_paths_agent_related.n_points_short_term, device=device, dtype=torch.float32)
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
            weighting_deviate_from_ref_path=self.map_data["mean_lane_width"] / 2,
            near_boundary=torch.tensor(penalty_near_boundary, device=device, dtype=torch.float32),
            near_other_agents=torch.tensor(penalty_near_other_agents, device=device, dtype=torch.float32),
            collide_with_agents=torch.tensor(penalty_collide_with_agents, device=device, dtype=torch.float32),
            collide_with_boundaries=torch.tensor(penalty_collide_with_boundaries, device=device, dtype=torch.float32),
            change_steering=torch.tensor(penalty_change_steering, device=device, dtype=torch.float32),
            time=torch.tensor(penalty_time, device=device, dtype=torch.float32),
        )
        
        self.observations = Observations(
            is_partial=torch.tensor(self.parameters.is_partial_observation, device=device, dtype=torch.bool),
            is_global_coordinate_sys=torch.tensor(self.parameters.is_global_coordinate_sys, device=device, dtype=torch.bool),
            n_nearing_agents=torch.tensor(self.parameters.n_nearing_agents_observed, device=device, dtype=torch.int32),
            is_add_noise=torch.tensor(is_add_noise, device=device, dtype=torch.bool),
            noise_level=torch.tensor(noise_level, device=device, dtype=torch.float32),
            n_stored_steps=torch.tensor(n_stored_steps, device=device, dtype=torch.int32),
            n_observed_steps=torch.tensor(n_observed_steps, device=device, dtype=torch.int32),
            nearing_agents_indices=torch.zeros((batch_dim, self.n_agents, self.parameters.n_nearing_agents_observed), device=device, dtype=torch.int32)
        )
        assert self.observations.n_stored_steps >= 1, "The number of stored steps should be at least 1."
        assert self.observations.n_observed_steps >= 1, "The number of observed steps should be at least 1."
        assert self.observations.n_stored_steps >= self.observations.n_observed_steps, "The number of stored steps should be greater or equal than the number of observed steps."
        
        if not self.observations.is_global_coordinate_sys:
            # Each agent observe other agents locally (ego view)
            self.observations.past_pri = torch.zeros((batch_dim, n_stored_steps, self.n_agents, self.n_agents), device=device, dtype=torch.bool) # A True means the corresponding agent has a higher priotiry than the ego agent
            self.observations.past_pos = torch.zeros((batch_dim, n_stored_steps, self.n_agents, self.n_agents, 2), device=device, dtype=torch.float32)
            self.observations.past_rot = torch.zeros((batch_dim, n_stored_steps, self.n_agents, self.n_agents), device=device, dtype=torch.float32)
            self.observations.past_corners = torch.zeros((batch_dim, n_stored_steps, self.n_agents, self.n_agents, 4, 2), device=device, dtype=torch.float32)
            self.observations.past_vel = torch.zeros((batch_dim, n_stored_steps, self.n_agents, self.n_agents, 2), device=device, dtype=torch.float32)
            self.observations.past_short_term_ref_points = torch.zeros((batch_dim, n_stored_steps, self.n_agents, self.n_agents, self.ref_paths_agent_related.n_points_short_term, 2), device=device, dtype=torch.float32)
        else:
            self.observations.past_pri = torch.zeros((batch_dim, n_stored_steps, self.n_agents), device=device, dtype=torch.float32)
            # All agents are observed globally (top-down view)
            self.observations.past_pos = torch.zeros((batch_dim, n_stored_steps, self.n_agents, 2), device=device, dtype=torch.float32)
            self.observations.past_rot = torch.zeros((batch_dim, n_stored_steps, self.n_agents), device=device, dtype=torch.float32)
            self.observations.past_corners = torch.zeros((batch_dim, n_stored_steps, self.n_agents, 4, 2), device=device, dtype=torch.float32)
            self.observations.past_vel = torch.zeros((batch_dim, n_stored_steps, self.n_agents, 2), device=device, dtype=torch.float32)
            self.observations.past_short_term_ref_points = torch.zeros((batch_dim, n_stored_steps, self.n_agents, self.ref_paths_agent_related.n_points_short_term, 2), device=device, dtype=torch.float32)

        self.observations.past_action_vel = torch.zeros((batch_dim, n_stored_steps, self.n_agents), device=device, dtype=torch.float32)
        self.observations.past_action_steering = torch.zeros((batch_dim, n_stored_steps, self.n_agents), device=device, dtype=torch.float32)
        self.observations.past_distance_to_ref_path = torch.zeros((batch_dim, n_stored_steps, self.n_agents), device=device, dtype=torch.float32)
        self.observations.past_distance_to_boundaries = torch.zeros((batch_dim, n_stored_steps, self.n_agents), device=device, dtype=torch.float32)
        self.observations.past_distance_to_left_boundary = torch.zeros((batch_dim, n_stored_steps, self.n_agents), device=device, dtype=torch.float32)
        self.observations.past_distance_to_right_boundary = torch.zeros((batch_dim, n_stored_steps, self.n_agents), device=device, dtype=torch.float32)
        self.observations.past_distance_to_agents = torch.zeros((batch_dim, n_stored_steps, self.n_agents, self.n_agents), device=device, dtype=torch.float32)

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
            priority=torch.tensor(self.n_agents, device=device, dtype=torch.float32),
        )
        
        # Distances to boundaries and reference path, and also the closest point on the reference paths of agents
        distance_type = "MTV" # One of {"c2c", "MTV"}
        self.distances = Distances(
            type = distance_type, # Type of distances between agents. One of {"c2c", "MTV"}
            agents=torch.zeros(world.batch_dim, self.n_agents, self.n_agents, dtype=torch.float32),
            left_boundaries=torch.zeros((batch_dim, self.n_agents, 1 + 4), device=device, dtype=torch.float32), # The first entry for the center, the last 4 entries for the four corners
            right_boundaries=torch.zeros((batch_dim, self.n_agents, 1 + 4), device=device, dtype=torch.float32),
            boundaries=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.float32),
            ref_paths=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.float32),
            closest_point_on_ref_path=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32)
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
            distance_mask_agents=self.normalizers.pos[0],
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
            empty_actions=torch.zeros((batch_dim, agent.action.action_size), device=device, dtype=torch.float32),
            mask_pos=torch.tensor(1, device=device, dtype=torch.float32),
            mask_zero=torch.tensor(0, device=device, dtype=torch.float32),
            mask_one=torch.tensor(1, device=device, dtype=torch.float32),
            reset_agent_min_distance=torch.tensor((l_f+l_r) ** 2 + width ** 2, device=device, dtype=torch.float32).sqrt() * 1.2,
            reset_scenario_probabilities=torch.tensor([0.8, 0.1, 0.1], device=device, dtype=torch.float32), # 1 for intersection, 2 for merge-in, 3 for merge-out scenario
            # reset_scenario_probabilities=torch.tensor([0.7, 0.15, 0.15], device=device, dtype=torch.float32), # 1 for intersection, 2 for merge-in, 3 for merge-out scenario TODO Check if nevessary
        )
        
        # Initialize collision matrix
        self.collisions = Collisions(
            with_agents=torch.zeros((world.batch_dim, self.n_agents, self.n_agents), device=device, dtype=torch.bool),
            with_lanelets=torch.zeros((world.batch_dim, self.n_agents), device=device, dtype=torch.bool),
            with_entry_segments=torch.zeros((world.batch_dim, self.n_agents), device=device, dtype=torch.bool),
            with_exit_segments=torch.zeros((world.batch_dim, self.n_agents), device=device, dtype=torch.bool),
        )
        
        self.prioritization = Prioritization(
            values=torch.arange(1, self.n_agents + 1, device=device, dtype=torch.int32).repeat(batch_dim, 1),
        )
        
        self.initial_state_buffer = InitialStateBuffer( # Used only when "training_strategy == '4'"
            buffer_size=torch.tensor(buffer_size, device=device, dtype=torch.int32),
            probability_record=torch.tensor(probability_record, device=device, dtype=torch.float32),
            probability_use_recording=torch.tensor(probability_use_recording, device=device, dtype=torch.float32),
            buffer=torch.zeros((buffer_size, self.n_agents, 8), device=device, dtype=torch.float32), # [pos_x, pos_y, rot, vel_x, vel_y, scenario_id, path_id, point_id]
        )

        # Store the states of agents at previous several time steps
        self.state_buffer = StateBuffer(
            buffer_size=torch.tensor(n_steps_stored, device=device, dtype=torch.int32),
            buffer=torch.zeros((n_steps_stored, batch_dim, self.n_agents, 8), device=device, dtype=torch.float32), # [pos_x, pos_y, rot, vel_x, vel_y, scenario_id, path_id, point_id],
        )
        
        noise_pri_level = 1 / self.n_agents * 0.2 if self.observations.is_global_coordinate_sys else 0.1
        
        self.noise = Noise(
            vel=torch.zeros((batch_dim, self.observations.n_observed_steps), device=device, dtype=torch.float32),
            ref=torch.zeros((batch_dim, self.observations.n_observed_steps, self.ref_paths_agent_related.n_points_short_term, 2), device=device, dtype=torch.float32),
            dis_ref=torch.zeros((batch_dim, self.observations.n_observed_steps), device=device, dtype=torch.float32),
            dis_lanelets=torch.zeros((batch_dim, self.observations.n_observed_steps), device=device, dtype=torch.float32),
            other_agents_pri=torch.zeros((batch_dim, self.observations.n_observed_steps, self.observations.n_nearing_agents, 1), device=device, dtype=torch.float32),
            other_agents_pos=torch.zeros((batch_dim, self.observations.n_observed_steps, self.observations.n_nearing_agents, 2), device=device, dtype=torch.float32),
            other_agents_rot=torch.zeros((batch_dim, self.observations.n_observed_steps, self.observations.n_nearing_agents, 1), device=device, dtype=torch.float32),
            other_agents_vel=torch.zeros((batch_dim, self.observations.n_observed_steps, self.observations.n_nearing_agents, 2), device=device, dtype=torch.float32),
            other_agents_dis=torch.zeros((batch_dim, self.observations.n_observed_steps, self.observations.n_nearing_agents, 1), device=device, dtype=torch.float32),
            level_vel=torch.tensor(0.1 * max_speed, device=device, dtype=torch.float32),
            level_pos=torch.tensor(0.1 * agent_width, device=device, dtype=torch.float32),
            level_rot=torch.tensor(0.1, device=device, dtype=torch.float32),
            level_dis=torch.tensor(0.1 * agent_width, device=device, dtype=torch.float32),
            level_pri=torch.tensor(noise_pri_level, device=device, dtype=torch.float32),
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

        # if hasattr(self, "training_info"):
        #     print(self.training_info["agents"]["episode_reward"].mean())
        is_reset_single_agent = agent_index is not None
        
        if is_reset_single_agent:
            assert env_index is not None
        
        # env_index = slice(None) if env_index is None else env_index # `slice(None)` is equivalent to `:`

        for env_index in [env_index] if env_index is not None else range(self.world.batch_dim):
                
            # Begining of a new simulation (only record for the first env)
            if env_index == 0:
                self.timer.step_duration[:] = 0
                self.timer.start = time.time()
                self.timer.step_begin = time.time()
                self.timer.end = 0
                
            if not is_reset_single_agent:
                # Each time step of a simulation
                self.timer.step[env_index] = 0
                
            # TODO Prioritization modul
            # Reset the priorities of agents
            self.prioritization.values[env_index, :] = self.prioritization.values[env_index, :]

            # Get the center line and boundaries of the long-term reference path for each agent
            if self.parameters.training_strategy in {"1", "2", "3"}:
                ref_paths_scenario = self.ref_paths_map_related.long_term_all
                extended_points = self.ref_paths_map_related.point_extended_all
                self.ref_paths_agent_related.scenario_id[env_index, :] = 0 # 0 for the whole map, 1 for intersection, 2 for merge-in, 3 for merge-out scenario
            else:
                if is_reset_single_agent:
                    scenario_id = self.ref_paths_agent_related.scenario_id[env_index, agent_index] # Keep the same scenario
                else:
                    scenario_id = torch.multinomial(self.constants.reset_scenario_probabilities, 1, replacement=True).item() + 1 # A random interger {1, 2, 3}
                    self.ref_paths_agent_related.scenario_id[env_index, :] = scenario_id
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
            
            if ((self.parameters.training_strategy == "3") and 
                (torch.rand(1) < self.initial_state_buffer.probability_use_recording) and 
                (self.initial_state_buffer.valid_size >= 1)):
                # Use initial state buffer
                is_use_state_buffer = True
                initial_state = self.initial_state_buffer.get_random()
                self.ref_paths_agent_related.scenario_id[env_index] = initial_state[:, self.initial_state_buffer.idx_scenario] # Update
                self.ref_paths_agent_related.path_id[env_index] = initial_state[:, self.initial_state_buffer.idx_path] # Update
                self.ref_paths_agent_related.point_id[env_index] = initial_state[:, self.initial_state_buffer.idx_point] # Update
                print(f"\033[91mReset with path ids: {initial_state[:, -2]}'.\033[0m") # Print in red
            else:
                is_use_state_buffer = False
                
            for i_agent in range(self.n_agents) if not is_reset_single_agent else agent_index.unsqueeze(0):
                if is_use_state_buffer:
                    path_id = initial_state[i_agent, self.initial_state_buffer.idx_path].int()
                    ref_path = ref_paths_scenario[path_id]
                
                    agents[i_agent].set_pos(initial_state[i_agent, 0:2], batch_index=env_index)
                    agents[i_agent].set_rot(initial_state[i_agent, 2], batch_index=env_index)
                    agents[i_agent].set_vel(initial_state[i_agent, 3:5], batch_index=env_index)
                    
                else:
                    is_feasible_initial_position_found = False
                    random_count = 0
                    # Ramdomly generate initial states for each agent
                    while not is_feasible_initial_position_found:
                        # if random_count >= 1:
                        #     print(f"Resetting agent(s): random_count = {random_count}.")
                        # random_count += 1
                        path_id = torch.randint(0, len(ref_paths_scenario), (1,)).item() # Select randomly a path
                        self.ref_paths_agent_related.path_id[env_index, i_agent] = path_id # Update
                        ref_path = ref_paths_scenario[path_id]
                        
                        num_points = ref_path["center_line"].shape[0]
                        random_point_id = torch.randint(3, num_points-5, (1,)).item() # Random point on the center as initial position TODO Find the suitable range
                        self.ref_paths_agent_related.point_id[env_index, i_agent] = random_point_id # Update
                        position_start = ref_path["center_line"][random_point_id]
                        agents[i_agent].set_pos(position_start, batch_index=env_index)

                        # Check if the initial position is feasible
                        if not is_reset_single_agent:
                            if i_agent == 0:
                                # The initial position of the first agent is always feasible
                                is_feasible_initial_position_found = True
                                continue
                            else:
                                positions = torch.stack([self.world.agents[i].state.pos[env_index] for i in range(i_agent+1)])
                        else:
                            # Check if the initial position of the agent to be reset is collision-free with other agents
                            positions = torch.stack([self.world.agents[i].state.pos[env_index] for i in range(self.n_agents)])
                                            
                        diff_sq = (positions[i_agent, :] - positions) ** 2 # Calculate pairwise squared differences in positions
                        initial_mutual_distances_sq = torch.sum(diff_sq, dim=-1)
                        initial_mutual_distances_sq[i_agent] = torch.max(initial_mutual_distances_sq) + 1 # Set self-to-self distance to a sufficiently high value
                        min_distance_sq = torch.min(initial_mutual_distances_sq)
                        
                        is_feasible_initial_position_found = min_distance_sq >= (self.constants.reset_agent_min_distance ** 2)

                    rot_start = ref_path["center_line_yaw"][random_point_id]
                    vel_start_abs = torch.rand(1, dtype=torch.float32, device=self.world.device) * agents[i_agent].max_speed # Random initial velocity
                    vel_start = torch.hstack([vel_start_abs * torch.cos(rot_start), vel_start_abs * torch.sin(rot_start)])

                    agents[i_agent].set_rot(rot_start, batch_index=env_index)
                    agents[i_agent].set_vel(vel_start, batch_index=env_index)
                
                # Long-term reference paths for agents
                n_points_long_term = ref_path["center_line"].shape[0]
                
                self.ref_paths_agent_related.long_term[env_index, i_agent, 0:n_points_long_term, :] = ref_path["center_line"]
                self.ref_paths_agent_related.long_term[env_index, i_agent, n_points_long_term:(n_points_long_term+self.ref_paths_agent_related.n_points_short_term * self.ref_paths_map_related.sample_interval), :] = extended_points[path_id, :, :]
                self.ref_paths_agent_related.long_term[env_index, i_agent, (n_points_long_term+self.ref_paths_agent_related.n_points_short_term * self.ref_paths_map_related.sample_interval):, :] = extended_points[path_id, -1, :]
                self.ref_paths_agent_related.n_points_long_term[env_index, i_agent] = n_points_long_term
                
                self.ref_paths_agent_related.long_term_vec_normalized[env_index, i_agent, 0:n_points_long_term-1, :] = ref_path["center_line_vec_normalized"]
                self.ref_paths_agent_related.long_term_vec_normalized[env_index, i_agent, (n_points_long_term-1):(n_points_long_term-1+self.ref_paths_agent_related.n_points_short_term * self.ref_paths_map_related.sample_interval), :] = ref_path["center_line_vec_normalized"][-1, :]

                n_points_left_b = ref_path["left_boundary_shared"].shape[0]
                self.ref_paths_agent_related.left_boundary[env_index, i_agent, 0:n_points_left_b, :] = ref_path["left_boundary_shared"]
                self.ref_paths_agent_related.left_boundary[env_index, i_agent, n_points_left_b:, :] = ref_path["left_boundary_shared"][-1, :]
                self.ref_paths_agent_related.n_points_left_b[env_index, i_agent] = n_points_left_b
                
                n_points_right_b = ref_path["right_boundary_shared"].shape[0]
                self.ref_paths_agent_related.right_boundary[env_index, i_agent, 0:n_points_right_b, :] = ref_path["right_boundary_shared"]
                self.ref_paths_agent_related.right_boundary[env_index, i_agent, n_points_right_b:, :] = ref_path["right_boundary_shared"][-1, :]
                self.ref_paths_agent_related.n_points_right_b[env_index, i_agent] = n_points_right_b

                self.ref_paths_agent_related.entry[env_index, i_agent, 0, :] = ref_path["left_boundary_shared"][0, :]
                self.ref_paths_agent_related.entry[env_index, i_agent, 1, :] = ref_path["right_boundary_shared"][0, :]

                self.ref_paths_agent_related.exit[env_index, i_agent, 0, :] = ref_path["left_boundary_shared"][-1, :]
                self.ref_paths_agent_related.exit[env_index, i_agent, 1, :] = ref_path["right_boundary_shared"][-1, :]
                
                self.ref_paths_agent_related.is_loop[env_index, i_agent] = ref_path["is_loop"]


            for i_agent in range(self.n_agents) if not is_reset_single_agent else agent_index.unsqueeze(0):
                
                # Distances to reference paths
                self.distances.ref_paths[env_index, i_agent], self.distances.closest_point_on_ref_path[env_index, i_agent] = get_perpendicular_distances(
                    point=agents[i_agent].state.pos[env_index, :], 
                    polyline=self.ref_paths_agent_related.long_term[env_index, i_agent],
                    n_points_long_term=self.ref_paths_agent_related.n_points_long_term[env_index, i_agent],
                )
                # Distances from center to left boundaries
                center_2_left_b, _ = get_perpendicular_distances(
                    point=agents[i_agent].state.pos[env_index, :], 
                    polyline=self.ref_paths_agent_related.left_boundary[env_index, i_agent],
                    n_points_long_term=self.ref_paths_agent_related.n_points_left_b[env_index, i_agent],
                )
                self.distances.left_boundaries[env_index, i_agent, 0] = center_2_left_b - (agents[i_agent].shape.width / 2)
                # Distances from center to right boundaries
                center_2_right_b, _ = get_perpendicular_distances(
                    point=agents[i_agent].state.pos[env_index, :], 
                    polyline=self.ref_paths_agent_related.right_boundary[env_index, i_agent],
                    n_points_long_term=self.ref_paths_agent_related.n_points_right_b[env_index, i_agent],
                )
                self.distances.right_boundaries[env_index, i_agent, 0] = center_2_right_b - (agents[i_agent].shape.width / 2)
                
                # Calculate the positions of the four corners of the agents
                self.corners[env_index, i_agent] = get_rectangle_corners(
                    center=agents[i_agent].state.pos[env_index, :], 
                    yaw=agents[i_agent].state.rot[env_index, :], 
                    width=agents[i_agent].shape.width, 
                    length=agents[i_agent].shape.length,
                    is_close_shape=True
                )
                
                # Distances of the four corners of the agent to its left and right lanelet boundaries
                for c_i in range(4):
                    self.distances.left_boundaries[env_index, i_agent, c_i+1], _ = get_perpendicular_distances(
                        point=self.corners[env_index, i_agent, c_i, :], 
                        polyline=self.ref_paths_agent_related.left_boundary[env_index, i_agent],
                        n_points_long_term=self.ref_paths_agent_related.n_points_left_b[env_index, i_agent],
                    )
                    self.distances.right_boundaries[env_index, i_agent, c_i+1], _ = get_perpendicular_distances(
                        point=self.corners[env_index, i_agent, c_i, :], 
                        polyline=self.ref_paths_agent_related.right_boundary[env_index, i_agent],
                        n_points_long_term=self.ref_paths_agent_related.n_points_right_b[env_index, i_agent],
                    )
                # Minimum distance of the four corners of the agent to its lanelet boundaries
                self.distances.boundaries[env_index, i_agent], _ = torch.min(
                    torch.hstack(
                        (
                            self.distances.left_boundaries[env_index, i_agent],
                            self.distances.right_boundaries[env_index, i_agent]
                        )
                    ),
                    dim=-1
                ) 

                # Get the short-term reference paths
                self.ref_paths_agent_related.short_term[env_index, i_agent], self.ref_paths_agent_related.short_term_indices[env_index, i_agent] = get_short_term_reference_path(
                    reference_path=self.ref_paths_agent_related.long_term[env_index, i_agent],
                    closest_point_on_ref_path=self.distances.closest_point_on_ref_path[env_index, i_agent],
                    n_points_short_term=self.ref_paths_agent_related.n_points_short_term, 
                    device=self.world.device,
                    is_ref_path_loop=self.ref_paths_agent_related.is_loop[env_index, i_agent],
                    n_points_long_term=self.ref_paths_agent_related.n_points_long_term[env_index, i_agent],
                    sample_interval=self.ref_paths_map_related.sample_interval,
                )

            # Compute mutual distances between agents 
            # TODO Add the possibility of computing the mutual distances of agents in env `env_index`
            mutual_distances = get_distances_between_agents(self=self, distance_type=self.distances.type, is_set_diagonal=True)
            # Reset mutual distances of all envs
            self.distances.agents[env_index, :, :] = mutual_distances[env_index, :, :]

            # Reset the collision matrix
            self.collisions.with_agents[env_index, :, :] = False
            self.collisions.with_lanelets[env_index, :] = False
            self.collisions.with_entry_segments[env_index, :] = False
            self.collisions.with_exit_segments[env_index, :] = False
            
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
        
        # print(self.ref_paths_agent_related.scenario_id)

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
            
        # [update] mutual distances between agents, corners of each agent, and collision matrices
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
                self.corners[:, a_i] = get_rectangle_corners(
                    center=self.world.agents[a_i].state.pos,
                    yaw=self.world.agents[a_i].state.rot,
                    width=self.world.agents[a_i].shape.width,
                    length=self.world.agents[a_i].shape.length,
                    is_close_shape=True,
                )
                # Update the collision matrices
                if self.distances.type == "c2c":
                    for a_j in range(a_i+1, self.n_agents):
                        # Check for intersection using the interX function
                        collision_batch_index = interX(self.corners[:, a_i], self.corners[:, a_j], False)
                        self.collisions.with_agents[torch.nonzero(collision_batch_index), a_i, a_j] = True
                        self.collisions.with_agents[torch.nonzero(collision_batch_index), a_j, a_i] = True
                elif self.distances.type == "MTV":
                    # If two agents collide, their MTV-based distance is zero 
                    self.collisions.with_agents[:] = self.distances.agents == 0
                    
                # Check for collisions between agents and lanelet boundaries
                collision_with_left_boundary = interX(
                    L1=self.corners[:, a_i], 
                    L2=self.ref_paths_agent_related.left_boundary[:, a_i], 
                    is_return_points=False,
                ) # [batch_dim]
                collision_with_right_boundary = interX(
                    L1=self.corners[:, a_i], 
                    L2=self.ref_paths_agent_related.right_boundary[:, a_i],
                    is_return_points=False,
                ) # [batch_dim]
                self.collisions.with_lanelets[(collision_with_left_boundary | collision_with_right_boundary), a_i] = True

                # Check for collisions with entry or exit segments
                if not self.ref_paths_agent_related.is_loop[:, a_i].any(): # Mixed scenarios
                    self.collisions.with_entry_segments[:, a_i] = interX(
                        L1=self.corners[:, a_i],
                        L2=self.ref_paths_agent_related.entry[:, a_i],
                        is_return_points=False,
                    )
                    self.collisions.with_exit_segments[:, a_i] = interX(
                        L1=self.corners[:, a_i],
                        L2=self.ref_paths_agent_related.exit[:, a_i],
                        is_return_points=False,
                    )
                    
        # Distance of the center of the agent to its reference path
        self.distances.ref_paths[:, agent_index], self.distances.closest_point_on_ref_path[:, agent_index] = get_perpendicular_distances(
            point=agent.state.pos, 
            polyline=self.ref_paths_agent_related.long_term[:, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_long_term[:, agent_index],
        )
        # Distances from center to left boundaries
        center_2_left_b, _ = get_perpendicular_distances(
            point=agent.state.pos[:, :], 
            polyline=self.ref_paths_agent_related.left_boundary[:, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_left_b[:, agent_index],
        )
        self.distances.left_boundaries[:, agent_index, 0] = center_2_left_b - (agent.shape.width / 2)
        # Distances from center to right boundaries
        center_2_right_b, _ = get_perpendicular_distances(
            point=agent.state.pos[:, :], 
            polyline=self.ref_paths_agent_related.right_boundary[:, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_right_b[:, agent_index],
        )
        self.distances.right_boundaries[:, agent_index, 0] = center_2_right_b - (agent.shape.width / 2)
        # Distances of the four corners of the agent to its left and right lanelet boundaries
        for c_i in range(4):
            self.distances.left_boundaries[:, agent_index, c_i + 1], _ = get_perpendicular_distances(
                point=self.corners[:, agent_index, c_i, :],
                polyline=self.ref_paths_agent_related.left_boundary[:, agent_index],
                n_points_long_term=self.ref_paths_agent_related.n_points_left_b[:, agent_index],
            )
            self.distances.right_boundaries[:, agent_index, c_i + 1], _ = get_perpendicular_distances(
                point=self.corners[:, agent_index, c_i, :],
                polyline=self.ref_paths_agent_related.right_boundary[:, agent_index],
                n_points_long_term=self.ref_paths_agent_related.n_points_right_b[:, agent_index],
            )
        # Minimum distance of the four corners of the agent to its lanelet boundaries
        self.distances.boundaries[:, agent_index], _ = torch.min(
            torch.hstack(
                (
                    self.distances.left_boundaries[:, agent_index],
                    self.distances.right_boundaries[:, agent_index]
                )
            ),
            dim=-1
        )
        
        # Agents that are too close to lanelet boundaries or higher-priority agents will not receive any rewards
        too_close_to_boundaries = self.distances.boundaries[:, agent_index] <= self.thresholds.no_reward_if_too_close_to_boundaries
        too_close_to_other_agents = self.distances.agents[:, agent_index] <= self.thresholds.no_reward_if_too_close_to_other_agents
        are_others_with_higher_priority = self.prioritization.values[:, agent_index].unsqueeze(-1) > self.prioritization.values
        too_close_to_higher_priority_agents = too_close_to_other_agents & are_others_with_higher_priority
        agents_no_reward_indices = too_close_to_boundaries | too_close_to_higher_priority_agents.any(dim=-1)

        ##################################################
        ## [reward] forward movement
        ##################################################
        latest_state = self.state_buffer.get_latest(n=1)
        move_vec = (agent.state.pos - latest_state[:, agent_index, 0:2]).unsqueeze(1) # Vector of the current movement
        
        ref_points_vecs = self.ref_paths_agent_related.short_term[:, agent_index] - latest_state[:, agent_index, 0:2].unsqueeze(1) # Vectors from the previous position to the points on the short-term reference path
        move_projected = torch.sum(move_vec * ref_points_vecs, dim=-1)
        move_projected_weighted = torch.matmul(move_projected, self.rewards.weighting_ref_directions) # Put more weights on nearing reference points

        move_projected_clamped = torch.where(agents_no_reward_indices, torch.clamp(move_projected_weighted, max=0), move_projected_weighted) # Set the reward toi zero for the specified agents
        reward_movement = move_projected_clamped / (agent.max_speed * self.world.dt) * self.rewards.progress
        self.rew += reward_movement # Relative to the maximum possible movement
        
        ##################################################
        ## [reward] high velocity
        ##################################################
        # TODO Check if this reward is necessary        
        v_proj = torch.sum(agent.state.vel.unsqueeze(1) * ref_points_vecs, dim=-1).mean(-1)
        factor_moving_direction = torch.where(v_proj>0, 1, 2) # Get penalty if move in negative direction
        
        v_proj_clamped = torch.where(agents_no_reward_indices, torch.clamp(v_proj, max=0), v_proj)
        reward_vel = factor_moving_direction * v_proj_clamped / agent.max_speed * self.rewards.higth_v
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
        # TODO Remove?
        self.rew += self.distances.ref_paths[:, agent_index] / self.penalties.weighting_deviate_from_ref_path * self.penalties.deviate_from_ref_path


        ##################################################
        ## [penalty] changing steering too quick
        ##################################################
        steering_change = torch.clamp(
            (self.observations.past_action_steering[:, -1, agent_index] - self.observations.past_action_steering[:, -2, agent_index]).abs() - self.thresholds.change_steering,
            min=0,
        )
        steering_change_reward_factor = steering_change / (2 * agent.u_range[1] - 2 * self.thresholds.change_steering)
        penalty_change_steering = steering_change_reward_factor * self.penalties.change_steering
        self.rew += penalty_change_steering

        # ##################################################
        # ## [penalty] colliding with other agents (with higher priorities)
        # ##################################################
        is_collide_with_agents = self.collisions.with_agents[:, agent_index]
        is_collide_with_higher_pri_agents = is_collide_with_agents & are_others_with_higher_priority # collide with higher-priority agents
        
        penalty_collide_other_agents = is_collide_with_higher_pri_agents.any(dim=-1) * self.penalties.collide_with_agents
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
        
        self.ref_paths_agent_related.short_term[:, agent_index], self.ref_paths_agent_related.short_term_indices[:, agent_index] = get_short_term_reference_path(
            reference_path=self.ref_paths_agent_related.long_term[:, agent_index], 
            closest_point_on_ref_path=self.distances.closest_point_on_ref_path[:, agent_index],
            n_points_short_term=self.ref_paths_agent_related.n_points_short_term, 
            device=self.world.device,
            is_ref_path_loop=self.ref_paths_agent_related.is_loop[:, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_long_term[:, agent_index],
            sample_interval=self.ref_paths_map_related.sample_interval,
        )

        assert not self.rew.isnan().any(), "Rewards contain nan."
        assert not self.rew.isinf().any(), "Rewards contain inf."
        
        
        # Clamed the reward to avoid abs(reward) being too large values
        rew_clamed = torch.clamp(self.rew, min=-1, max=1)
        
        # reward_all = torch.vstack(
        #     [
        #         reward_movement,
        #         reward_vel,
        #         penalty_close_to_lanelets,
        #         penalty_close_to_agents,
        #         penalty_change_steering,
        #         penalty_collide_lanelet,
        #         penalty_collide_other_agents,
        #     ]
        # )
        # if agent_index == 0:
        #     print(" ".join(f"{v:.1f}" for v in reward_all[:, 0].view(-1)))

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
        
        if agent_index == 0: # Avoid repeated computations
            # Shift old observations by one step
            self.observations.past_pri[:, 0:-1] = self.observations.past_pri[:, 1:].clone()
            self.observations.past_pos[:, 0:-1] = self.observations.past_pos[:, 1:].clone()
            self.observations.past_rot[:, 0:-1] = self.observations.past_rot[:, 1:].clone()
            self.observations.past_vel[:, 0:-1] = self.observations.past_vel[:, 1:].clone()
            self.observations.past_corners[:, 0:-1] = self.observations.past_corners[:, 1:].clone()
            self.observations.past_short_term_ref_points[:, 0:-1] = self.observations.past_short_term_ref_points[:, 1:].clone()
            
            self.observations.past_action_vel[:, 0:-1] = self.observations.past_action_vel[:, 1:].clone()
            self.observations.past_action_steering[:, 0:-1] = self.observations.past_action_steering[:, 1:].clone()

            self.observations.past_distance_to_agents[:, 0:-1] = self.observations.past_distance_to_agents[:, 1:].clone()
            self.observations.past_distance_to_ref_path[:, 0:-1] = self.observations.past_distance_to_ref_path[:, 1:].clone()
            self.observations.past_distance_to_left_boundary[:, 0:-1] = self.observations.past_distance_to_left_boundary[:, 1:].clone()
            self.observations.past_distance_to_right_boundary[:, 0:-1] = self.observations.past_distance_to_right_boundary[:, 1:].clone()
            self.observations.past_distance_to_boundaries[:, 0:-1] = self.observations.past_distance_to_boundaries[:, 1:].clone()
            
            # Store new observation
            self.observations.past_distance_to_agents[:, -1] = self.distances.agents[:]
            self.observations.past_distance_to_ref_path[:, -1] = self.distances.ref_paths[:]
            self.observations.past_distance_to_left_boundary[:, -1], _ = torch.min(self.distances.left_boundaries, dim=-1)
            self.observations.past_distance_to_right_boundary[:, -1], _ = torch.min(self.distances.right_boundaries, dim=-1)
            self.observations.past_distance_to_boundaries[:, -1] = self.distances.boundaries

            if not self.observations.is_global_coordinate_sys:
                for a_i in range(self.n_agents):
                    pos_i = self.world.agents[a_i].state.pos
                    rot_i = self.world.agents[a_i].state.rot

                    # Store new observation - priority
                    self.observations.past_pri[:, -1, a_i] = self.prioritization.values[:, a_i].unsqueeze(-1) > self.prioritization.values[:, :] # True if other agents have higher priorities than the ego agent (lower priority values correspond to higher priorities)
                    
                    # Store new observation - position
                    self.observations.past_pos[:, -1, a_i] = transform_from_global_to_local_coordinate(
                        pos_i=pos_i,
                        pos_j=positions_global,
                        rot_i=rot_i,
                    )

                    # Store new observation - rotation
                    self.observations.past_rot[:, -1, a_i] = rotations_global - rot_i
                    
                    for a_j in range(self.n_agents):                        
                        # Store new observation - velocities
                        rot_rel = self.observations.past_rot[:, -1, a_i, a_j].unsqueeze(1)
                        vel_abs = torch.norm(self.world.agents[a_j].state.vel, dim=1).unsqueeze(1) # TODO Check if relative velocities here are better
                        self.observations.past_vel[:, -1, a_i, a_j] = torch.hstack(
                            (
                                vel_abs * torch.cos(rot_rel), 
                                vel_abs * torch.sin(rot_rel)
                            )
                        )
                        
                        # Store new observation - reference paths
                        self.observations.past_short_term_ref_points[:, -1, a_i, a_j] = transform_from_global_to_local_coordinate(
                            pos_i=pos_i,
                            pos_j=self.ref_paths_agent_related.short_term[:, a_j],
                            rot_i=rot_i,
                        )
                        
                        # Store new observation - corners
                        self.observations.past_corners[:, -1, a_i, a_j] = transform_from_global_to_local_coordinate(
                            pos_i=pos_i,
                            pos_j=self.corners[:, a_j, 0:4, :],
                            rot_i=rot_i,
                        )
            else: # Global coordinate system
                # Store new observations
                self.observations.past_pri[:, -1] = self.prioritization.values[:]
                self.observations.past_pos[:, -1] = positions_global[:]
                self.observations.past_vel[:, -1] = torch.stack([a.state.vel for a in self.world.agents], dim=1)
                self.observations.past_rot[:, -1] = rotations_global[:]
                self.observations.past_corners[:, -1] = self.corners[:, :, 0:4, :]
                self.observations.past_short_term_ref_points[:, -1] = self.ref_paths_agent_related.short_term[:]

            # Normalize
            if self.observations.is_global_coordinate_sys:
                # Priorities are only normalized if the global coordinate system is used
                self.observations.past_pos[:, -1] /= self.normalizers.pos_world
                self.observations.past_corners[:, -1] /= self.normalizers.pos_world
                self.observations.past_short_term_ref_points[:, -1] /= self.normalizers.pos_world
                self.observations.past_pri[:, -1] /= self.normalizers.priority
            else:
                self.observations.past_pos[:, -1] /= self.normalizers.pos
                self.observations.past_corners[:, -1] /= self.normalizers.pos
                self.observations.past_short_term_ref_points[:, -1] /= self.normalizers.pos
            self.observations.past_rot[:, -1] = angle_eliminate_two_pi(self.observations.past_rot[:, -1]) / self.normalizers.rot
            self.observations.past_vel[:, -1] /= self.normalizers.v
            self.observations.past_distance_to_agents[:, -1] /= self.normalizers.distance_agent
            self.observations.past_distance_to_ref_path[:, -1] /= self.normalizers.distance_ref
            self.observations.past_distance_to_left_boundary[:, -1] /= self.normalizers.distance_lanelet
            self.observations.past_distance_to_right_boundary[:, -1] /= self.normalizers.distance_lanelet
            self.observations.past_distance_to_boundaries[:, -1] /= self.normalizers.distance_lanelet
            
            # Workaround for the fact that at the simulation begining, no history observations are available
            if (self.observations.past_pos[:, :-1]==0).all():
                # Assume agents are previously at their initial positions with zero velocities and the same priorities
                self.observations.past_pri[:, :-1] = self.observations.past_pri[:, -1].unsqueeze(1)
                self.observations.past_pos[:, :-1] = self.observations.past_pos[:, -1].unsqueeze(1)
                self.observations.past_rot[:, :-1] = self.observations.past_rot[:, -1].unsqueeze(1)

                self.observations.past_corners[:, :-1] = self.observations.past_corners[:, -1].unsqueeze(1)
                self.observations.past_short_term_ref_points[:, :-1] = self.observations.past_short_term_ref_points[:, -1].unsqueeze(1)

                self.observations.past_distance_to_agents[:, :-1] = self.observations.past_distance_to_agents[:, -1].unsqueeze(1)
                self.observations.past_distance_to_boundaries[:, :-1] = self.observations.past_distance_to_boundaries[:, -1].unsqueeze(1)
                self.observations.past_distance_to_left_boundary[:, :-1] = self.observations.past_distance_to_left_boundary[:, -1].unsqueeze(1)
                self.observations.past_distance_to_right_boundary[:, :-1] = self.observations.past_distance_to_right_boundary[:, -1].unsqueeze(1)
                self.observations.past_distance_to_ref_path[:, :-1] = self.observations.past_distance_to_ref_path[:, -1].unsqueeze(1)

        # Store new observation - actions
        self.observations.past_action_vel[:, -1, agent_index] = agent.action.u[:, 0] if (agent.action.u is not None) else self.constants.empty_actions[:, 0]
        self.observations.past_action_steering[:, -1, agent_index] = agent.action.u[:, 1] if (agent.action.u is not None) else self.constants.empty_actions[:, 1]

        # Normalize
        self.observations.past_action_steering[:, -1] /= self.normalizers.action_steering
        self.observations.past_action_vel[:, -1] /= self.normalizers.action_vel
        

        ##################################################
        ## Observation of other agents
        ##################################################
        obs_step_start = self.observations.n_stored_steps - self.observations.n_observed_steps
        
        if self.observations.is_partial:
            # Each agent observes only a fixed number of nearest agents
            nearing_agents_distances, nearing_agents_indices = torch.topk(self.distances.agents[:, agent_index], k=self.observations.n_nearing_agents, largest=False)

            # Nearing agents that are faraway will be masked
            mask_nearing_agents_too_far = (nearing_agents_distances >= self.thresholds.distance_mask_agents)
            
            step_slice = slice(obs_step_start, None)  # Slice from obs_step_start to the end
            indexing_tuple_1 = (self.constants.env_idx_broadcasting, step_slice) + \
                            ((agent_index,) if not self.observations.is_global_coordinate_sys else ()) + \
                            (nearing_agents_indices,)

            indexing_tuple_2 = (self.constants.env_idx_broadcasting, step_slice) + (nearing_agents_indices,)

            # Priorities of agents
            obs_pri_other_agents = self.observations.past_pri[indexing_tuple_1].float() # [batch_size, n_nearing_agents, n_observed_steps]
            obs_pri_other_agents[mask_nearing_agents_too_far] = self.constants.mask_zero # Priority mask (set agents that are faraway to have zero priority)
            obs_pri_other_agents = obs_pri_other_agents.transpose(1, 2) # [batch_size, n_observed_steps, n_nearing_agents]
            
            # Positions of nearing agents
            obs_pos_other_agents = self.observations.past_pos[indexing_tuple_1] # [batch_size, n_nearing_agents, n_observed_steps, 2]
            obs_pos_other_agents[mask_nearing_agents_too_far] = self.constants.mask_one # Position mask
            obs_pos_other_agents = obs_pos_other_agents.transpose(1, 2) # [batch_size, n_observed_steps, n_nearing_agents, 2, 2]

            # Rotations of nearing agents
            obs_rot_other_agents = self.observations.past_rot[indexing_tuple_1] # [batch_size, n_nearing_agents, n_observed_steps]
            obs_rot_other_agents[mask_nearing_agents_too_far] = self.constants.mask_zero # Rotation mask
            obs_rot_other_agents = obs_rot_other_agents.transpose(1, 2) # [batch_size, n_observed_steps, n_nearing_agents]

            # Velocities of nearing agents
            obs_vel_other_agents = self.observations.past_vel[indexing_tuple_1] # [batch_size, n_nearing_agents, n_observed_steps]
            obs_vel_other_agents[mask_nearing_agents_too_far] = self.constants.mask_zero # Velocity mask
            obs_vel_other_agents = obs_vel_other_agents.transpose(1, 2) # [batch_size, n_observed_steps, n_nearing_agents, 2]
            
            # Reference paths of nearing agents
            obs_ref_path_other_agents = self.observations.past_short_term_ref_points[indexing_tuple_1] # [batch_size, n_nearing_agents, n_observed_steps, n_points_short_term, 2]
            obs_ref_path_other_agents[mask_nearing_agents_too_far] = self.constants.mask_one # Reference-path mask
            obs_ref_path_other_agents = obs_ref_path_other_agents.transpose(1, 2) # [batch_size, n_observed_steps, n_nearing_agents, n_points_short_term, 2]

            # Distances to nearing agents
            obs_distance_other_agents = self.observations.past_distance_to_agents[self.constants.env_idx_broadcasting, step_slice, agent_index, nearing_agents_indices] # [batch_size, n_nearing_agents, n_observed_steps]
            obs_distance_other_agents[mask_nearing_agents_too_far] = self.constants.mask_one # Distance mask
            obs_distance_other_agents = obs_distance_other_agents.transpose(1, 2) # [batch_size, n_observed_steps, n_nearing_agents]
            
            # Distances of nearing agents to boundaries
            obs_distance_boundaries_other_agents = self.observations.past_distance_to_boundaries[indexing_tuple_2]
            obs_distance_boundaries_other_agents[mask_nearing_agents_too_far] = self.constants.mask_zero # Distance-to-bundary mask
            obs_distance_boundaries_other_agents = obs_distance_boundaries_other_agents.transpose(1, 2) # [batch_size, n_observed_steps, n_nearing_agents]
            
            # Distances of nearing agents to left boundaries
            obs_distance_left_boundary_other_agents = self.observations.past_distance_to_left_boundary[indexing_tuple_2]
            obs_distance_left_boundary_other_agents[mask_nearing_agents_too_far] = self.constants.mask_zero # Distance-to-bundary mask
            obs_distance_left_boundary_other_agents = obs_distance_left_boundary_other_agents.transpose(1, 2) # [batch_size, n_observed_steps, n_nearing_agents]
            
            # Distances of nearing agents to right boundaries
            obs_distance_right_boundary_other_agents = self.observations.past_distance_to_right_boundary[indexing_tuple_2]
            obs_distance_right_boundary_other_agents[mask_nearing_agents_too_far] = self.constants.mask_zero # Distance-to-bundary mask
            obs_distance_right_boundary_other_agents = obs_distance_right_boundary_other_agents.transpose(1, 2) # [batch_size, n_observed_steps, n_nearing_agents]

        else:
            obs_pri_other_agents = self.observations.past_pri[:, obs_step_start:] # [batch_size, n_observed_steps, n_agents]
            obs_pos_other_agents = self.observations.past_pos[:, obs_step_start:] # [batch_size, n_observed_steps, n_agents, 2]
            obs_rot_other_agents = self.observations.past_rot[:, obs_step_start:] # [batch_size, n_observed_steps, n_agents]
            obs_vel_other_agents = self.observations.past_vel[:, obs_step_start:] # [batch_size, n_observed_steps, n_agents, 2]
            obs_ref_path_other_agents = self.observations.past_short_term_ref_points[:, obs_step_start:] # [batch_size, n_observed_steps, n_agents, n_points_short_term * 2, 2]
            obs_distance_other_agents = self.observations.past_distance_to_agents[:, obs_step_start:] # [batch_size, n_observed_steps, n_agents, 2]
            obs_distance_boundaries_other_agents = self.observations.past_distance_to_boundaries[:, obs_step_start:] # [batch_size, n_observed_steps, n_agents]
            obs_distance_left_boundary_other_agents = self.observations.past_distance_to_left_boundary[:, obs_step_start:] # [batch_size, n_observed_steps, n_agents]
            obs_distance_right_boundary_other_agents = self.observations.past_distance_to_right_boundary[:, obs_step_start:] # [batch_size, n_observed_steps, n_agents]

            
        # Flatten the last dimensions to combine all features into a single dimension
        obs_pri_other_agents_flat = obs_pri_other_agents.reshape(self.world.batch_dim, self.observations.n_observed_steps, self.observations.n_nearing_agents, -1)
        obs_pos_other_agents_flat = obs_pos_other_agents.reshape(self.world.batch_dim, self.observations.n_observed_steps, self.observations.n_nearing_agents, -1)
        obs_rot_other_agents_flat = obs_rot_other_agents.reshape(self.world.batch_dim, self.observations.n_observed_steps, self.observations.n_nearing_agents, -1)
        obs_vel_other_agents_flat = obs_vel_other_agents.reshape(self.world.batch_dim, self.observations.n_observed_steps, self.observations.n_nearing_agents, -1)
        obs_ref_path_other_agents_flat = obs_ref_path_other_agents.reshape(self.world.batch_dim, self.observations.n_observed_steps, self.observations.n_nearing_agents, -1)
        obs_distance_other_agents_flat = obs_distance_other_agents.reshape(self.world.batch_dim, self.observations.n_observed_steps, self.observations.n_nearing_agents, -1)
        obs_distance_left_boundary_other_agents_flat = obs_distance_left_boundary_other_agents.reshape(self.world.batch_dim, self.observations.n_observed_steps, self.observations.n_nearing_agents, -1)
        obs_distance_right_boundary_other_agents_flat = obs_distance_right_boundary_other_agents.reshape(self.world.batch_dim, self.observations.n_observed_steps, self.observations.n_nearing_agents, -1)
        
        # Concatenate along the last dimension to combine all observations
        obs_other_agents = torch.cat([
            # obs_pri_other_agents_flat,
            obs_pos_other_agents_flat,                # [others] positions
            obs_rot_other_agents_flat,                      # [others] rotations
            obs_vel_other_agents_flat,                      # [others] velocities
            # obs_ref_path_other_agents_flat,               # [others] short-term reference paths
            obs_distance_other_agents_flat,           # [others] mutual distances
            # obs_distance_left_boundary_other_agents_flat,   # [others] min distances to left boundaries
            # obs_distance_right_boundary_other_agents_flat,  # [others] min distances to right boundaries
        ], dim=-1)
        
        

        # Ensure the final tensor is reshaped as required: [batch_size, n_observed_steps * n_nearing_agents, -1]
        # This combines all steps across all nearing agents into single entries per agent, if needed.
        obs_other_agents = obs_other_agents.reshape(obs_other_agents.shape[0], -1, obs_other_agents.shape[-1])
        
        

        indexing_tuple_3 = (self.constants.env_idx_broadcasting, step_slice) + \
                            (agent_index,) + \
                            ((agent_index,) if not self.observations.is_global_coordinate_sys else ())
        
        indexing_tuple_vel = (self.constants.env_idx_broadcasting, step_slice) + \
                            (agent_index,) + \
                            ((agent_index, 0) if not self.observations.is_global_coordinate_sys else ()) # In local coordinate system, only the first component is interesting, as the second is always 0
                            
        obs_list = [
            self.observations.past_pri[indexing_tuple_3].reshape(self.world.batch_dim, -1) if self.observations.is_global_coordinate_sys else None,                  # [own] priority,
            self.observations.past_pos[indexing_tuple_3].reshape(self.world.batch_dim, -1) if self.observations.is_global_coordinate_sys else None,                  # [own] position,
            self.observations.past_rot[indexing_tuple_3].reshape(self.world.batch_dim, -1) if self.observations.is_global_coordinate_sys else None,                  # [own] rotation,
            self.observations.past_vel[indexing_tuple_vel].reshape(self.world.batch_dim, -1),                  # [own] velocity
            self.observations.past_short_term_ref_points[indexing_tuple_3].reshape(self.world.batch_dim, -1),       # [own] short-term reference path
            self.observations.past_distance_to_ref_path[:, obs_step_start:, agent_index].reshape(self.world.batch_dim, -1),           # [own] distances to reference paths
            # self.observations.past_distance_to_boundaries[:, obs_step_start:, agent_index].reshape(self.world.batch_dim, -1),       # [own] min distances to boundaries 
            self.observations.past_distance_to_left_boundary[:, obs_step_start:, agent_index].reshape(self.world.batch_dim, -1),      # [own] min distances to left boundaries 
            self.observations.past_distance_to_right_boundary[:, obs_step_start:, agent_index].reshape(self.world.batch_dim, -1),     # [own] min distances to right boundaries

            # self.observations.past_distance_to_agents[:, obs_step_start:, agent_index].reshape(self.world.batch_dim, -1),                   # [own] distance to other agents
            # Reorder the obervations of nearing agents
            obs_other_agents.reshape(self.world.batch_dim, -1),                                                                             # [others]                             
        ]
        
        obs = torch.hstack(obs_list if self.observations.is_global_coordinate_sys else obs_list[3:])
        
        assert not obs.isnan().any(), "Observations contain nan."
        assert not obs.isinf().any(), "Observations contain inf."
        assert not (obs.abs() > 2).any(), "Observations contain values greater than 1."
        
        return obs + (0.02 * torch.rand_like(obs, device=self.world.device, dtype=torch.float32)) # Add noise to avoid overfitting
    
    def done(self):
        # print("[DEBUG] done()")
        is_collision_with_agents_occur = self.collisions.with_agents.view(self.world.batch_dim,-1).any(dim=-1) # [batch_dim]
        is_collision_with_lanelets_occur = self.collisions.with_lanelets.view(self.world.batch_dim,-1).any(dim=-1)
        is_leaving_entry_segment = self.collisions.with_entry_segments.any(dim=-1) & (self.timer.step >= 20)
        is_any_agents_leaving_exit_segment = self.collisions.with_exit_segments.any(dim=-1)
        is_max_steps_reached = self.timer.step == (self.parameters.max_steps - 1)
        
        if self.parameters.training_strategy == "3": # Record into the initial state buffer
            if torch.rand(1) > (1 - self.initial_state_buffer.probability_record): # Only a certain probability to record
                for env_collide in torch.where(is_collision_with_agents_occur)[0]:
                    self.initial_state_buffer.add(self.state_buffer.get_latest(n=n_steps_stored)[env_collide])
                    print(f"\033[94mRecord states with path ids: {self.ref_paths_agent_related.path_id[env_collide]}'.\033[0m") # Print in red
                    
                    print(f"Valid size: {self.initial_state_buffer.valid_size}")
        
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
            if self.parameters.training_strategy == "4":
                # is_done = is_max_steps_reached
                is_done = is_max_steps_reached | is_collision_with_agents_occur | is_collision_with_lanelets_occur
                
                # Reset single agnet
                agents_reset = (
                    # self.collisions.with_agents.any(dim=-1) |
                    # self.collisions.with_lanelets |
                    self.collisions.with_entry_segments |
                    self.collisions.with_exit_segments
                )
                agents_reset_indices = torch.where(agents_reset)
                for env_idx, agent_idx in zip(agents_reset_indices[0], agents_reset_indices[1]):
                    if not is_done[env_idx]:
                        self.reset_world_at(env_index=env_idx, agent_index=agent_idx)
                        # print(f"Reset agent {agent_idx} in env {env_idx}")
                
            else:
                is_done = is_max_steps_reached | is_collision_with_agents_occur | is_collision_with_lanelets_occur | is_leaving_entry_segment | is_any_agents_leaving_exit_segment

            assert not (is_collision_with_agents_occur & (self.timer.step == 0)).any()
            assert not (is_collision_with_lanelets_occur & (self.timer.step == 0)).any()
            assert not (is_leaving_entry_segment & (self.timer.step == 0)).any()
            assert not (is_max_steps_reached & (self.timer.step == 0)).any()
            assert not (is_any_agents_leaving_exit_segment & (self.timer.step == 0)).any()
            
        # Logs
        # if is_collision_with_agents_occur.any():
        #     print("Collide with other agents.")
        # if is_collision_with_lanelets_occur.any():
        #     print("Collide with lanelet.")
        # if is_leaving_entry_segment.any():
        #     print("At least one agent is leaving its entry segment.")
        if is_max_steps_reached.any():
            print("The number of the maximum steps is reached.")
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

        info = {
            "pri": self.prioritization.values[:, agent_index].reshape(self.world.batch_dim, -1) / self.normalizers.priority,
            "pos": agent.state.pos / self.normalizers.pos_world,
            "rot": angle_eliminate_two_pi(agent.state.rot) / self.normalizers.rot,
            "vel": agent.state.vel / self.normalizers.v,
            "act_vel": (agent.action.u[:, 0] / self.normalizers.action_vel) if not is_action_empty else self.constants.empty_actions[:, 0],
            "act_steer": (agent.action.u[:, 1] / self.normalizers.action_steering) if not is_action_empty else self.constants.empty_actions[:, 1],
            "ref": (self.ref_paths_agent_related.short_term[:, agent_index] / self.normalizers.pos_world).reshape(self.world.batch_dim, -1),
            "distance_ref": self.distances.ref_paths[:, agent_index] / self.normalizers.distance_ref,
            "distance_left_b": self.distances.left_boundaries[:, agent_index].min(dim=-1)[0] / self.normalizers.distance_lanelet,
            "distance_right_b": self.distances.right_boundaries[:, agent_index].min(dim=-1)[0] / self.normalizers.distance_lanelet,
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
        for i in range(len(self.map_data["lanelets"])):
            lanelet = self.map_data["lanelets"][i]
            
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
            if self.parameters.is_visualize_short_term_path & (agent_i == 0):
                geom = rendering.PolyLine(
                    v = self.ref_paths_agent_related.short_term[env_index, agent_i],
                    close=False,
                )
                xform = rendering.Transform()
                geom.add_attr(xform)            
                geom.set_color(*colors[agent_i])
                geoms.append(geom)
                
                render_points_on_short_term_ref_path = True
                if render_points_on_short_term_ref_path:
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
            
            # Visualize the agent IDs
            geom = rendering.TextLine(
                text=f"{agent_i}",
                x=(self.world.agents[agent_i].state.pos[env_index, 0] / (self.world.x_semidim)) * self.viewer_size[0],
                y=self.world.agents[agent_i].state.pos[env_index, 1] / (self.world.y_semidim) * self.viewer_size[1],
                font_size=14,
            )
            xform = rendering.Transform()
            geom.add_attr(xform)  
            geoms.append(geom)
                
            # Visualize the lanelet boundaries of agents" reference path
            # agent_i = 0
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
