import os
import sys
import time
# !Important: Add project root to system path if you want to run this file directly
script_dir = os.path.dirname(__file__) # Directory of the current script
project_root = os.path.dirname(script_dir) # Project root directory
if project_root not in sys.path:
    sys.path.append(project_root)
    
import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World, Line
# from vmas.simulator.dynamics.kinematic_bicycle import KinematicBicycle
from vmas.simulator.scenario import BaseScenario
from utilities.colors import Color

import numpy as np
import matplotlib.pyplot as plt

from utilities.helper_training import Parameters


from utilities.helper_scenario import Distances, Normalizers, Observations, Penalties, ReferencePathsAgentRelated, ReferencePathsMapRelated, Rewards, Thresholds, Timer, exponential_decreasing_fcn, get_distances_between_agents, get_perpendicular_distances, get_rectangle_corners, get_short_term_reference_path, interX, transform_from_global_to_local_coordinate

from utilities.kinematic_bicycle import KinematicBicycle

# Get road data
from utilities.get_cpm_lab_map import get_map_data, get_center_length_yaw_polyline
from utilities.get_reference_paths import get_reference_paths
 
# Sample time
dt = 0.1 # [s]

# Geometry
world_x_dim = 4.5               # The x-dimension of the world in [m]
world_y_dim = 4.0               # The y-dimension of the world in [m]
agent_width = 0.10              # The width of the agent in [m]
agent_length = 0.20             # The length of the agent in [m]
agent_wheelbase_front = 0.10    # Front wheelbase in [m]
agent_wheelbase_rear = agent_length - agent_wheelbase_front # Rear wheelbase in [m]
lane_width = 0.15               # The width of each lane in [m]

# Maximum control commands
agent_max_speed = 1.0           # Maximum speed in [m/s]
agent_max_steering_angle = 45   # Maximum steering angle in degree

n_agents = 10        # The number of agents
agent_mass = 0.5    # The mass of each agent in [m]

# Reward
reward_progress = 4
reward_speed = 0.5 # Should be smaller than reward_progress to discourage moving in a high speed without actually progressing
assert reward_speed < reward_progress, "Speed reward should be smaller than progress reward to discourage moving in a high speed in a wrong direction"
finish_a_loop = 100 # Reward for finish a loop # TODO Set a high goal reward
        
# Penalty for deviating from reference path
penalty_deviate_from_ref_path = -1
threshold_deviate_from_ref_path = (lane_width - agent_width) / 2 # 0.02 m, maxmimum allowed deviation such that agent still stays at its own lane 
# Penalty for being too close to lanelet boundaries
penalty_near_boundary = -2

threshold_reach_goal = agent_width / 2 # Agents are considered at their goal positions if their distances to the goal positions are less than this threshold

threshold_near_boundary_low = 0 # Threshold for distance to lanelet boundaries above which agents will be penalized
threshold_near_boundary_high = (lane_width / 2 - agent_width / 2) * 0.85 # Threshold for distance to lanelet boundaries beneath which agents will be penalized. lane_width / 2 - agent_width / 2 is the minimum distance between the agent's corners and lane boundaries if angent stay exactly on the center of the lane

threshold_near_other_agents_c2c_low = agent_width / 2 # Threshold for center-to-center distance above which agents will be penalized (if the c2c distance is less than the half of the agent width, they are colliding, which will be penalized by another repalty)
threshold_near_other_agents_c2c_high = agent_length # Threshold for center-to-center distance beneath which agents will be penalized

threshold_near_other_agents_MTV_low = 0 # Threshold for MTV-based distance above which agents will be penalized
threshold_near_other_agents_MTV_high = agent_width # Threshold for MTV-based distance beneath which agents will be penalized. Should be (agent_width_i + agent_width_j) / 4. Since we consider homogeneous agents, we can use agent_width / 2 

# Penalty for being too close to other agents
penalty_near_other_agents = -4
# Penalty for colliding with other agents or lanelet boundaries
penalty_collide_with_agents = -20
penalty_collide_with_boundaries = -20
# Penalty for losing time
penalty_time = -0

# Visualization
viewer_size = (1000, 1000) # TODO Check if we can use a fix camera view in vmas
viewer_zoom = 1
is_testing_mode = False # In testing mode, collisions do not lead to the termination of the simulation 
is_visualize_short_term_path = True

# Reference path
n_points_short_term = 6 # The number of points on the short-term reference path

# Observation
is_local_observation = True # Set to True if each agent can only observe a subset of other agents, i.e., limitations on sensor range are considered. Note that this also reduces the observation size, which may facilitate training
n_nearing_agents_observed = 3 # The number of most nearing agents to be observed by each agent. This parameter will be used if `is_local_observation = True`.
is_global_coordinate_sys = False # Set to True if you want to use global coordinate system

max_steps = 1000

is_real_time_rendering = True # Simulation will be paused at each time step for a certain duration to enable real-time rendering

# Training
training_strategy = '1' # One of {'1', '2', '3', '4'}
                        # '1': Train in a single, comprehensive scenario 
                        # '2': Train in a single, comprehensive scenario with prioritized experience replay
                        # '3': Train in a single, comprehensive scenario with milestones
                        # '4': Training in mixed scenarios

max_ref_path_points = 200 # The estimated maximum points on the reference path

class ScenarioRoadTraffic(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        print("[DEBUG] make_world() car_like_robots_road_traffic")
        # device = torch.device("mps") # For mac with m chip to use GPU acceleration (however, seems not fully supported by VMAS)
        self.shared_reward = kwargs.get("shared_reward", False)

        self.n_agents = kwargs.get("n_agents", n_agents) # Agent number
        
        width = kwargs.get("width", agent_width) # Agent width
        l_f = kwargs.get("l_f", agent_wheelbase_front) # Distance between the front axle and the center of gravity
        l_r = kwargs.get("l_r", agent_wheelbase_rear) # Distance between the rear axle and the center of gravity
        max_steering_angle = kwargs.get("max_steering_angle", torch.deg2rad(torch.tensor(agent_max_steering_angle)))
        max_speed = kwargs.get("max_speed", torch.tensor(agent_max_speed))
        
        self.viewer_size = viewer_size
        self.viewer_zoom = viewer_zoom

        # Specify parameters if not given
        if not hasattr(self, "parameters"):
            self.parameters = Parameters(
                is_local_observation=is_local_observation,
                is_testing_mode=is_testing_mode,
                is_visualize_short_term_path=is_visualize_short_term_path,
                max_steps=max_steps,
                is_global_coordinate_sys=is_global_coordinate_sys,
                training_strategy=training_strategy,
                n_nearing_agents_observed=n_nearing_agents_observed,
                is_real_time_rendering=is_real_time_rendering,
            )
            
        self.timer = Timer( # Timer for the first env
            start=time.time(),
            end=0,
            step=torch.zeros(batch_dim, device=device, dtype=torch.int),
            step_duration=torch.zeros(self.parameters.max_steps, device=device, dtype=torch.float32),
            step_begin=time.time(),
            render_begin=0,
        )
        
        if self.parameters.training_strategy == "4":
            self.n_agents = min(self.n_agents, 8) # The map size of mixed scenarios are smaller than the whole map, therefore maximum 8 agents are needed for training
            
        # Make world
        world = World(
            batch_dim, 
            device, 
            x_semidim=torch.tensor(world_x_dim, device=device, dtype=torch.float32),
            y_semidim=torch.tensor(world_y_dim, device=device, dtype=torch.float32),
            dt=dt
        )
        
        # Get map data 
        self.map_data = get_map_data(device=device)
        reference_paths_all, reference_paths_intersection, reference_paths_merge_in, reference_paths_merge_out = get_reference_paths(self.n_agents, self.map_data) # Long-term reference path

        # Determine the maximum number of points on the reference path
        if self.parameters.training_strategy in ("1", "2", "3"):
            # Train in one single, comprehensive scenario
            max_ref_path_points = max([
                ref_p["center_line"].shape[0] for ref_p in reference_paths_all
            ]) + 10 # 10 as a smaller buffer
        else:
            # Train in mixed scenarios 
            max_ref_path_points = max([
                ref_p["center_line"].shape[0] for ref_p in reference_paths_intersection + reference_paths_merge_in + reference_paths_merge_out
            ]) + 10
            
        # Get all reference paths (agent-specific reference paths should be assigned in `reset_world_at` function)
        self.ref_paths_map_related = ReferencePathsMapRelated(
            long_term_all=reference_paths_all,
            long_term_intersection=reference_paths_intersection,
            long_term_merge_in=reference_paths_merge_in,
            long_term_merge_out=reference_paths_merge_out,
            point_extended_all=torch.zeros((len(reference_paths_all), 2), device=device, dtype=torch.float32), # Not interesting, may be useful in the future
            point_extended_intersection=torch.zeros((len(reference_paths_intersection), 2), device=device, dtype=torch.float32),
            point_extended_merge_in=torch.zeros((len(reference_paths_merge_in), 2), device=device, dtype=torch.float32),
            point_extended_merge_out=torch.zeros((len(reference_paths_merge_out), 2), device=device, dtype=torch.float32),
        )
        
        # Extended the reference path by one point at the end
        for idx, i_path in enumerate(reference_paths_all):
            center_line_i = i_path["center_line"]
            self.ref_paths_map_related.point_extended_all[idx, :] = 2 * center_line_i[-1, :] - center_line_i[-2, :]
        for idx, i_path in enumerate(reference_paths_intersection):
            center_line_i = i_path["center_line"]
            self.ref_paths_map_related.point_extended_intersection[idx, :] = 2 * center_line_i[-1, :] - center_line_i[-2, :]
        for idx, i_path in enumerate(reference_paths_merge_in):
            center_line_i = i_path["center_line"]
            self.ref_paths_map_related.point_extended_merge_in[idx, :] = 2 * center_line_i[-1, :] - center_line_i[-2, :]
        for idx, i_path in enumerate(reference_paths_merge_out):
            center_line_i = i_path["center_line"]
            self.ref_paths_map_related.point_extended_merge_out[idx, :] = 2 * center_line_i[-1, :] - center_line_i[-2, :]
        
        self.ref_paths_agent_related = ReferencePathsAgentRelated(
            long_term=torch.zeros((batch_dim, self.n_agents, max_ref_path_points, 2), device=device, dtype=torch.float32), # Long-term reference paths of agents
            long_term_vec_normalized=torch.zeros((batch_dim, self.n_agents, max_ref_path_points, 2), device=device, dtype=torch.float32),
            point_extended=torch.zeros((batch_dim, self.n_agents, 2), device=device, dtype=torch.float32),
            left_boundary=torch.zeros((batch_dim, self.n_agents, max_ref_path_points, 2), device=device, dtype=torch.float32),
            right_boundary=torch.zeros((batch_dim, self.n_agents, max_ref_path_points, 2), device=device, dtype=torch.float32),
            is_loop=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.bool),
            n_points_long_term=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int),
            n_points_left_b=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int),
            n_points_right_b=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int),
            short_term=torch.zeros((batch_dim, self.n_agents, n_points_short_term, 2), device=device, dtype=torch.float32), # Short-term reference path
            short_term_indices = torch.zeros((batch_dim, self.n_agents, n_points_short_term), device=device, dtype=torch.int),
            n_points_short_term=torch.tensor(n_points_short_term, device=device, dtype=torch.int),
        )
        
        self.corners = torch.zeros((batch_dim, self.n_agents, 5, 2), device=device, dtype=torch.float32) # The shape of each car-like robots is considered a rectangle with 4 corners. The first corner is repeated after the last corner to close the shape. We use the corner data in the global coordinate system only during training. One training is done, we use local corner data
        self.corners_local = torch.zeros((batch_dim, self.n_agents, 5, 2), device=device, dtype=torch.float32)
        
        weighting_ref_directions = torch.linspace(1, 0.2, steps=self.ref_paths_agent_related.n_points_short_term-1, device=device, dtype=torch.float32)
        weighting_ref_directions /= weighting_ref_directions.sum()
        self.rewards = Rewards(
            progress=torch.tensor(reward_progress, device=device, dtype=torch.float32),
            weighting_ref_directions=weighting_ref_directions, # Progress in the weighted directions (directions indicating by closer short-term reference points have higher weights)
            higth_v=torch.tensor(reward_speed, device=device, dtype=torch.float32),
        )

        self.penalties = Penalties(
            deviate_from_ref_path=torch.tensor(penalty_deviate_from_ref_path, device=device, dtype=torch.float32),
            weighting_deviate_from_ref_path=self.map_data["mean_lane_width"] / 2,
            near_boundary=torch.tensor(penalty_near_boundary, device=device, dtype=torch.float32),
            near_other_agents=torch.tensor(penalty_near_other_agents, device=device, dtype=torch.float32),
            collide_with_agents=torch.tensor(penalty_collide_with_agents, device=device, dtype=torch.float32),
            collide_with_boundaries=torch.tensor(penalty_collide_with_boundaries, device=device, dtype=torch.float32),
            time=torch.tensor(penalty_time, device=device, dtype=torch.float32),
        )
        
        self.observations = Observations(
            is_local=torch.tensor(self.parameters.is_local_observation, device=device, dtype=torch.bool),
            is_global_coordinate_sys=torch.tensor(self.parameters.is_global_coordinate_sys, device=device, dtype=torch.bool),
            n_nearing_agents=torch.tensor(self.parameters.n_nearing_agents_observed, device=device, dtype=torch.int),
        )

        if self.observations.is_global_coordinate_sys:
            norm_x = world_x_dim
            norm_y = world_y_dim
        else:
            norm_x = agent_length * 10
            norm_y = agent_width * 10
        
        self.normalizers = Normalizers(
            pos=torch.tensor([norm_x, norm_y], device=device, dtype=torch.float32),
            v=max_speed,
            yaw=torch.tensor(2 * torch.pi, device=device, dtype=torch.float32)
        )
        
        # Distances to boundaries and reference path, and also the closest point on the reference paths of agents
        distance_type = 'MTV' # One of {'c2c', 'MTV'}
        self.distances = Distances(
            type = distance_type, # Type of distances between agents. One of {'c2c', 'MTV'}: center-to-center distance and minimum translation vector (MTV)-based distance
            agents=torch.zeros(world.batch_dim, self.n_agents, self.n_agents, dtype=torch.float32),
            left_boundaries=torch.zeros((batch_dim, self.n_agents, 4), device=device, dtype=torch.float32), # Each car-like agent is treated as a rectangle (with four corners)
            right_boundaries=torch.zeros((batch_dim, self.n_agents, 4), device=device, dtype=torch.float32),
            ref_paths=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.float32),
            closest_point_on_ref_path=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int)
        )

        self.thresholds = Thresholds(
            reach_goal=torch.tensor(threshold_reach_goal, device=device, dtype=torch.float32),
            deviate_from_ref_path=torch.tensor(threshold_deviate_from_ref_path, device=device, dtype=torch.float32),
            near_boundary_low=torch.tensor(threshold_near_boundary_low, device=device, dtype=torch.float32),
            near_boundary_high=torch.tensor(threshold_near_boundary_high, device=device, dtype=torch.float32),
            near_other_agents_low=torch.tensor(
                threshold_near_other_agents_c2c_low if self.distances.type == 'c2c' else threshold_near_other_agents_MTV_low, 
                device=device, 
                dtype=torch.float32
            ),
            near_other_agents_high=torch.tensor(
                threshold_near_other_agents_c2c_high if self.distances.type == 'c2c' else threshold_near_other_agents_MTV_high, 
                device=device, 
                dtype=torch.float32
            )
        )
        
        if (self.observations.n_nearing_agents >= self.n_agents - 1) | (not self.observations.is_local):
            self.observations.n_nearing_agents = self.n_agents - 1 # Substract self
            self.observations.is_local = False
            print("The number of nearing agents to be observed is more than the total number of other agents. Therefore, all other agents will be observed.")
        
        for i in range(self.n_agents):
            # Use the kinematic bicycle model for each agent
            agent = Agent(
                name=f"agent_{i}",
                shape=Box(length=l_f+l_r, width=width),
                collide=False,
                render_action=True,
                u_range=[max_speed, max_steering_angle], # Control command serves as velocity command 
                u_multiplier=[1, 1],
                max_speed=max_speed,
                dynamics=KinematicBicycle(
                    world, 
                    width=width, 
                    l_f=l_f, 
                    l_r=l_r, 
                    max_steering_angle=max_steering_angle, 
                    integration="rk4" # one of "euler", "rk4"
                )
            )
            # Create a variable to store the position of the agent at the previous time step 
            agent.state.pos_previous = torch.zeros((batch_dim, 2), device=device, dtype=torch.float32)
            world.add_agent(agent)

        # Initialize collision matrix 
        self.collision_with_agents = torch.zeros(world.batch_dim, self.n_agents, self.n_agents, dtype=torch.bool) # [batch_dim, n_agents, n_agents] The indices of colliding agents
        self.collision_with_lanelets = torch.zeros(world.batch_dim, self.n_agents, dtype=torch.bool) # [batch_dim, n_agents] The indices of agents that collide with lanelet boundaries
                
        self.step_count = torch.tensor(0, device=device, dtype=torch.int)

        return world

    def reset_world_at(self, env_index: int = None):
        # print(f"[DEBUG] reset_world_at(): env_index = {env_index}")
        """
        This function resets the world at the specified env_index.

        Args:
        :param env_index: index of the environment to reset. If None a vectorized reset should be performed

        """
        agents = self.world.agents
        
        # if hasattr(self, 'training_info'):
        #     print(self.training_info["agents"]["episode_reward"].mean())

        if (env_index is None) or (env_index == 0):
            self.timer.step_duration[:] = 0
            self.timer.start = time.time()
            self.timer.step_begin = time.time()
            self.timer.end = 0
        
        # Get the center line and boundaries of the long-term reference path for each agent
        if self.parameters.training_strategy == '4':
            # Train in mixed scenarios
            # Probabilities for each category (must sum to 1)
            probabilities = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.world.device) # Customize the probabilities of training in intersection scenario, merge-in scenario, and merge-out scenario
            random_scenario_id = torch.multinomial(probabilities, 1, replacement=True).item() + 1 # 1 for intersection, 2 for merge-in, 3 for merge-out scenario
            if random_scenario_id == 1:
                # Intersection scenario
                ref_paths_scenario = self.ref_paths_map_related.long_term_intersection
                extended_points = self.ref_paths_map_related.point_extended_intersection
            elif random_scenario_id == 2:
                # Merge-in scenario
                ref_paths_scenario = self.ref_paths_map_related.long_term_merge_in
                extended_points = self.ref_paths_map_related.point_extended_merge_in
            elif random_scenario_id == 3:
                # Merge-out scenario
                ref_paths_scenario = self.ref_paths_map_related.long_term_merge_out
                extended_points = self.ref_paths_map_related.point_extended_merge_out
            else:
                raise ValueError("Scenario ID should not exceed the number of mixed scenarios.")
        else:
            ref_paths_scenario = self.ref_paths_map_related.long_term_all
            extended_points = self.ref_paths_map_related.point_extended_all

        num_ref_paths = len(ref_paths_scenario)
        
        for i_agent in range(self.n_agents):
            is_feasible_initial_position_found = False
            random_count = 0
            
            # Ramdomly generate initial states for each agent
            if self.parameters.training_strategy in ("1", "2", "4"):
                while not is_feasible_initial_position_found:
                    # if random_count >= 1:
                    #     print(random_count)
                    random_count += 1
                    
                    random_path_id = torch.randint(0, num_ref_paths, (1,)).item() # Select randomly a path
                    ref_path = ref_paths_scenario[random_path_id]
                    
                    num_points = ref_path["center_line"].shape[0]
                    random_point_id = torch.randint(0, num_points - 10, (1,)).item() # Random point on the center as initial position
                    position_start = ref_path["center_line"][random_point_id]
                    agents[i_agent].set_pos(position_start, batch_index=env_index)
                    
                    # Check the distance between the current agents and those agents that have already have initial positions
                    if i_agent == 0:
                        # The initial position of the first agent is always feasible
                        is_feasible_initial_position_found = True
                    else:
                        if env_index is None:
                            positions = torch.stack([self.world.agents[i].state.pos[0] for i in range(i_agent+1)])
                        else:
                            positions = torch.stack([self.world.agents[i].state.pos[env_index] for i in range(i_agent+1)])
                        diff_sq = (positions[-1, :] - positions) ** 2 # Calculate pairwise squared differences in positions
                        initial_mutual_distances_sq = torch.sum(diff_sq, dim=-1) # Sum the squared differences along the last dimension and take the square root
                        min_distance_sq = torch.min(initial_mutual_distances_sq[0:-1])
                        is_feasible_initial_position_found = min_distance_sq >= (agents[0].shape.length ** 2 + agents[0].shape.width ** 2)
                        
                rot_start = ref_path["center_line_yaw"][random_point_id]
                vel_start_abs = torch.rand(1, dtype=torch.float32, device=self.world.device) * agents[i_agent].max_speed # Random initial velocity
                vel_start = torch.hstack([vel_start_abs * torch.cos(rot_start), vel_start_abs * torch.sin(rot_start)])

                agents[i_agent].set_rot(rot_start, batch_index=env_index)
                agents[i_agent].set_vel(vel_start, batch_index=env_index)
            else:
                # TODO Vanilla model with milestones
                raise NotImplementedError
            
            # Lng-term reference paths for agents
            if env_index is None:
                n_points_long_term = ref_path["center_line"].shape[0]
                
                self.ref_paths_agent_related.long_term[:, i_agent, 0:n_points_long_term, :] = ref_path["center_line"]
                self.ref_paths_agent_related.long_term[:, i_agent, n_points_long_term:, :] = ref_path["center_line"][-1, :] # Instead of zero-padding, repeat the last point
                
                self.ref_paths_agent_related.long_term_vec_normalized[:, i_agent, 0:n_points_long_term-1, :] = ref_path["center_line_vec_normalized"]
                self.ref_paths_agent_related.long_term_vec_normalized[:, i_agent, n_points_long_term-1:, :] = ref_path["center_line_vec_normalized"][-1, :]
                
                self.ref_paths_agent_related.n_points_long_term[:, i_agent] = n_points_long_term
                
                n_points_left_b = ref_path["left_boundary_shared"].shape[0]
                self.ref_paths_agent_related.left_boundary[:, i_agent, 0:n_points_left_b, :] = ref_path["left_boundary_shared"]
                self.ref_paths_agent_related.left_boundary[:, i_agent, n_points_left_b:, :] = ref_path["left_boundary_shared"][-1, :]
                
                self.ref_paths_agent_related.n_points_left_b[:, i_agent] = n_points_left_b
                
                n_points_right_b = ref_path["right_boundary_shared"].shape[0]
                self.ref_paths_agent_related.right_boundary[:, i_agent, 0:n_points_right_b, :] = ref_path["right_boundary_shared"]
                self.ref_paths_agent_related.right_boundary[:, i_agent, n_points_right_b:, :] = ref_path["right_boundary_shared"][-1, :]
                
                self.ref_paths_agent_related.n_points_right_b[:, i_agent] = n_points_right_b                    

                self.ref_paths_agent_related.is_loop[:, i_agent] = ref_path["is_loop"]
                self.ref_paths_agent_related.point_extended[:, i_agent, :] = extended_points[random_path_id, :]
            else:
                n_points_long_term = ref_path["center_line"].shape[0]
                
                self.ref_paths_agent_related.long_term[env_index, i_agent, 0:n_points_long_term, :] = ref_path["center_line"]
                self.ref_paths_agent_related.long_term[env_index, i_agent, n_points_long_term:, :] = ref_path["center_line"][-1, :] # Instead of zero-padding, repeat the last point
                
                self.ref_paths_agent_related.long_term_vec_normalized[env_index, i_agent, 0:n_points_long_term-1, :] = ref_path["center_line_vec_normalized"]
                self.ref_paths_agent_related.long_term_vec_normalized[env_index, i_agent, n_points_long_term-1:, :] = ref_path["center_line_vec_normalized"][-1, :]
                self.ref_paths_agent_related.n_points_long_term[env_index, i_agent] = n_points_long_term

                n_points_left_b = ref_path["left_boundary_shared"].shape[0]
                self.ref_paths_agent_related.left_boundary[env_index, i_agent, 0:n_points_left_b, :] = ref_path["left_boundary_shared"]
                self.ref_paths_agent_related.left_boundary[env_index, i_agent, n_points_left_b:, :] = ref_path["left_boundary_shared"][-1, :]
                
                self.ref_paths_agent_related.n_points_left_b[env_index, i_agent] = n_points_left_b
                
                n_points_right_b = ref_path["right_boundary_shared"].shape[0]
                self.ref_paths_agent_related.right_boundary[env_index, i_agent, 0:n_points_right_b, :] = ref_path["right_boundary_shared"]
                self.ref_paths_agent_related.right_boundary[env_index, i_agent, n_points_right_b:, :] = ref_path["right_boundary_shared"][-1, :]
                
                self.ref_paths_agent_related.n_points_right_b[env_index, i_agent] = n_points_right_b
                
                self.ref_paths_agent_related.is_loop[env_index, i_agent] = ref_path["is_loop"]
                self.ref_paths_agent_related.point_extended[env_index, i_agent, :] = extended_points[random_path_id, :]

        # Reset variables for each agent
        for i in range(self.n_agents):            
            if env_index is None: # Reset all envs
                # Reset distances to lanelet boundaries and reference path          
                self.distances.ref_paths[:, i], self.distances.closest_point_on_ref_path[:, i] = get_perpendicular_distances(
                    point=agents[i].state.pos, 
                    polyline=self.ref_paths_agent_related.long_term[:, i],
                    n_points_long_term=self.ref_paths_agent_related.n_points_long_term[:, i],
                )
                
                # Reset the corners of agents
                self.corners[:, i] = get_rectangle_corners(
                    center=agents[i].state.pos, 
                    yaw=agents[i].state.rot, 
                    width=agents[i].shape.width, 
                    length=agents[i].shape.length,
                    is_close_shape=True
                )
                
                # Calculate the distance from each corner of the agent to lanelet boundaries
                for c_i in range(4):
                    self.distances.left_boundaries[:, i, c_i], _ = get_perpendicular_distances(
                        point=self.corners[:, i, c_i, :], 
                        polyline=self.ref_paths_agent_related.left_boundary[:, i],
                        n_points_long_term=self.ref_paths_agent_related.n_points_left_b[:, i],
                    )
                    self.distances.right_boundaries[:, i, c_i], _ = get_perpendicular_distances(
                        point=self.corners[:, i, c_i, :], 
                        polyline=self.ref_paths_agent_related.right_boundary[:, i],
                        n_points_long_term=self.ref_paths_agent_related.n_points_right_b[:, i],
                    )
                    
                # Reset the short-term reference path of agents in all envs
                self.ref_paths_agent_related.short_term[:, i], self.ref_paths_agent_related.short_term_indices[:, i] = get_short_term_reference_path(
                    reference_path=self.ref_paths_agent_related.long_term[:, i], 
                    closest_point_on_ref_path=self.distances.closest_point_on_ref_path[:, i].unsqueeze(1),
                    n_points_short_term=self.ref_paths_agent_related.n_points_short_term, 
                    device=self.world.device,
                    is_ref_path_loop=self.ref_paths_agent_related.is_loop[:, i],
                    point_extended=self.ref_paths_agent_related.point_extended[:, i],
                    n_points_long_term=self.ref_paths_agent_related.n_points_long_term[:, i],
                )
                 
                # Reset the previous position
                agents[i].state.pos_previous = agents[i].state.pos.clone()
                
                self.timer.step[:] = 0

            else: # Reset the env specified by `env_index`
                # Reset distances to lanelet boundaries and reference path          
                self.distances.ref_paths[env_index,i], self.distances.closest_point_on_ref_path[env_index,i] = get_perpendicular_distances(
                    point=agents[i].state.pos[env_index, :].unsqueeze(0), 
                    polyline=self.ref_paths_agent_related.long_term[env_index, i],
                    n_points_long_term=self.ref_paths_agent_related.n_points_long_term[env_index, i],
                )

                # Reset the corners of agents in env `env_index`
                self.corners[env_index,i] = get_rectangle_corners(
                    center=agents[i].state.pos[env_index, :].unsqueeze(0), 
                    yaw=agents[i].state.rot[env_index, :].unsqueeze(0), 
                    width=agents[i].shape.width, 
                    length=agents[i].shape.length,
                    is_close_shape=True
                )
                
                # Calculate the distance from each corner of the agent to lanelet boundaries
                for c_i in range(4):
                    self.distances.left_boundaries[env_index,i,c_i], _ = get_perpendicular_distances(
                        point=self.corners[env_index,i,c_i, :].unsqueeze(0), 
                        polyline=self.ref_paths_agent_related.left_boundary[env_index, i],
                        n_points_long_term=self.ref_paths_agent_related.n_points_left_b[env_index, i],
                    )
                    self.distances.right_boundaries[env_index,i,c_i], _ = get_perpendicular_distances(
                        point=self.corners[env_index,i,c_i, :].unsqueeze(0), 
                        polyline=self.ref_paths_agent_related.right_boundary[env_index, i],
                        n_points_long_term=self.ref_paths_agent_related.n_points_right_b[env_index, i],
                    )
                    
                # Reset the short-term reference path of agents in env `env_index`
                self.ref_paths_agent_related.short_term[env_index,i], self.ref_paths_agent_related.short_term_indices[env_index,i] = get_short_term_reference_path(
                    reference_path=self.ref_paths_agent_related.long_term[env_index, i].unsqueeze(0),
                    closest_point_on_ref_path=self.distances.closest_point_on_ref_path[env_index, i].unsqueeze(0).unsqueeze(-1),
                    n_points_short_term=self.ref_paths_agent_related.n_points_short_term, 
                    device=self.world.device,
                    is_ref_path_loop=self.ref_paths_agent_related.is_loop[env_index, i].unsqueeze(0),
                    n_points_long_term=self.ref_paths_agent_related.n_points_long_term[env_index, i].unsqueeze(0),
                    point_extended=self.ref_paths_agent_related.point_extended[env_index, i].unsqueeze(0),
                )


                # Reset the previous position
                agents[i].state.pos_previous[env_index, :] = agents[i].state.pos[env_index, :].clone()
                            
                self.timer.step[env_index] = 0

        # Compute mutual distances between agents 
        # TODO Add the possibility of computing the mutual distances of agents in env `env_index`
        mutual_distances = get_distances_between_agents(self=self, distance_type=self.distances.type)
        
        if env_index is None:
            # Reset collision matrices of all envs
            self.collision_with_agents = torch.zeros((self.world.batch_dim, self.n_agents, self.n_agents), device=self.world.device, dtype=torch.bool)
            self.collision_with_lanelets = torch.zeros((self.world.batch_dim, self.n_agents), device=self.world.device, dtype=torch.bool)

            # Reset mutual distances of all envs
            self.distances.agents = mutual_distances
        else:
            # Reset collision matrices of env `env_index`
            self.collision_with_agents[env_index, :, :] = False
            self.collision_with_lanelets[env_index, :] = False

            # Reset mutual distances of all envs
            self.distances.agents[env_index] = mutual_distances[env_index]
        
        # Reset step count
        self.step_count.fill_(0)

    def process_action(self, agent: Agent):
        # print("[DEBUG] process_action()")
        if hasattr(agent, 'dynamics') and hasattr(agent.dynamics, 'process_force'):
            agent.dynamics.process_force()
        else:
            # The agent does not have a dynamics property, or it does not have a process_force method
            pass

    def reward(self, agent: Agent):
        # print("[DEBUG] reward()")
        # Initialize
        self.rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)
        agent_index = self.world.agents.index(agent) # Get the index of the current agent

        # Timer
        if agent_index == 0:
            self.timer.step_duration[self.timer.step] = time.time() - self.timer.step_begin                
            self.timer.step_begin = time.time() # Set to the current time as the begin of the current time step
            # Increment step by 1
            self.timer.step += 1
            
        # If rewards are shared among agents
        if self.shared_reward:
            # TODO Support shared reward
            assert False, "Shared reward in this sceanrio is not supported yet."
            
        # Update the mutual distances between agents and the corners of each agent
        if agent_index == 0: # Avoid repeated computation
            self.distances.agents = get_distances_between_agents(self=self, distance_type=self.distances.type)
            for i in range(self.n_agents):
                self.corners[:, i] = get_rectangle_corners(
                    center=self.world.agents[i].state.pos,
                    yaw=self.world.agents[i].state.rot,
                    width=self.world.agents[i].shape.width,
                    length=self.world.agents[i].shape.length,
                    is_close_shape=True,
                )
        
        # Calculate the distance from the center of the agent to its reference path
        self.distances.ref_paths[:, agent_index], self.distances.closest_point_on_ref_path[:, agent_index] = get_perpendicular_distances(
            point=agent.state.pos, 
            polyline=self.ref_paths_agent_related.long_term[:, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_long_term[:, agent_index],
        )

        # Calculate the distances from each corner of the agent to lanelet boundaries
        for c_i in range(4):
            self.distances.left_boundaries[:, agent_index,c_i], _ = get_perpendicular_distances(
                point=self.corners[:, agent_index,c_i, :],
                polyline=self.ref_paths_agent_related.left_boundary[:, agent_index],
                n_points_long_term=self.ref_paths_agent_related.n_points_left_b[:, agent_index],
            )
            self.distances.right_boundaries[:, agent_index,c_i], _ = get_perpendicular_distances(
                point=self.corners[:, agent_index,c_i, :],
                polyline=self.ref_paths_agent_related.right_boundary[:, agent_index],
                n_points_long_term=self.ref_paths_agent_related.n_points_right_b[:, agent_index],
            )
        ##################################################
        ## Penalty for being too close to lanelet boundaries
        ##################################################
        min_distances_to_bound, _ = torch.min(
            torch.hstack(
                (
                    self.distances.left_boundaries[:, agent_index], 
                    self.distances.right_boundaries[:, agent_index]
                )
            ),
            dim=1
        ) # Pick the smallest distance among all distances from the four corners to the left and the right boundaries
        self.rew += exponential_decreasing_fcn(
            x=min_distances_to_bound, 
            x0=self.thresholds.near_boundary_low, 
            x1=self.thresholds.near_boundary_high,
        ) * self.penalties.near_boundary

        ##################################################
        ## Penalty for being too close to other agents
        ##################################################
        mutual_distance_exp_fcn = exponential_decreasing_fcn(
            x=self.distances.agents[:, agent_index, :], 
            x0=self.thresholds.near_other_agents_low, 
            x1=self.thresholds.near_other_agents_high
        )
        mutual_distance_exp_fcn[:, agent_index] = 0 # Self-to-self distance is always 0 and should not be penalized
        self.rew += torch.sum(mutual_distance_exp_fcn, dim=1) * self.penalties.near_other_agents
        # print(f"Get Reward: {torch.sum(mutual_distance_exp_fcn, dim=1) * self.penalties.near_other_agents}. Distances = {self.distances.agents[:, agent_index, :]}. Distance_exp = {mutual_distance_exp_fcn}")

        ##################################################
        ## Penalty for deviating from reference path
        ##################################################
        self.rew += self.distances.ref_paths[:, agent_index] / self.penalties.weighting_deviate_from_ref_path * self.penalties.deviate_from_ref_path

        # Update the short-term reference path (extract from the long-term reference path)
        self.ref_paths_agent_related.short_term[:, agent_index], self.ref_paths_agent_related.short_term_indices[:, agent_index] = get_short_term_reference_path(
            reference_path=self.ref_paths_agent_related.long_term[:, agent_index], 
            closest_point_on_ref_path=self.distances.closest_point_on_ref_path[:, agent_index].unsqueeze(1),
            n_points_short_term=self.ref_paths_agent_related.n_points_short_term, 
            device=self.world.device,
            is_ref_path_loop=self.ref_paths_agent_related.is_loop[:, agent_index],
            point_extended=self.ref_paths_agent_related.point_extended[:, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_long_term[:, agent_index],
        )
        
        ##################################################
        ## Reward for the actual movement
        ##################################################
        movement = (agent.state.pos - agent.state.pos_previous).unsqueeze(1) # Calculate the progress of the agent
        short_term_path_vec_normalized = self.ref_paths_agent_related.long_term_vec_normalized[
            torch.arange(self.world.batch_dim, device=self.world.device, dtype=torch.int).unsqueeze(-1),
            agent_index, 
            self.ref_paths_agent_related.short_term_indices[:, agent_index, 0:-1],
            :
        ] # Narmalized vector of the short-term reference path
        movement_normalized_proj = torch.sum(movement * short_term_path_vec_normalized, dim=2)
        movement_weighted_sum_proj = torch.matmul(movement_normalized_proj, self.rewards.weighting_ref_directions)
        self.rew += movement_weighted_sum_proj / (agent.max_speed * self.world.dt) * self.rewards.progress # Relative to the maximum possible movement
        
        ##################################################
        ## Reward for moving in a high velocity and the right direction
        ##################################################
        # TODO: optional?
        v_proj = torch.sum(agent.state.vel.unsqueeze(1) * short_term_path_vec_normalized, dim=2)
        v_proj_weighted_sum = torch.matmul(v_proj, self.rewards.weighting_ref_directions)
        self.rew += v_proj_weighted_sum / agent.max_speed * self.rewards.higth_v

        # Save previous positions
        agent.state.pos_previous = agent.state.pos.clone() 
        
        # Check for collisions between each pair of agents in the environment
        if agent_index == 0: # Avoid repeated computation            
            self.collision_with_agents = torch.zeros((self.world.batch_dim, self.n_agents, self.n_agents), dtype=torch.bool) # Default to no collision
            self.collision_with_lanelets = torch.zeros((self.world.batch_dim, self.n_agents), dtype=torch.bool) # Default to no collision
            for a_i in range(self.n_agents):
                if self.distances.type == 'c2c':
                    for a_j in range(a_i+1, self.n_agents):
                        # Check for intersection using the interX function
                        collision_batch_index = interX(self.corners[:, a_i], self.corners[:, a_j], False)
                        self.collision_with_agents[torch.nonzero(collision_batch_index), a_i, a_j] = True
                        self.collision_with_agents[torch.nonzero(collision_batch_index), a_j, a_i] = True
                elif self.distances.type == 'MTV':
                    # If two agents collide, their MTV-based distance is zero 
                    mask = ~torch.eye(self.n_agents, device=self.world.device, dtype=torch.bool) # Create an inverted identity matrix mask
                    self.collision_with_agents = (self.distances.agents == 0) & mask # Use the mask to set all diagonal elements to False
                    
                # Check for collisions between agents and lanelet boundaries
                collision_with_left_bound_batch_index = interX(
                    self.corners[:, a_i], 
                    self.ref_paths_agent_related.left_boundary[:, a_i], 
                    False    
                ) # [batch_dim]
                collision_with_right_bound_batch_index = interX(
                    self.corners[:, a_i], 
                    self.ref_paths_agent_related.right_boundary[:, a_i],
                    False                                                    
                ) # [batch_dim]
                self.collision_with_lanelets[collision_with_left_bound_batch_index | collision_with_right_bound_batch_index, a_i] = True
            
            # Increment step by 1
            self.step_count += 1 

        n_colliding_agents = self.collision_with_agents[:, agent_index].sum(dim=1) # [batch_dim] Summing over to see how many agents does the current agent collides with
        is_collide_with_lanelets = self.collision_with_lanelets[:, agent_index]
        ##################################################
        ## Penalty for colliding with other agents
        ##################################################
        self.rew += n_colliding_agents * self.penalties.collide_with_agents

        ##################################################
        ## Penalty for colliding with lanelet boundaries
        ##################################################
        self.rew += is_collide_with_lanelets * self.penalties.collide_with_boundaries

        ##################################################
        ## Penalty for losing time
        ##################################################
        # TODO: check if this is necessary
        self.rew += self.penalties.time

        return self.rew


    def observation(self, agent: Agent):
        # print("[DEBUG] observation()")
        """
        Generate an observation for the given agent in all envs.

        Parameters:
        - agent (Agent): The agent for which the observation is to be generated.

        Returns:
        - The observation for the given agent in all envs.
        """
        agent_index = self.world.agents.index(agent)
        
        if self.observations.is_global_coordinate_sys:
            corners = self.corners[:, :, 0:4, :] / self.normalizers.pos
            velocities = torch.stack([a.state.vel for a in self.world.agents], dim=1) / self.normalizers.v
            ##################################################
            ## Observation of short-term reference path
            ##################################################
            obs_ref_point_rel_norm = (self.ref_paths_agent_related.short_term[:, agent_index] / self.normalizers.pos).reshape(self.world.batch_dim, -1)
        else:
            # Computer the positions of the corners of all agents relative to the current agents, relative velocity of other agents relative to the current agent, the absolute velovity of all agents, and the rotation of all agents relative to the current agent
            corners = torch.zeros((self.world.batch_dim, self.n_agents, 4, 2), device=self.world.device, dtype=torch.float32)

            velocities = torch.zeros((self.world.batch_dim, self.n_agents, 2), device=self.world.device, dtype=torch.float32)
            vel_abs = torch.zeros((self.world.batch_dim, self.n_agents, 1), device=self.world.device, dtype=torch.float32)
            rot_all_rel = torch.zeros((self.world.batch_dim, self.n_agents, 1), device=self.world.device, dtype=torch.float32)
            for agent_i in range(self.n_agents):
                corners[:, agent_i] = transform_from_global_to_local_coordinate(
                    pos_i=agent.state.pos,
                    pos_j=self.corners[:, agent_i, 0:4, :],
                    rot_i=agent.state.rot,
                )
                
                rot_all_rel[:, agent_i] = self.world.agents[agent_i].state.rot - agent.state.rot

                vel_abs[:, agent_i] = torch.norm(self.world.agents[agent_i].state.vel, dim=1).unsqueeze(1)
                velocities[:, agent_i] = torch.hstack(
                    (
                        vel_abs[:, agent_i] * torch.cos(rot_all_rel[:, agent_i]), 
                        vel_abs[:, agent_i] * torch.sin(rot_all_rel[:, agent_i])
                    )
                )
            # Normalize
            corners = corners / self.normalizers.pos
            velocities = velocities / self.normalizers.v
            ##################################################
            ## Observation of the short-term reference path
            ##################################################
            # Normalized short-term reference path relative to the agent's current position
            ref_points_rel = self.ref_paths_agent_related.short_term[:, agent_index] - agent.state.pos.unsqueeze(1)
            ref_points_rel_abs = ref_points_rel.norm(dim=2)
            ref_points_rel_rot = torch.atan2(ref_points_rel[:, :, 1], ref_points_rel[:, :, 0]) - agent.state.rot
            obs_ref_point_rel_norm = (torch.stack(
                (
                    ref_points_rel_abs * torch.cos(ref_points_rel_rot), 
                    ref_points_rel_abs * torch.sin(ref_points_rel_rot)    
                ), dim=2
            ) / self.normalizers.pos).reshape(self.world.batch_dim, -1)


        ##################################################
        ## Observations of self
        ##################################################
        self_corners_norm_reshaped = corners[:, agent_index].reshape(self.world.batch_dim, -1)
        obs_self = torch.hstack((
            self_corners_norm_reshaped,   # Positions of the four corners
            velocities[:, agent_index],   # Velocity 
        ))
        
        ##################################################
        ## Observation of lanelets
        ##################################################
        distances_lanelets = torch.hstack((
            self.distances.left_boundaries[:, agent_index],   # Range [0, 0.45]
            self.distances.right_boundaries[:, agent_index],  # Range [0, 0.45]
            self.distances.ref_paths[:, agent_index].unsqueeze(-1),     # Range [0, 0.45]
            )
        ) # Distance to lanelet boundaries and reference path
        obs_lanelet_distances_norm = distances_lanelets / self.normalizers.pos[1] # Use the width direction as the normalizer
        
        
        ##################################################
        ## Observation of other agents
        ##################################################
        if not self.observations.is_local:
            # Each agent observes all other agents
            obs_other_agents_norm = []        
            other_agents_indices = torch.cat((
                torch.arange(0, agent_index, device=self.world.device, dtype=torch.int), 
                torch.arange(agent_index+1, self.n_agents, device=self.world.device, dtype=torch.int
            )))

            for agent_j in other_agents_indices:
                corners_j = corners[:, agent_j] # Relative positions of the four corners
                v_rel_norm_j = velocities[:, agent_j] # Relative velocity
                obs_other_agents_norm.append(
                    torch.cat((
                        corners_j.reshape(self.world.batch_dim, -1), 
                        v_rel_norm_j,
                        ), dim=1
                    )
                )
        else:
            # Each agent observes only a fixed number of nearest agents
            _, nearing_agents_indices = torch.topk(self.distances.agents[:, agent_index], k=self.observations.n_nearing_agents + 1, largest=False)
            if nearing_agents_indices.shape[1] >= 1: # In case the observed number of nearing agents is 0
                nearing_agents_indices = nearing_agents_indices[:, 1:] # Delete self
            
            obs_other_agents_norm = []
            for nearing_agent_idx in range(self.observations.n_nearing_agents):
                agent_obs_idx = nearing_agents_indices[:, nearing_agent_idx]
                nearing_agents_corners_norm = torch.stack([corners[batch_idx,a_idx] for batch_idx, a_idx in enumerate(agent_obs_idx)], dim=0)
                nearing_agents_vel_norm = torch.stack([velocities[batch_idx,a_idx, :] for batch_idx, a_idx in enumerate(agent_obs_idx)], dim=0)
                
                obs_other_agents_norm.append(
                    torch.cat((
                        nearing_agents_corners_norm.reshape(self.world.batch_dim, -1), 
                        nearing_agents_vel_norm, 
                        ), dim=1
                    )
                )  
        
        return torch.cat(
            [
                obs_self,                   # 10
                obs_lanelet_distances_norm, # 9
                obs_ref_point_rel_norm,     # 12
                *obs_other_agents_norm      # 10 * self.observations.n_nearing_agents
            ],
            dim=-1,
        )


    def done(self):
        # print("[DEBUG] done()")
        is_collision_with_agents_occur = torch.any(self.collision_with_agents.view(self.world.batch_dim,-1), dim=-1) # [batch_dim]
        is_collision_with_lanelets_occur = torch.any(self.collision_with_lanelets.view(self.world.batch_dim,-1), dim=-1) # [batch_dim]
        # Reaching the maximum time steps terminates an episode
        is_max_steps_reached = self.timer.step == (self.parameters.max_steps - 1)

        
        if self.parameters.training_strategy == "4":
            # Terminate the current simulation if at least one agent is near its end point when training in mixed scenarios
            idx_near_end_point = min(self.ref_paths_agent_related.n_points_short_term, 3)
            is_any_agent_near_end_point = (self.ref_paths_agent_related.short_term_indices[:, :, -idx_near_end_point] >= (self.ref_paths_agent_related.n_points_long_term-1)).any(dim=1)
        else:
            is_any_agent_near_end_point = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.bool)
        
        is_done = is_collision_with_agents_occur | is_collision_with_lanelets_occur | is_max_steps_reached | is_any_agent_near_end_point
        
        # Logs
        # if is_collision_with_agents_occur.any():
        #     print("Collide with other agents.")
        # if is_collision_with_lanelets_occur.any():
        #     print("Collide with lanelet.")
        # if is_max_steps_reached.any():
        #     print("The number of the maximum steps is reached.")
        if is_any_agent_near_end_point.any():
            print("At least one agent is near its end point.")
            
        if self.parameters.is_testing_mode:
            is_done = torch.zeros(is_done.shape, dtype=torch.bool)
            if self.step_count % 20 == 0:
                print("You are in testing mode. Collisions do not terminate the simulation.")
        
        return is_done


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
            # Visualize goal
            if not self.ref_paths_agent_related.is_loop[env_index, agent_i]:
                color = Color.red100
                circle = rendering.make_circle(radius=self.thresholds.reach_goal, filled=True)
                xform = rendering.Transform()
                circle.add_attr(xform)
                xform.set_translation(
                    self.ref_paths_agent_related.long_term[env_index, agent_i, -1, 0], 
                    self.ref_paths_agent_related.long_term[env_index, agent_i, -1, 1]
                )
                circle.set_color(*color)
                geoms.append(circle)

            # Visualize short-term reference paths of agents
            if self.parameters.is_visualize_short_term_path:
                geom = rendering.PolyLine(
                    v = self.ref_paths_agent_related.short_term[env_index, agent_i],
                    close=False,
                )
                xform = rendering.Transform()
                geom.add_attr(xform)            
                geom.set_color(*Color.green100)
                geoms.append(geom)
                
            # # Visualize the lanelet boundaries of agents' reference path
            # geom = rendering.PolyLine(
            #     v = self.ref_paths_agent_related.left_boundary[env_index, agent_i],
            #     close=False,
            # )
            # xform = rendering.Transform()
            # geom.add_attr(xform)            
            # geom.set_color(*Color.red100)
            # geoms.append(geom)

            # geom = rendering.PolyLine(
            #     v = self.ref_paths_agent_related.right_boundary[env_index, agent_i],
            #     close=False,
            # )
            # xform = rendering.Transform()
            # geom.add_attr(xform)            
            # geom.set_color(*Color.red100)
            # geoms.append(geom)
            
        return geoms


if __name__ == "__main__":
    scenario = ScenarioRoadTraffic()
    render_interactively(
        scenario=scenario, control_two_agents=False, shared_reward=False,
    )
