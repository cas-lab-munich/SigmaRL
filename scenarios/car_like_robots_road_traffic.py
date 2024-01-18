import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World, Line
# from vmas.simulator.dynamics.kinematic_bicycle import KinematicBicycle
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# !Important: Add project root to system path if you want to run this file directly
script_dir = os.path.dirname(__file__)  # Directory of scriptA.py
project_root = os.path.dirname(script_dir)  # Project root directory
if project_root not in sys.path:
    sys.path.append(project_root)


from utilities.helper_scenario import Distances, Normalizers, Observations, Penalties, ReferencePaths, Rewards, Thresholds, exponential_decreasing_fcn, get_distances_between_agents, get_perpendicular_distances, get_rectangle_corners, get_short_term_reference_path, interX

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

n_agents = 2        # The number of agents
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

safety_buffer_to_boundaries = agent_width / 4 # A safety buffer for distance to lanelet boundaries below which agents will receive the full penalty of getting too close to lanelet boundaries (`penalty_near_boundary`)
threshold_near_boundary_low = safety_buffer_to_boundaries # Threshold for distance to lanelet boundaries above which agents will be penalized
threshold_near_boundary_high = agent_width / 2 # Threshold for distance to lanelet boundaries beneath which agents will be penalized

threshold_near_other_agents_c2c_low = agent_width / 2 # Threshold for center-to-center distance above which agents will be penalized (if the c2c distance is less than the half of the agent width, they are colliding, which will be penalized by another repalty)
threshold_near_other_agents_c2c_high = agent_length # Threshold for center-to-center distance beneath which agents will be penalized

safety_buffer_between_agents = agent_width / 2 # A safety buffer for MTV-based distances below which agents will receive the full penalty of too close to other agents (`penalty_near_other_agents`)
threshold_near_other_agents_MTV_low = safety_buffer_between_agents # Threshold for MTV-based distance above which agents will be penalized
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

# Reference path
n_short_term_points = 6 # The number of points on the short-term reference path

# Observation
is_local_observation = False # Set to True if each agent can only observe a subset of other agents, i.e., limitations on sensor range are considered. Note that this also reduces the observation size, which may facilitate training
n_nearing_agents_observed = 3 # The number of most nearing agents to be observed by each agent. This parameter will be used if `is_local_observation = True`.
is_global_coordinate_sys = True # Set to True if you want to use global coordinate system

is_testing = True # In testing mode, collisions do not lead to the termination of the simulation 
is_visualize_short_term_path = True
        
class Scenario(BaseScenario):
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
        
        # Read potential parameter
        if hasattr(self, "parameters"):
            is_local_observation = self.parameters.is_local_observation
            n_nearing_agents_observed = self.parameters.n_nearing_agents_observed
            is_global_coordinate_sys = self.parameters.is_global_coordinate_sys
            self.is_testing = self.parameters.is_testing
            self.is_visualize_short_term_path = self.parameters.is_visualize_short_term_path
        else:
            self.is_testing = is_testing
            self.is_visualize_short_term_path = is_visualize_short_term_path

        # Make world
        world = World(batch_dim, device, x_semidim=world_x_dim, y_semidim=world_y_dim, dt=dt)
        
        # Get map data 
        self.map_data = get_map_data(device=device)
        
        # Define reference paths
        self.ref_paths = ReferencePaths(
            long_term=get_reference_paths(self.n_agents, self.map_data), # Long-term reference path
            n_short_term_points=torch.tensor(n_short_term_points, device=device, dtype=torch.int),
            short_term=torch.zeros((batch_dim, self.n_agents, n_short_term_points, 2), device=device, dtype=torch.float32), # Short-term reference path
            short_term_indices = torch.zeros((batch_dim, self.n_agents, n_short_term_points), device=device, dtype=torch.int),
            left_boundary_repeated=None,
            right_boundary_repeated=None,
        )
        self.ref_paths.left_boundary_repeated = [self.ref_paths.long_term[a_i]["left_boundary_shared"].repeat(batch_dim, 1, 1) for a_i in range(self.n_agents)] # Create a variable to store the repeated data, because the function `interX` cannot handle broadcasting
        self.ref_paths.right_boundary_repeated = [self.ref_paths.long_term[a_i]["right_boundary_shared"].repeat(batch_dim, 1, 1) for a_i in range(self.n_agents)]
        
        self.corners_gloabl = torch.zeros((batch_dim, self.n_agents, 5, 2), device=device, dtype=torch.float32) # The shape of each car-like robots is considered a rectangle with 4 corners. The first corner is repeated after the last corner to close the shape. We use the corner data in the global coordinate system only during training. One training is done, we use local corner data
        self.corners_local = torch.zeros((batch_dim, self.n_agents, 5, 2), device=device, dtype=torch.float32)
        
        self.normalizers = Normalizers(
            pos=torch.tensor([world_x_dim, world_y_dim], device=device, dtype=torch.float32),
            v=2 * max_speed,
            rot=torch.tensor(2 * torch.pi, device=device, dtype=torch.float32)
        )
        
        weighting_ref_directions = torch.linspace(1, 0.2, steps=self.ref_paths.n_short_term_points-1, device=device, dtype=torch.float32)
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
            is_local=torch.tensor(is_local_observation, device=device, dtype=torch.bool),
            is_global_coordinate_sys=torch.tensor(is_global_coordinate_sys, device=device, dtype=torch.bool),
            n_nearing_agents=torch.tensor(n_nearing_agents_observed, device=device, dtype=torch.int),
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

        # Store the index of the nearest point on the reference path, which indicates the moving progress of the agent
        self.progress = torch.zeros((batch_dim, self.n_agents), device=device)
        self.progress_previous = torch.zeros((batch_dim, self.n_agents), device=device)
        
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
        
        # TODO Reset the reference path according to the training progress
        # self.ref_paths.long_term = get_reference_paths(self.n_agents, self.map_data) # A new random long-term reference path

        # Reset variables for each agent
        for i in range(self.n_agents):
            # Reset initial position and rotation (yaw angle)
            position_start = self.ref_paths.long_term[i]["center_line"][0,:] # Starting position
            rot_start = self.ref_paths.long_term[i]["center_line_yaw"][0]

            agents[i].set_pos(position_start, batch_index=env_index)
            agents[i].set_rot(rot_start, batch_index=env_index)
            
            if env_index is None:
                # Reset all envs
                # Reset the short-term reference path of agents in all envs
                self.ref_paths.short_term[:,i], self.ref_paths.short_term_indices[:,i] = get_short_term_reference_path( 
                    self.ref_paths.long_term[i]["center_line"], 
                    self.distances.closest_point_on_ref_path[:,i],
                    self.ref_paths.n_short_term_points, 
                    self.world.device
                )

                # Reset the corners of agents
                self.corners_gloabl[:,i] = get_rectangle_corners(
                    center=agents[i].state.pos, 
                    yaw=agents[i].state.rot, 
                    width=agents[i].shape.width, 
                    length=agents[i].shape.length,
                    is_close_shape=True
                )
                
                # Reset distances to lanelet boundaries and reference path          
                self.distances.ref_paths[:,i], self.distances.closest_point_on_ref_path[:,i] = get_perpendicular_distances(
                    point=agents[i].state.pos, 
                    boundary=self.ref_paths.long_term[i]["center_line"]
                )
                # Calculate the distance from each corner of the agent to lanelet boundaries
                for c_i in range(4):
                    self.distances.left_boundaries[:,i,c_i], _ = get_perpendicular_distances(
                        point=self.corners_gloabl[:,i,c_i,:], 
                        boundary=self.ref_paths.long_term[i]["left_boundary_shared"],
                    )
                    self.distances.right_boundaries[:,i,c_i], _ = get_perpendicular_distances(
                        point=self.corners_gloabl[:,i,c_i,:], 
                        boundary=self.ref_paths.long_term[i]["right_boundary_shared"],
                    )
                    
                # Reset the previous position
                agents[i].state.pos_previous = agents[i].state.pos.clone()

                # Store the index of the nearest point on the reference path, which indicates the moving progress of the agent
                # TODO Necessary?
                self.progress_previous[:,i] = self.progress[:,i].clone() # Store the values of previous time step 
                self.progress[:,i] = self.ref_paths.short_term_indices[:,i,0]        

            else:
                # Reset the env specified by `env_index`
                # Reset the short-term reference path of agents in env `env_index`
                self.ref_paths.short_term[env_index,i], self.ref_paths.short_term_indices[env_index,i] = get_short_term_reference_path( 
                    self.ref_paths.long_term[i]["center_line"], 
                    self.distances.closest_point_on_ref_path[env_index,i].unsqueeze(0),
                    self.ref_paths.n_short_term_points, 
                    self.world.device
                )
                # Reset the corners of agents in env `env_index`
                self.corners_gloabl[env_index,i] = get_rectangle_corners(
                    center=agents[i].state.pos[env_index,:].unsqueeze(0), 
                    yaw=agents[i].state.rot[env_index,:].unsqueeze(0), 
                    width=agents[i].shape.width, 
                    length=agents[i].shape.length,
                    is_close_shape=True
                )
                    
                # Reset distances to lanelet boundaries and reference path          
                self.distances.ref_paths[env_index,i], self.distances.closest_point_on_ref_path[env_index,i] = get_perpendicular_distances(
                    point=agents[i].state.pos[env_index,:].unsqueeze(0), 
                    boundary=self.ref_paths.long_term[i]["center_line"]
                )
                # Calculate the distance from each corner of the agent to lanelet boundaries
                for c_i in range(4):
                    self.distances.left_boundaries[env_index,i,c_i], _ = get_perpendicular_distances(
                        point=self.corners_gloabl[env_index,i,c_i,:].unsqueeze(0), 
                        boundary=self.ref_paths.long_term[i]["left_boundary_shared"],
                    )
                    self.distances.right_boundaries[env_index,i,c_i], _ = get_perpendicular_distances(
                        point=self.corners_gloabl[env_index,i,c_i,:].unsqueeze(0), 
                        boundary=self.ref_paths.long_term[i]["right_boundary_shared"],
                    )

                # Reset the previous position
                agents[i].state.pos_previous[env_index,:] = agents[i].state.pos[env_index,:].clone()
            
                # Store the index of the nearest point on the reference path, which indicates the moving progress of the agent
                # TODO Necessary?
                self.progress_previous[env_index,i] = self.progress[env_index,i].clone() # Store the values of previous time step 
                self.progress[env_index,i] = self.ref_paths.short_term_indices[env_index,i,0]
                
             
        
        # Compute mutual distances between agents 
        # TODO Add the possibility of computing the mutual distances of agents in env `env_index`
        mutual_distances = get_distances_between_agents(self=self)
        
        if env_index is None:
            # Reset collision matrices of all envs
            self.collision_with_agents = torch.zeros(self.world.batch_dim, self.n_agents, self.n_agents, device=self.world.device, dtype=torch.bool)
            self.collision_with_lanelets = torch.zeros(self.world.batch_dim, self.n_agents, device=self.world.device, dtype=torch.bool)

            # Reset mutual distances of all envs
            self.distances.agents = mutual_distances
        else:
            # Reset collision matrices of env `env_index`
            self.collision_with_agents[env_index,:,:] = False
            self.collision_with_lanelets[env_index,:] = False

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

        # If rewards are shared among agents
        if self.shared_reward:
            # TODO Support shared reward
            assert False, "Shared reward in this sceanrio is not supported yet."
            
        # Update the mutual distances between agents and the corners of each agent
        if agent_index == 0: # Avoid repeated computation
            self.distances.agents = get_distances_between_agents(self=self)
            for i in range(self.n_agents):
                self.corners_gloabl[:,i] = get_rectangle_corners(
                    center=self.world.agents[i].state.pos,
                    yaw=self.world.agents[i].state.rot,
                    width=self.world.agents[i].shape.width,
                    length=self.world.agents[i].shape.length,
                    is_close_shape=True,
                )
        
        # Calculate the distance from the center of the agent to its reference path
        self.distances.ref_paths[:,agent_index], self.distances.closest_point_on_ref_path[:,agent_index] = get_perpendicular_distances(
            point=agent.state.pos, 
            boundary=self.ref_paths.long_term[agent_index]["center_line"]
        )

        # Calculate the distances from each corner of the agent to lanelet boundaries
        for c_i in range(4):
            self.distances.left_boundaries[:,agent_index,c_i], _ = get_perpendicular_distances(
                point=self.corners_gloabl[:,agent_index,c_i,:],
                boundary=self.ref_paths.long_term[agent_index]["left_boundary_shared"],
            )
            self.distances.right_boundaries[:,agent_index,c_i], _ = get_perpendicular_distances(
                point=self.corners_gloabl[:,agent_index,c_i,:],
                boundary=self.ref_paths.long_term[agent_index]["right_boundary_shared"],
            )
        ##################################################
        ## Penalized if too close to lanelet boundaries
        ##################################################
        min_distances_to_bound, _ = torch.min(
            torch.hstack(
                (
                    self.distances.left_boundaries[:,agent_index], 
                    self.distances.right_boundaries[:,agent_index]
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
        ## Penalized if too close to other agents
        ##################################################
        mutual_distance_exp_fcn = exponential_decreasing_fcn(
            x=self.distances.agents[:,agent_index,:], 
            x0=self.thresholds.near_other_agents_low, 
            x1=self.thresholds.near_other_agents_high
        )
        mutual_distance_exp_fcn[:,agent_index] = 0 # Self-to-self distance is always 0 and should not be penalized
        self.rew += torch.sum(mutual_distance_exp_fcn, dim=1) * self.penalties.near_other_agents
        # print(f"Get Reward: {torch.sum(mutual_distance_exp_fcn, dim=1) * self.penalties.near_other_agents}. Distances = {self.distances.agents[:,agent_index,:]}. Distance_exp = {mutual_distance_exp_fcn}")

        ##################################################
        ## Penalized if deviating from reference path
        ##################################################
        self.rew += self.distances.ref_paths[:,agent_index] / self.penalties.weighting_deviate_from_ref_path * self.penalties.deviate_from_ref_path

        # Update the short-term reference path (extract from the long-term reference path)
        self.ref_paths.short_term[:,agent_index], self.ref_paths.short_term_indices[:,agent_index] = get_short_term_reference_path(
            self.ref_paths.long_term[agent_index]["center_line"], 
            self.distances.closest_point_on_ref_path[:,agent_index],
            self.ref_paths.n_short_term_points, 
            self.world.device
        )
        
        # [not used] Store the index of the nearest point on the reference path, which indicates the moving progress of the agent
        self.progress_previous[:,agent_index] = self.progress[:,agent_index].clone() # Store the values of previous time step 
        self.progress[:,agent_index] = self.ref_paths.short_term_indices[:,agent_index,0]
        
        ##################################################
        ## Reward for the actual movement
        ##################################################
        movement = (agent.state.pos - agent.state.pos_previous).unsqueeze(1) # Calculate the progress of the agent
        short_term_path_vec_normalized = self.ref_paths.long_term[agent_index]["center_line_vec_normalized"][self.ref_paths.short_term_indices[:,agent_index,0:-1]] # Narmalized vector of the short-term reference path
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
            self.collision_with_agents = torch.zeros(self.world.batch_dim, self.n_agents, self.n_agents, dtype=torch.bool) # Default to no collision
            self.collision_with_lanelets = torch.zeros(self.world.batch_dim, self.n_agents, dtype=torch.bool) # Default to no collision
            for a_i in range(self.n_agents):
                if self.distances.type == 'c2c':
                    for a_j in range(a_i+1, self.n_agents):
                        # Check for intersection using the interX function
                        collision_batch_index = interX(self.corners_gloabl[:,a_i], self.corners_gloabl[:,a_j], False)
                        self.collision_with_agents[torch.nonzero(collision_batch_index), a_i, a_j] = True
                        self.collision_with_agents[torch.nonzero(collision_batch_index), a_j, a_i] = True
                elif self.distances.type == 'MTV':
                    # If two agents collide, their MTV-based distance is zero 
                    mask = ~torch.eye(self.n_agents, dtype=torch.bool) # Create an inverted identity matrix mask
                    self.collision_with_agents = (self.distances.agents == 0) & mask # Use the mask to set all diagonal elements to False
                    
                # Check for collisions between agents and lanelet boundaries
                collision_with_left_bound_batch_index = interX(self.corners_gloabl[:,a_i], self.ref_paths.left_boundary_repeated[a_i], False) # [batch_dim]
                collision_with_right_bound_batch_index = interX(self.corners_gloabl[:,a_i], self.ref_paths.right_boundary_repeated[a_i], False) # [batch_dim]
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
            corners = self.corners_gloabl[:, :, 0:4, :] / self.normalizers.pos
            velocities = torch.stack([a.state.vel for a in self.world.agents], dim=1) / self.normalizers.v
            ##################################################
            ## Observation of short-term reference path
            ##################################################
            # Skip the first point on the short-term reference path, because, mostly, it is behind the agent. The second point is in front of the agent.
            obs_ref_point_rel_norm = (self.ref_paths.short_term[:,agent_index,1:] / self.normalizers.pos).reshape(self.world.batch_dim, -1)
        else:
            # Computer the positions of the corners of all agents relative to the current agents, relative velocity of other agents relative to the current agent, the absolute velovity of all agents, and the rotation of all agents relative to the current agent
            corners = torch.zeros((self.world.batch_dim, self.n_agents, 4, 2), device=self.world.device, dtype=torch.float32)
            velocities = torch.zeros((self.world.batch_dim, self.n_agents, 2), device=self.world.device, dtype=torch.float32)
            v_all_abs = torch.zeros((self.world.batch_dim, self.n_agents, 1), device=self.world.device, dtype=torch.float32)
            rot_all_rel_norm = torch.zeros((self.world.batch_dim, self.n_agents, 1), device=self.world.device, dtype=torch.float32)
            for agent_i in range(self.n_agents):
                pos_others_rel = self.world.agents[agent_i].state.pos - agent.state.pos
                rot_all_rel_norm[:, agent_i] = self.world.agents[agent_i].state.rot - agent.state.rot
                corners[:, agent_i] = get_rectangle_corners(
                    center=pos_others_rel, 
                    yaw=rot_all_rel_norm[:, agent_i], 
                    width=self.world.agents[agent_i].shape.width, 
                    length=self.world.agents[agent_i].shape.length, 
                    is_close_shape=False
                )
                
                v_all_abs[:, agent_i] = torch.norm(self.world.agents[agent_i].state.vel, dim=1).unsqueeze(1)
                velocities[:, agent_i] = torch.hstack(
                    (
                        v_all_abs[:, agent_i] * torch.cos(rot_all_rel_norm[:, agent_i]), 
                        v_all_abs[:, agent_i] * torch.sin(rot_all_rel_norm[:, agent_i])
                    )
                )
            # Normalize
            corners = corners / self.normalizers.pos
            velocities = velocities / self.normalizers.v
            ##################################################
            ## Observation of short-term reference path
            ##################################################
            # Normalized short-term reference path relative to the agent's current position
            # Skip the first point on the short-term reference path, because, mostly, it is behind the agent. The second point is in front of the agent.
            ref_points_rel_norm = (self.ref_paths.short_term[:,agent_index,1:] - agent.state.pos.unsqueeze(1)) / self.normalizers.pos
            ref_points_rel_abs = ref_points_rel_norm.norm(dim=2)
            ref_points_rel_rot = torch.atan2(ref_points_rel_norm[:,:,1], ref_points_rel_norm[:,:,0]) - agent.state.rot
            obs_ref_point_rel_norm = torch.stack(
                (
                    ref_points_rel_abs * torch.cos(ref_points_rel_rot), 
                    ref_points_rel_abs * torch.sin(ref_points_rel_rot)    
                ), dim=2
            ).reshape(self.world.batch_dim, -1)


        ##################################################
        ## Observations of self
        ##################################################
        self_corners_norm_local_reshaped = corners[:, agent_index].reshape(self.world.batch_dim, -1)
        obs_self = torch.hstack((
            self_corners_norm_local_reshaped,   # Positions of the four corners
            velocities[:, agent_index],     # Velocity 
        ))
        
        ##################################################
        ## Observation of lanelets
        ##################################################
        distances_lanelets = torch.hstack((
            self.distances.left_boundaries[:,agent_index],   # Range [0, 0.45]
            self.distances.right_boundaries[:,agent_index],  # Range [0, 0.45]
            self.distances.ref_paths[:,agent_index].unsqueeze(-1),     # Range [0, 0.45]
            )
        ) # Distance to lanelet boundaries and reference path
        obs_lanelet_distances_norm = distances_lanelets / self.normalizers.pos.norm()
        
        
        ##################################################
        ## Observation of other agents
        ##################################################
        if self.observations.is_local:
            # Each agent observes all other agents
            obs_other_agents_rel_norm = []        
            other_agents_indices = torch.cat((torch.arange(0, agent_index), torch.arange(agent_index+1, self.n_agents)))

            for agent_j in other_agents_indices:
                corners_rel_norm_j = corners[:, agent_j] # Relative positions of the four corners
                v_rel_norm_j = velocities[:, agent_j] # Relative velocity
                obs_other_agents_rel_norm.append(
                    torch.cat((
                        corners_rel_norm_j.reshape(self.world.batch_dim, -1), 
                        v_rel_norm_j,
                        ), dim=1
                    )
                )
        else:
            # Each agent observes only a fixed number of nearest agents
            _, nearing_agents_indices = torch.topk(self.distances.agents[:, agent_index], k=self.observations.n_nearing_agents + 1, largest=False)
            if nearing_agents_indices.shape[1] >= 1: # In case the observed number of nearing agents is 0
                nearing_agents_indices = nearing_agents_indices[:,1:] # Delete self
            
            obs_other_agents_rel_norm = []
            for nearing_agent_idx in range(self.observations.n_nearing_agents):
                agent_obs_idx = nearing_agents_indices[:,nearing_agent_idx]
                nearing_agents_corners_rel_norm = torch.stack([corners[batch_idx,a_idx] for batch_idx, a_idx in enumerate(agent_obs_idx)], dim=0) # Relative positions of the four corners
                # Relative velocity
                nearing_agents_v_rel_norm = torch.stack([velocities[batch_idx,a_idx,:] for batch_idx, a_idx in enumerate(agent_obs_idx)], dim=0)
                
                obs_other_agents_rel_norm.append(
                    torch.cat((
                        nearing_agents_corners_rel_norm.reshape(self.world.batch_dim, -1), 
                        nearing_agents_v_rel_norm, 
                        ), dim=1
                    )
                )  
        
        return torch.cat(
            [
                obs_self,                # Contains relative position and rotation to the short-term reference path, and self velocity 
                obs_lanelet_distances_norm,
                obs_ref_point_rel_norm,
                *obs_other_agents_rel_norm
            ],
            dim=-1,
        ) # [batch_dim, 3 + 3 + n_short_term_points*2 + 5*(n_agent-1)]


    def done(self):
        # print("[DEBUG] done()")
        is_collision_with_agents_occur = torch.any(self.collision_with_agents.view(self.world.batch_dim,-1), dim=-1) # [batch_dim]
        is_collision_with_lanelets_occur = torch.any(self.collision_with_lanelets.view(self.world.batch_dim,-1), dim=-1) # [batch_dim]
        is_done = is_collision_with_agents_occur | is_collision_with_lanelets_occur
        
        if self.is_testing:
            # print(f"Step: {self.step_count}")
            is_done = torch.zeros(is_done.shape, dtype=torch.bool)
            if self.step_count % 20 == 0:
                print("You are in testing mode. Collisions do not terminate the simulation.")
        
        return is_done


    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering

        geoms = []

        # Visualize lanelets (may slow the computation time because the amount of lanelets is high)
        for i in range(len(self.map_data["lanelets"])):
            lanelet = self.map_data["lanelets"][i]
            for type in ["left", "right"]:
                if type == "left":
                    center_points = lanelet["left_boundary_center_points"]
                    lengths = lanelet["left_boundary_lengths"]
                    yaws = lanelet["left_boundary_yaws"]
                elif type == "right":
                    center_points = lanelet["right_boundary_center_points"]
                    lengths = lanelet["right_boundary_lengths"]
                    yaws = lanelet["right_boundary_yaws"]
                else:
                    pass
                        
                for j in range(len(lengths)):
                    geom = Line(
                        length=lengths[j]
                    ).get_geometry()

                    xform = rendering.Transform()
                    geom.add_attr(xform)

                    # Set the positions of the centers of the line segment
                    xform.set_translation(
                        center_points[j, 0], 
                        center_points[j, 1]
                    )
                    
                    # Set orientations
                    xform.set_rotation(yaws[j])
                    
                    color = Color.BLACK.value
                    if isinstance(color, torch.Tensor) and len(color.shape) > 1:
                        color = color[env_index]
                    geom.set_color(*color)
                    geoms.append(geom)

        if self.is_visualize_short_term_path:
            for agent_i in range(self.n_agents):
                center_points, lengths, yaws = get_center_length_yaw_polyline(polyline=self.ref_paths.short_term[env_index,agent_i])
                for j in range(len(lengths)):
                    geom = Line(
                        length=lengths[j]
                    ).get_geometry()

                    xform = rendering.Transform()
                    geom.add_attr(xform)

                    # Set the positions of the centers of the line segment
                    xform.set_translation(
                        center_points[j, 0], 
                        center_points[j, 1]
                    )
                    
                    # Set orientations
                    xform.set_rotation(yaws[j])
                    
                    color = Color.BLACK.value
                    if isinstance(color, torch.Tensor) and len(color.shape) > 1:
                        color = color[env_index]
                    geom.set_color(*color)
                    geoms.append(geom)
        return geoms


if __name__ == "__main__":
    scenario = Scenario()
    render_interactively(
        scenario=scenario, control_two_agents=False, shared_reward=False,
    )
