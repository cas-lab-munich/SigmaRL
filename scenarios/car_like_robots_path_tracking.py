import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World, Line
# from vmas.simulator.dynamics.kinematic_bicycle import KinematicBicycle
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color

from torch import Tensor
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# !Important: Add project root to system path if you want to run this file directly
script_dir = os.path.dirname(__file__) # Directory of the current script
project_root = os.path.dirname(script_dir) # Project root directory
if project_root not in sys.path:
    sys.path.append(project_root)


from utilities.helper_scenario import Distances, Evaluation, Normalizers, Observations, Penalties, ReferencePaths, Rewards, Thresholds, exponential_decreasing_fcn, get_distances_between_agents, get_perpendicular_distances, get_rectangle_corners, get_short_term_reference_path, interX

from utilities.helper_training import Parameters

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

# Maximum control commands
agent_max_speed = 0.5          # Maximum speed in [m/s]
agent_max_steering_angle = 45   # Maximum steering angle in degree

n_agents = 1        # The number of agents
agent_mass = 0.5    # The mass of each agent in [m]

# Reward
reward_progress = 4
reward_speed = 0.5 # Should be smaller than reward_progress to discourage moving in a high speed without actually progressing
assert reward_speed < reward_progress, "Speed reward should be smaller than progress reward to discourage moving in a high speed in a wrong direction"
reward_reaching_goal = 100 # Reward for reaching goal

# Penalty for deviating from reference path
penalty_deviate_from_ref_path = -0.2
penalty_deviate_from_goal = -1

# Penalty for losing time
penalty_time = -0.2

# Visualization
viewer_size = (1000, 1000) # TODO Check if we can use a fix camera view in vmas
viewer_zoom = 1

# Reference path
n_short_term_points = 6 # The number of points on the short-term reference path

# Observation
is_global_coordinate_sys = True # Set to True if you want to use global coordinate system
is_local_observation = False # TODO Set to True if sensor range is limited

is_testing_mode = False # In testing mode, collisions do not lead to the termination of the simulation 
is_visualize_short_term_path = True

threshold_reaching_goal = agent_width / 8 # Agents are considered at their goal positions if their distances to the goal positions are less than this threshold

max_steps = 2000
is_dynamic_goal_reward = True

# Current implementation includes four path-tracking types
path_tracking_types = [
    "line",         # Tracking a straight vertical line with the initial velocity having an initial angle, e.g., 45 degrees
    "turning",      # Tracking a turning path (90 degrees)
    "circle",       # Tracking a circle with the initial position on the circle and the initial velocity being tangent to the circle
    "sine",         # Tracking a sine curve with initial position on the curve and the initial velocity being tangent to the circle
    "horizontal_8", # Horizontal "8"-path
]
path_tracking_type_default = path_tracking_types[0]
is_observe_full_path = False # TODO Compare the performances of observing the full reference path and observing a short-term reference path

def get_ref_path_for_tracking_scenarios(path_tracking_type, max_speed = agent_max_speed, device = torch.device("cpu"), is_visualize: bool = False):
    if path_tracking_type == "line":
        start_pos = torch.tensor([-1, 0], device=device, dtype=torch.float32)
        start_rot = torch.deg2rad(torch.tensor(45, device=device, dtype=torch.float32))

        path_length = 3 # [m]
        
        goal_pos = start_pos.clone()
        goal_pos[0] += path_length
        goal_rot = torch.deg2rad(torch.tensor(0, device=device, dtype=torch.float32))

        num_points = 20 # Number of points to discretize the reference path
        tracking_path = torch.stack([torch.linspace(start_pos[i], goal_pos[i], num_points, device=device, dtype=torch.float32) for i in range(2)], dim=1)

    elif path_tracking_type == "turning":
        start_pos = torch.tensor([-1, 0], device=device, dtype=torch.float32)
        start_rot = torch.deg2rad(torch.tensor(0, device=device, dtype=torch.float32))

        horizontal_length = 3  # [m] Length of the horizontal part
        vertical_length = 2  # [m] Length of the vertical part
        num_points = 100  # Total number of points to discretize the reference path

        goal_pos = tracking_path[-1, :]
        goal_rot = torch.deg2rad(torch.tensor(90, device=device, dtype=torch.float32))

        # Number of points for each segment
        num_points_horizontal = int(num_points * horizontal_length / (horizontal_length + vertical_length))
        num_points_vertical = num_points - num_points_horizontal

        # Generate horizontal segment
        x_coords_horizontal = torch.linspace(start_pos[0], start_pos[0] + horizontal_length, num_points_horizontal, device=device)
        y_coords_horizontal = torch.full((num_points_horizontal,), start_pos[1], device=device)

        # Generate vertical segment
        x_coords_vertical = torch.full((num_points_vertical,), start_pos[0] + horizontal_length, device=device)
        y_coords_vertical = torch.linspace(start_pos[1], start_pos[1] + vertical_length, num_points_vertical, device=device)

        # Combine segments
        x_coords = torch.cat((x_coords_horizontal, x_coords_vertical))
        y_coords = torch.cat((y_coords_horizontal, y_coords_vertical))
        tracking_path = torch.stack((x_coords, y_coords), dim=1)
        
    elif path_tracking_type == "circle":
        start_pos = torch.tensor([-1, 0], device=device, dtype=torch.float32)
        start_rot = torch.deg2rad(torch.tensor(90, device=device, dtype=torch.float32))
    
        circle_radius = 1.5 # [m]                
        circle_origin = start_pos.clone()
        circle_origin[0] += circle_radius
        
        goal_pos = start_pos.clone()
        goal_rot = start_rot.clone()
        
        path_length = 2 * torch.pi * circle_radius # [m]
        num_points = 100 # Number of points to discretize the reference path
        
        # Generate angles for each point on the tracking path
        angles = torch.linspace(torch.pi, -torch.pi, num_points, device=device)

        # Calculate x and y coordinates for each point 
        x_coords = circle_origin[0] + circle_radius * torch.cos(angles)
        y_coords = circle_origin[1] + circle_radius * torch.sin(angles)

        tracking_path = torch.stack((x_coords, y_coords), dim=1)

    elif path_tracking_type == "sine":
        start_pos = torch.tensor([-1, 0], device=device, dtype=torch.float32)
        start_rot = torch.deg2rad(torch.tensor(90, device=device, dtype=torch.float32))

        path_length = 3  # [m] Length along x-axis
        num_points = 100  # Number of points to discretize the reference path
        amplitude = 1.0  # Amplitude of the sine wave

        # Generate linearly spaced x coordinates
        x_coords = torch.linspace(start_pos[0], start_pos[0] + path_length, num_points, device=device)
        # Generate sine y coordinates
        y_coords = start_pos[1] + amplitude * torch.sin(2 * torch.pi * (x_coords - start_pos[0]) / path_length)

        tracking_path = torch.stack((x_coords, y_coords), dim=1)

        goal_pos = tracking_path[-1, :]
        goal_rot = torch.deg2rad(torch.tensor(90, device=device, dtype=torch.float32))


    elif path_tracking_type == "horizontal_8":
        # Use lemniscate of Bernoulli to generate a horizontal "8" path (inspired by https://mathworld.wolfram.com/Lemniscate.html)
        start_pos = torch.tensor([-1, 0], device=device, dtype=torch.float32)
        start_rot = torch.deg2rad(torch.tensor(90, device=device, dtype=torch.float32))

        center_point = start_pos.clone() # Center point of the lemniscate
        a = 1.5  # half-width of the lemniscate
        center_point[0] += a
        num_points = 100  # Number of points to discretize the reference path

        # Generate parameter t
        t = torch.linspace(-torch.pi, torch.pi, num_points, device=device)

        # Parametric equations for the lemniscate
        x_coords = start_pos[0] + (a * torch.cos(t)) / (1 + torch.sin(t)**2)
        y_coords = start_pos[1] + (a * torch.sin(t) * torch.cos(t)) / (1 + torch.sin(t)**2)

        # Combine x and y coordinates
        tracking_path = torch.stack((x_coords, y_coords), dim=1)

        goal_pos = tracking_path[-1, :]
        goal_rot = torch.deg2rad(torch.tensor(90, device=device, dtype=torch.float32))

    else:
        raise ValueError("Invalid path tracking type provided. Must be one of 'line', 'turning', 'circle', 'sine', and 'turning'.")
    
    start_vel = torch.tensor([max_speed*torch.cos(start_rot), max_speed*torch.sin(start_rot)], device=device, dtype=torch.float32) 

    # Visualization
    if is_visualize:
        plt.plot(tracking_path[:,0], tracking_path[:,1])
        plt.axis("equal")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
    
    # Mean length of the line segments on the path
    mean_length_line_segments = tracking_path.diff(dim=0).norm(dim=1).mean()
    # print(f"The mean length of the line segments of the tracking path is {mean_length_line_segments}.")
    
    return tracking_path, start_pos, start_rot, start_vel, goal_pos, goal_rot



class ScenarioPathTracking(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        
        n_agents_received = kwargs.get("n_agents", n_agents)
        if n_agents_received is not 1:
            print(f"In path-tracking scenarios, (only) one agent is needed (but received '{n_agents_received})'.")
        
        self.n_agents = 1 # Only one agent is needed in the path-tracking scenarios
            
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
                is_testing_mode=is_testing_mode,
                is_visualize_short_term_path=is_visualize_short_term_path,
                max_steps=max_steps,
                is_dynamic_goal_reward=is_dynamic_goal_reward,
                path_tracking_type='line',
            )

        # Make world
        world = World(batch_dim, device, x_semidim=world_x_dim, y_semidim=world_y_dim, dt=dt)
        # world._drag = 0 # !No drag/friction

        tracking_path, self.start_pos, self.start_rot, self.start_vel, self.goal_pos, self.goal_rot = get_ref_path_for_tracking_scenarios(path_tracking_type=self.parameters.path_tracking_type)
        
        self.is_currently_at_goal = torch.zeros(batch_dim, device=device, dtype=torch.bool)        # If agents currently are at their goal positions
        self.has_reached_goal  = torch.zeros(batch_dim, device=device, dtype=torch.bool) # Record if agents have reached their goal at least once 
        
        center_points_path, lengths_path, yaws_path, vecs_path = get_center_length_yaw_polyline(polyline=tracking_path)
        
        vecs_path_norm = vecs_path / lengths_path.unsqueeze(1)
        
        # Define reference paths
        self.ref_paths = ReferencePaths(
            long_term=tracking_path.unsqueeze(0), # Long-term reference path
            long_term_center_points=center_points_path,
            long_term_lengths=lengths_path,
            long_term_yaws=yaws_path,
            long_term_vecs_normalized=vecs_path_norm,
            n_short_term_points=torch.tensor(n_short_term_points, device=device, dtype=torch.int),
            short_term=torch.zeros((batch_dim, self.n_agents, n_short_term_points, 2), device=device, dtype=torch.float32), # Short-term reference path
            short_term_indices = torch.zeros((batch_dim, self.n_agents, n_short_term_points), device=device, dtype=torch.int),
        )

        # Determine position range
        x_min = self.ref_paths.long_term[:,:,0].min()
        x_max = self.ref_paths.long_term[:,:,0].max()
        y_min = self.ref_paths.long_term[:,:,1].min()
        y_max = self.ref_paths.long_term[:,:,1].max()
        x_pos_normalzer = x_max - x_min
        y_pos_normalzer = y_max - y_min
        
        # Handle the case where the reference path is a horizontal or vertical line
        if x_pos_normalzer == 0:
            x_pos_normalzer.fill_(1)
        if y_pos_normalzer == 0:
            y_pos_normalzer.fill_(1)
        
        self.normalizers = Normalizers(
            pos=torch.tensor([x_pos_normalzer, y_pos_normalzer], device=device, dtype=torch.float32),
            v=max_speed,
            rot=torch.tensor(2 * torch.pi, device=device, dtype=torch.float32)
        )
        
        weighting_ref_directions = torch.linspace(1, 0.2, steps=self.ref_paths.n_short_term_points-1, device=device, dtype=torch.float32)
        weighting_ref_directions /= weighting_ref_directions.sum()
        self.rewards = Rewards(
            progress=torch.tensor(reward_progress, device=device, dtype=torch.float32),
            weighting_ref_directions=weighting_ref_directions, # Progress in the weighted directions (directions indicating by closer short-term reference points have higher weights)
            higth_v=torch.tensor(reward_speed, device=device, dtype=torch.float32),
            reaching_goal=torch.tensor(reward_reaching_goal, device=device, dtype=torch.float32),
        )

        self.penalties = Penalties(
            deviate_from_ref_path=torch.tensor(penalty_deviate_from_ref_path, device=device, dtype=torch.float32),
            deviate_from_goal=torch.tensor(penalty_deviate_from_goal, device=device, dtype=torch.float32),
            weighting_deviate_from_ref_path=agent_width,
            time=torch.tensor(penalty_time, device=device, dtype=torch.float32),
        )
        
        self.observations = Observations(
            is_local=torch.tensor(is_local_observation, device=device, dtype=torch.bool),
            is_global_coordinate_sys=torch.tensor(is_global_coordinate_sys, device=device, dtype=torch.bool),
        )
        
        # Distances to boundaries and reference path, and also the closest point on the reference paths of agents
        self.distances = Distances(
            ref_paths=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.float32),
            closest_point_on_ref_path=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int),
            goal=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.float32),
        )
        
        self.thresholds = Thresholds(
            reaching_goal=torch.tensor(threshold_reaching_goal, device=device, dtype=torch.float32)
        )
        
        if self.parameters.is_dynamic_goal_reward:
            self.evaluation = Evaluation(
                pos_traj=torch.zeros((batch_dim, self.n_agents, self.parameters.max_steps, 2), device=device, dtype=torch.float32),
                v_traj=torch.zeros((batch_dim, self.n_agents, self.parameters.max_steps, 2), device=device, dtype=torch.float32),
                rot_traj=torch.zeros((batch_dim, self.n_agents, self.parameters.max_steps), device=device, dtype=torch.float32),
                deviation_from_ref_path=torch.zeros((batch_dim, self.n_agents, self.parameters.max_steps), device=device, dtype=torch.float32),
                path_tracking_error_mean=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.float32),
            )
        
        # Store the index of the nearest point on the reference path, which indicates the moving progress of the agent
        self.progress = torch.zeros((batch_dim, self.n_agents), device=device)
        self.progress_previous = torch.zeros((batch_dim, self.n_agents), device=device)
        
        # Use the kinematic bicycle model for the agent
        agent = Agent(
            name=f"agent_0",
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

        self.step_count = torch.zeros(batch_dim, device=device, dtype=torch.int)

        return world

    def reset_world_at(self, env_index: int = None):
        # print(f"[DEBUG] reset_world_at(): env_index = {env_index}")
        """
        This function resets the world at the specified env_index.

        Args:
        :param env_index: index of the environment to reset. If None a vectorized reset should be performed

        """
        i = 0 # Only one agent
        agent = self.world.agents[i]
        
        # if hasattr(self, 'training_info'):
        #     print(self.training_info["agents"]["episode_reward"].mean())
        
        # TODO Reset the reference path according to the training progress
        # self.ref_paths.long_term = get_reference_paths(self.n_agents, self.map_data) # A new random long-term reference path

        agent.set_pos(self.start_pos, batch_index=env_index)
        agent.set_rot(self.start_rot, batch_index=env_index)
        agent.set_vel(self.start_vel, batch_index=env_index)

        # Reset variables for the agent
        if env_index is None: # Reset all envs
            
            self.is_currently_at_goal[:] = False
            self.has_reached_goal[:] = False
            
            # Reset distances to the reference path          
            self.distances.ref_paths[:,i], self.distances.closest_point_on_ref_path[:,i] = get_perpendicular_distances(
                point=agent.state.pos, 
                boundary=self.ref_paths.long_term[i]
            )
            
            self.distances.goal[:,i] = (agent.state.pos - self.goal_pos).norm(dim=1)

            self.ref_paths.short_term[:,i], self.ref_paths.short_term_indices[:,i] = get_short_term_reference_path( 
                self.ref_paths.long_term[i],
                self.distances.closest_point_on_ref_path[:,i],
                self.ref_paths.n_short_term_points, 
                self.world.device
            )
            
            if self.parameters.is_dynamic_goal_reward:
                # Reset the data for evaluation for all envs
                self.evaluation.pos_traj[:] = 0
                self.evaluation.v_traj[:] = 0
                self.evaluation.rot_traj[:] = 0
                self.evaluation.path_tracking_error_mean[:] = 0

            # Reset the previous position
            agent.state.pos_previous = agent.state.pos.clone()

            # Store the index of the nearest point on the reference path, which indicates the moving progress of the agent
            # TODO Necessary?
            self.progress_previous[:,i] = self.progress[:,i].clone() # Store the values of previous time step 
            self.progress[:,i] = self.ref_paths.short_term_indices[:,i,0]
            
            self.step_count[:] = 0 

        else: # Reset the env specified by `env_index`
            self.is_currently_at_goal[env_index] = False
            self.has_reached_goal[env_index] = False
            
            # Reset distances to the reference path          
            self.distances.ref_paths[env_index,i], self.distances.closest_point_on_ref_path[env_index,i] = get_perpendicular_distances(
                point=agent.state.pos[env_index,:].unsqueeze(0), 
                boundary=self.ref_paths.long_term[i]
            )
            
            self.distances.goal[env_index,i] = (agent.state.pos[env_index,:] - self.goal_pos.unsqueeze(0)).norm(dim=1)

            # Reset the short-term reference path of agents in env `env_index`
            self.ref_paths.short_term[env_index,i], self.ref_paths.short_term_indices[env_index,i] = get_short_term_reference_path( 
                self.ref_paths.long_term[i], 
                self.distances.closest_point_on_ref_path[env_index,i].unsqueeze(0),
                self.ref_paths.n_short_term_points, 
                self.world.device
            )
            
            if self.parameters.is_dynamic_goal_reward:
                # Reset the data for evaluation for env `env_index`
                self.evaluation.pos_traj[env_index] = 0
                self.evaluation.v_traj[env_index] = 0
                self.evaluation.rot_traj[env_index] = 0
                self.evaluation.path_tracking_error_mean[env_index] = 0
            
            # Reset the previous position
            agent.state.pos_previous[env_index,:] = agent.state.pos[env_index,:].clone()
        
            # Store the index of the nearest point on the reference path, which indicates the moving progress of the agent
            # TODO Necessary?
            self.progress_previous[env_index,i] = self.progress[env_index,i].clone() # Store the values of previous time step 
            self.progress[env_index,i] = self.ref_paths.short_term_indices[env_index,i,0]
        
            self.step_count[env_index] = 0 


    def process_action(self, agent: Agent):
        """This function will be executed before the step function, i.e., states are not updated yet."""
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
        
        # Calculate the distance from the center of the agent to its reference path
        self.distances.ref_paths[:,agent_index], self.distances.closest_point_on_ref_path[:,agent_index] = get_perpendicular_distances(
            point=agent.state.pos, 
            boundary=self.ref_paths.long_term[agent_index]
        )

        if self.parameters.is_dynamic_goal_reward:
            # Update data for evaluation
            self.evaluation.pos_traj[:,agent_index,self.step_count[agent_index]] = agent.state.pos
            self.evaluation.v_traj[:,agent_index,self.step_count[agent_index]] = agent.state.vel
            self.evaluation.rot_traj[:,agent_index,self.step_count[agent_index]] = agent.state.rot.squeeze(1)
            self.evaluation.deviation_from_ref_path[:,agent_index,self.step_count[agent_index]] = self.distances.ref_paths[:,agent_index]
            self.evaluation.path_tracking_error_mean[:,agent_index] = self.evaluation.deviation_from_ref_path[:,agent_index].sum(dim=1) / (self.step_count + 1) # Step starts from 0

        ##################################################
        ## Penalty for deviating from reference path
        ##################################################
        self.rew += self.distances.ref_paths[:,agent_index] / self.penalties.weighting_deviate_from_ref_path * self.penalties.deviate_from_ref_path

        # Update the short-term reference path (extract from the long-term reference path)
        self.ref_paths.short_term[:,agent_index], self.ref_paths.short_term_indices[:,agent_index] = get_short_term_reference_path(
            self.ref_paths.long_term[agent_index], 
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
        # Handle non-loop reference path
        len_long_term_ref_path = len(self.ref_paths.long_term[agent_index])
        short_term_indices = torch.where(self.ref_paths.short_term_indices == len_long_term_ref_path - 1, len_long_term_ref_path - 2, self.ref_paths.short_term_indices)
        short_term_path_vec_normalized = self.ref_paths.long_term_vecs_normalized[short_term_indices[:,agent_index, 0:-1]] # Narmalized vector of the short-term reference path
        movement_normalized_proj = torch.sum(movement * short_term_path_vec_normalized, dim=2)
        movement_weighted_sum_proj = torch.matmul(movement_normalized_proj, self.rewards.weighting_ref_directions)
        self.rew += movement_weighted_sum_proj / (agent.max_speed * self.world.dt) * self.rewards.progress # Relative to the maximum possible movement
        
        ##################################################
        ## Reward for moving in a high velocity and the right direction
        ##################################################
        # TODO: optional?
        # v_proj = torch.sum(agent.state.vel.unsqueeze(1) * short_term_path_vec_normalized, dim=2)
        # v_proj_weighted_sum = torch.matmul(v_proj, self.rewards.weighting_ref_directions)
        # self.rew += v_proj_weighted_sum / agent.max_speed * self.rewards.higth_v

        # Save previous positions
        agent.state.pos_previous = agent.state.pos.clone() 
        
        if agent_index == 0:
            # Increment step by 1
            self.step_count += 1

        ##################################################
        ## Reward for reaching goal
        ##################################################
        # Update distances to goal positions
        self.distances.goal[:,agent_index] = (agent.state.pos - self.goal_pos).norm(dim=1)
        is_currently_at_goal = self.distances.goal[:,agent_index] <= self.thresholds.reaching_goal # If the agent is at its goal position
        # print(self.distances.goal[:,agent_index])
        
        goal_reward_factor = torch.ones(self.world.batch_dim, device=self.world.device, dtype=torch.float32)
        if self.parameters.is_dynamic_goal_reward:
            goal_reward_factor[:] = agent.shape.width / self.evaluation.path_tracking_error_mean[:,agent_index] # The smaller the mean tracking error, the higher the goal reward
        
        self.rew += is_currently_at_goal * self.rewards.reaching_goal * ~self.has_reached_goal * goal_reward_factor # Agents can only receive the goal reward once per iteration
        self.has_reached_goal[is_currently_at_goal] = True # Record if agents have reached their goal positions

        ##################################################
        ## Penalty for leaving goal position
        ##################################################
        # This penalty is only applied to agents that have reached their goal position before, with the purpose that these agents will stay at their goal positions
        self.rew += (self.distances.goal[:,agent_index] - self.thresholds.reaching_goal).clamp(min=0) * self.penalties.deviate_from_goal * self.has_reached_goal

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
            positions = torch.stack([a.state.pos for a in self.world.agents], dim=1) / self.normalizers.pos
            velocities = torch.stack([a.state.vel for a in self.world.agents], dim=1) / self.normalizers.v
            ##################################################
            ## Observation of short-term reference path
            ##################################################
            # Skip the first point on the short-term reference path, because, mostly, it is behind the agent. The second point is in front of the agent.
            obs_ref_point_norm = (self.ref_paths.short_term[:,agent_index,1:] / self.normalizers.pos).reshape(self.world.batch_dim, -1)
        else:
            positions = torch.zeros((self.world.batch_dim, 1, 2)) # Positions are at the origin of the local coordinate system
            velocities = torch.stack([a.state.vel for a in self.world.agents], dim=1) / self.normalizers.v
            velocities[:,:,0] = velocities.norm(dim=-1)
            velocities[:,:,1] = 0
            
            ##################################################
            ## Observation of short-term reference path
            ##################################################
            # Normalized short-term reference path relative to the agent's current position
            # Skip the first point on the short-term reference path, because, mostly, it is behind the agent. The second point is in front of the agent.
            ref_points_rel_norm = (self.ref_paths.short_term[:,agent_index,1:] - agent.state.pos.unsqueeze(1)) / self.normalizers.pos
            ref_points_rel_abs = ref_points_rel_norm.norm(dim=2)
            ref_points_rel_rot = torch.atan2(ref_points_rel_norm[:,:,1], ref_points_rel_norm[:,:,0]) - agent.state.rot
            obs_ref_point_norm = torch.stack(
                (
                    ref_points_rel_abs * torch.cos(ref_points_rel_rot), 
                    ref_points_rel_abs * torch.sin(ref_points_rel_rot)    
                ), dim=2
            ).reshape(self.world.batch_dim, -1)


        ##################################################
        ## Observations of self
        ##################################################
        obs_self = torch.hstack((
            positions.reshape(self.world.batch_dim, -1),    # Position
            velocities[:, agent_index],                     # Velocity 
        ))
        
        ##################################################
        ## Observation of distance to reference path
        ##################################################
        obs_ref_path_distances_norm = self.distances.ref_paths[:,agent_index].unsqueeze(-1) / self.normalizers.pos.norm()
        
        
        return torch.cat(
            [
                obs_self,
                obs_ref_path_distances_norm,
                obs_ref_point_norm,
            ],
            dim=-1,
        ) # [batch_dim, 3 + 3 + n_short_term_points*2 + 5*(n_agent-1)]


    def done(self):
        # print("[DEBUG] done()")
        is_done = self.has_reached_goal # [batch_dim]
        if is_done.any():
            print("done!")
        
        if self.parameters.is_testing_mode:
            # print(f"Step: {self.step_count}")
            is_done = torch.zeros(is_done.shape, device=self.world.device, dtype=torch.bool)
            if self.step_count % 20 == 0:
                print("You are in testing mode. Collisions do not terminate the simulation.")
        
        return is_done

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        """
        This function computes the info dict for 'agent' in a vectorized way
        The returned dict should have a key for each info of interest and the corresponding value should
        be a tensor of shape (n_envs, info_size)

        Implementors can access the world at 'self.world'

        To increase performance, tensors created should have the device set, like:
        torch.tensor(..., device=self.world.device)

        :param agent: Agent batch to compute info of
        :return: info: A dict with a key for each info of interest, and a tensor value  of shape (n_envs, info_size)
        """
        agent_index = self.world.agents.index(agent) # Index of the current agent
        
        info = {
            "pos": agent.state.pos,
            "vel": agent.state.vel,
            "rot": agent.state.rot,
            "deviation_from_ref_path": self.distances.ref_paths[:,agent_index],
        }
        
        return info
    
    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering

        geoms = []

        # Visualize reference path
        for i in range(len(self.ref_paths.long_term_yaws)):
            if self.ref_paths.long_term_lengths[i] != 0:
                # Handle the case where two successive points on the reference path are overlapping
                geom = Line(
                    length=self.ref_paths.long_term_lengths[i]
                ).get_geometry()

                xform = rendering.Transform()
                geom.add_attr(xform)

                # Set the positions of the centers of the line segment
                xform.set_translation(
                    self.ref_paths.long_term_center_points[i, 0], 
                    self.ref_paths.long_term_center_points[i, 1]
                )
                
                # Set orientations
                xform.set_rotation(self.ref_paths.long_term_yaws[i])
                
                color = Color.BLACK.value
                if isinstance(color, torch.Tensor) and len(color.shape) > 1:
                    color = color[env_index]
                geom.set_color(*color)
                geoms.append(geom)
                
        # Visualize goal
        color = Color.RED.value
        circle = rendering.make_circle(radius=self.thresholds.reaching_goal, filled=True)
        xform = rendering.Transform()
        circle.add_attr(xform)
        xform.set_translation(self.goal_pos[0], self.goal_pos[1])
        circle.set_color(*color)
        geoms.append(circle)

        
            
        if self.parameters.is_visualize_short_term_path:
            for agent_i in range(self.n_agents):
                center_points, lengths, yaws, _ = get_center_length_yaw_polyline(polyline=self.ref_paths.short_term[env_index,agent_i])
                for j in range(len(lengths)):
                    if lengths[j] != 0:
                        # Handle the case where two successive points on the reference path are overlapping
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
                        
                        color = Color.GREEN.value
                        if isinstance(color, torch.Tensor) and len(color.shape) > 1:
                            color = color[env_index]
                        geom.set_color(*color)
                        geoms.append(geom)
                    
        return geoms


if __name__ == "__main__":
    scenario_name = "car_like_robots_path_tracking" # car_like_robots_road_traffic, car_like_robots_go_to_formation, car_like_robots_path_tracking
    parameters = Parameters(
        n_agents=4,
        dt=0.1, # [s] sample time 
        device="cpu" if not torch.backends.cuda.is_built() else "cuda:0",  # The divice where learning is run
        scenario_name=scenario_name,
        
        # Training parameters
        n_iters=100, # Number of sampling and training iterations (on-policy: rollouts are collected during sampling phase, which will be immediately used in the training phase of the same iteration),
        frames_per_batch=2**10, # Number of team frames collected per training iteration (minibatch_size*10)
        num_epochs=30, # Number of optimization steps per training iteration,
        minibatch_size=2*8, # Size of the mini-batches in each optimization step (2**9 - 2**12?),
        lr=4e-4, # Learning rate,
        max_grad_norm=1.0, # Maximum norm for the gradients,
        clip_epsilon=0.2, # clip value for PPO loss,
        gamma=0.98, # discount factor (empirical formula: 0.1 = gamma^t, where t is the number of future steps that you want your agents to predict {0.96 -> 56 steps, 0.98 - 114 steps, 0.99 -> 229 steps, 0.995 -> 459 steps})
        lmbda=0.9, # lambda for generalised advantage estimation,
        entropy_eps=4e-4, # coefficient of the entropy term in the PPO loss,
        max_steps=2**8, # Episode steps before done (512)
        
        is_save_intermidiate_model=True, # Is this is true, the model with the hightest mean episode reward will be saved,
        n_nearing_agents_observed=4,
        episode_reward_mean_current=0.00,
        
        is_load_model=False, # Load offline model if available. The offline model in `where_to_save` whose name contains `episode_reward_mean_current` will be loaded
        is_continue_train=False, # If offline models are loaded, whether to continue to train the model
        mode_name=None, 
        episode_reward_intermidiate=-1e3, # The initial value should be samll enough
        where_to_save=f"outputs/{scenario_name}_ppo/test_nondynamic_goal_reward_no_v_rew_small_goal_threshold/", # folder where to save the trained models, fig, data, etc.
        
        # Scenario parameters
        is_local_observation=False, 
        is_global_coordinate_sys=True,
        n_short_term_points=6,
        is_testing_mode=False,
        is_visualize_short_term_path=True,
        
        path_tracking_type='sine', # [relevant to path-tracking scenarios] should be one of 'line', 'turning', 'circle', 'sine', and 'horizontal_8'
        is_dynamic_goal_reward=False, # [relevant to path-tracking scenarios] set to True if the goal reward is dynamically adjusted based on the performance of agents' history trajectories 
    )
    
    scenario = ScenarioPathTracking()
    scenario.parameters = parameters
    
    render_interactively(
        scenario=scenario, control_two_agents=False, shared_reward=False,
    )
