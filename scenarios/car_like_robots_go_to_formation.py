#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World, Line
from vmas.simulator.dynamics.kinematic_bicycle import KinematicBicycleDynamics
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color

import numpy as np
import matplotlib.pyplot as plt

# Collision check
from utilities.helper_scenario import interX

agent_width = 0.11 # The width of the agent in [m]
agent_length = 0.22 # The length of the agent in [m]
agent_wheelbase_front = 0.11 # Front wheelbase in [m]
agent_wheelbase_rear = agent_length - agent_wheelbase_front # Rear wheelbase in [m]
agent_max_speed = 1.0 # Maximum speed in [m/s]
agent_max_steering_angle = 40 # Maximum steering angle in degree
n_agents = 9 # The number of agents
agent_mass = 0.5 # The mass of each agent in [m]

world_x_dim = 4.5 # The x-dimension of the world in [m]
world_y_dim = 4.0 # The y-dimension of the world in [m]

pos_normalizer = torch.tensor([world_x_dim, world_y_dim])

agent_poses_start = np.array([
    [1.6, -1.85, np.pi/2],
    [1.2, -1.85, np.pi/2],
    [0.8, -1.85, np.pi/2],
    [1.6, -1.35, np.pi/2],
    [1.2, -1.35, np.pi/2],
    [0.8, -1.35, np.pi/2],
    [1.6, -0.85, np.pi/2],
    [1.2, -0.85, np.pi/2],
    [0.8, -0.85, np.pi/2]
]) # The initial poses of the agents [x, y, yaw angle]

agent_poses_goal = np.array([
    [-1.6, 1.85, np.pi/2],
    [-1.2, 1.85, np.pi/2],
    [-0.8, 1.85, np.pi/2],
    [-1.6, 1.35, np.pi/2],
    [-1.2, 1.35, np.pi/2],
    [-0.8, 1.35, np.pi/2],
    [-1.6, 0.85, np.pi/2],
    [-1.2, 0.85, np.pi/2],
    [-0.8, 0.85, np.pi/2]
]) # The goal poses of the agents [x, y, yaw angle]

obstacle_thickness = 0.05 # The width of the obstacles that are positioned in the boundaries of the world

goal_threshold = agent_width / 2 # Threshold that agents have arrived at their goal positions

# Reward
goal_first_time_reward = 100            # Reward agents if they arrive their goal positions for and only for the first time (to discourage agents from leaving and entering their goal positions repeatedly)
goal_staying_reward = 2                 # Reward agents if they stay at their goal positions
goal_approach_shaping_factor = 10       # A shaping factor to encourage agents approaching their goal positions
high_speed_reward_factor = 2            # A factor for the reward that encourages agents move in a high speed

# Penalty
collision_with_agents_penalty = -20     # Penalty for collisions with others agents
collision_with_obstacles_penalty = -20  # Penalty for collisions with obstacles such as lane boundaries
time_penalty = -0                       # Time-based penalty
goal_leave_penalty = -10                # Penaltize from leaving goal positions


def get_rectangle_corners(center, yaw, width, length):
    """
    Compute the corners of rectangles for a batch of agents given their centers, yaws (rotations),
    widths, and lengths, using PyTorch tensors.

    :param center: Center positions of the rectangles (batch_dim, 2).
    :param yaw: Rotation angles in radians (batch_dim, 1).
    :param width: Width of the rectangles.
    :param length: Length of the rectangles.
    :return: A tensor of corner points of the rectangles for each agent in the batch (batch_dim, 4, 2).
    """
    batch_dim = center.size(0)
    width_half = width / 2
    length_half = length / 2

    # Corner points relative to the center
    corners = torch.tensor([[length_half, width_half], [length_half, -width_half], [-length_half, -width_half], [-length_half, width_half], [length_half, width_half]], dtype=center.dtype, device=center.device) # Repeat the first vertex to close the shape

    # Expand corners to match batch size
    corners = corners.unsqueeze(0).repeat(batch_dim, 1, 1)

    # Create rotation matrices for each agent
    cos_yaw = torch.cos(yaw).squeeze(1)
    sin_yaw = torch.sin(yaw).squeeze(1)

    # Rotation matrix for each agent
    rot_matrix = torch.stack([
        torch.stack([cos_yaw, -sin_yaw], dim=1),
        torch.stack([sin_yaw, cos_yaw], dim=1)
    ], dim=1)

    # Apply rotation to corners
    rotated_corners = torch.matmul(rot_matrix, corners.transpose(1, 2)).transpose(1, 2)

    # Add center positions to the rotated corners
    rotated_translated_corners = rotated_corners + center.unsqueeze(1)

    return rotated_translated_corners

class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        print("[DEBUG] make_world() car_like_robots")
        self.shared_reward = kwargs.get("shared_reward", False)

        self.n_agents = kwargs.get("n_agents", 4) # Agent number
        width = kwargs.get("width", agent_width) # Agent width
        l_f = kwargs.get("l_f", agent_wheelbase_front) # Distance between the front axle and the center of gravity
        l_r = kwargs.get("l_r", agent_wheelbase_rear) # Distance between the rear axle and the center of gravity
        max_steering_angle = kwargs.get("max_steering_angle", torch.deg2rad(torch.tensor(agent_max_steering_angle)))
        max_speed = kwargs.get("max_speed", torch.tensor(agent_max_speed))

        # Start poses and goal poses
        self.poses_start = agent_poses_start
        self.poses_goal = agent_poses_goal

        # Make world
        world = World(batch_dim, device, x_semidim=world_x_dim, y_semidim=world_y_dim)

        self.agent_at_goal = torch.zeros(world.batch_dim, self.n_agents, device=world.device, dtype=torch.bool) # Whether agents are at their goal positions
        
        # Add agents and their goal positions
        for i in range(self.n_agents):
            # Use the kinematic bicycle model for each agent
            agent = Agent(
                name=f"agent_{i}",
                shape=Box(length=l_f+l_r, width=width),
                collide=False,
                render_action=True,
                u_range=max_speed, # Control command serves as velocity command 
                u_rot_range=max_steering_angle, # Control command serves as steering command 
                u_rot_multiplier=1,
                max_speed=max_speed,
            )
            agent.dynamics = KinematicBicycleDynamics(
                agent, world, width=width, l_f=l_f, l_r=l_r, max_steering_angle=max_steering_angle, integration="rk4" # one of "euler", "rk4"
            )     

            world.add_agent(agent)

            goal = Landmark(
                name=f"goal_{i}",
                collide=False,
                shape=Box(length=l_f+l_r, width=width),
                color=Color.LIGHT_GREEN,
            )

            agent.goal = goal
            world.add_landmark(goal)

        # Consider each boundary of the rectangle world an obstacle
        obstacle_lengths = [world.x_semidim, world.x_semidim, world.y_semidim, world.y_semidim]
        obstacle_widths = [obstacle_thickness, obstacle_thickness, world.y_semidim, world.y_semidim]
        self.n_obstacles = len(obstacle_lengths)
        for i in range(self.n_obstacles):
            obstacle = Landmark(
                name=f"obstacle_{i}",
                collide=False, 
                movable=False,
                shape=Box(length=obstacle_lengths[i], width=obstacle_thickness),
                color=Color.RED,
                collision_filter=lambda e: not isinstance(e.shape, Box),
            )
            
            # Add the landmark to the world
            world.add_landmark(obstacle)
        
        # Initialize collision matrix 
        self.collision_with_agents = torch.zeros(world.batch_dim, self.n_agents, self.n_agents, dtype=torch.bool) # [batch_dim, n_agents, n_agents] The indices of colliding agents
        
        self.collision_with_obstacles = torch.zeros(world.batch_dim, self.n_agents, self.n_obstacles, dtype=torch.bool) # [batch_dim, n_agents] The indices of agents that collide with landmarks

        return world

    def reset_world_at(self, env_index: int = None):
        # print(f"[DEBUG] reset_world_at(): env_index = {env_index}")
        """
        This function resets the world at the specified env_index.

        Args:
        :param env_index: index of the environment to reset. If None a vectorized reset should be performed

        """

        ##
        agents = self.world.agents
        goals = self.world.landmarks

        # Set the start and goal positions for each agent
        for i in range(self.n_agents):
            # Set initial position
            agents[i].set_pos(torch.tensor(self.poses_start[i][:2], device=self.world.device, dtype=torch.float32), batch_index=env_index)
            # Set initial rotation (yaw angle)
            agents[i].set_rot(torch.tensor(self.poses_start[i][2], device=self.world.device, dtype=torch.float32), batch_index=env_index)
            # Set goal position
            goals[i].set_pos(torch.tensor(self.poses_goal[i][:2], device=self.world.device, dtype=torch.float32), batch_index=env_index)
            # Set goal rotation (yaw angle)
            goals[i].set_rot(torch.tensor(self.poses_goal[i][2], device=self.world.device, dtype=torch.float32), batch_index=env_index)
            
            # Update the global shaping for each agent
            if env_index is None:
                agents[i].global_shaping = (
                    torch.linalg.vector_norm(agents[i].state.pos - agents[i].goal.state.pos, dim=1) * goal_approach_shaping_factor
                )
            else:
                agents[i].global_shaping[env_index] = (
                    torch.linalg.vector_norm(agents[i].state.pos[env_index] - agents[i].goal.state.pos[env_index]) * goal_approach_shaping_factor
                )

        # Reset obstacles
        obstacles = self.world.landmarks[self.n_agents :] # Note that the first several `landmarks` are the goal positions (not obstacles)
        # Define obstacle positions (center of gravity)
        obstacle_positions = [
            # Top boundary
            (0, self.world.y_semidim / 2),
            # Bottom boundary
            (0, -self.world.y_semidim / 2),
            # Left boundary
            (-self.world.x_semidim / 2, 0),
            # Right boundary
            (self.world.x_semidim / 2, 0)
        ]

        # Set the position of the obstacle
        for i, obstacle in enumerate(obstacles):
            obstacle.set_pos(
                torch.tensor(
                    obstacle_positions[i],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )

            # Horizontal rectangle has the default rotation angle, 0 degree. Therefore, vertical rectangles are considered to have a rotation angle of 90 degree
            obstacle_rot = torch.deg2rad(torch.tensor(
                [0, 0, 90, 90],
                dtype=torch.float32,
                device=self.world.device
            ))
            obstacle.set_rot(obstacle_rot[i], batch_index=env_index)

        # Reset variables
        if env_index is None:
            self.agent_at_goal = torch.zeros(self.world.batch_dim, self.n_agents, device=self.world.device, dtype=torch.bool)
            self.collision_with_agents = torch.zeros(self.world.batch_dim, self.n_agents, self.n_agents, dtype=torch.bool)
            self.collision_with_obstacles = torch.zeros(self.world.batch_dim, self.n_agents, self.n_obstacles, dtype=torch.bool)
        else:
            self.agent_at_goal[env_index,:] = torch.tensor(False)
            self.collision_with_agents[env_index,:,:] = torch.tensor(False)
            self.collision_with_obstacles[env_index,:,:] = torch.tensor(False)


    def process_action(self, agent: Agent):
        # print("[DEBUG] process_action()")
        if hasattr(agent, 'dynamics') and hasattr(agent.dynamics, 'process_force'):
            agent.dynamics.process_force()
        else:
            # The agent does not have a dynamics property, or it does not have a process_force method
            pass

    def reward(self, agent: Agent):
        # print("[DEBUG] reward()")

        # Get agent index (Ensure that the naming of agents follows the pattern agent_{i}, where i is the index of agent i (starting from 0))
        index_str = agent.name.split('_')[1]
        agent_index = int(index_str)
        # print(f'[DEBUG] agent name: {agent.name}')

        # If rewards are shared among agents
        if self.shared_reward:
            # Only calculate the shared reward once for the first agent
            if agent_index == 0:
                # Initialize the reward to zero for all instances in the batch
                self.rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)
                
                # Loop through all agents to calculate the shared reward
                for a in self.world.agents:
                    # Calculate the Euclidean distance from the agent's current position to its goal position
                    dist_to_goal = torch.linalg.vector_norm(a.state.pos - a.goal.state.pos, dim=1)
                    
                    # Compute the shaping value for the agent based on its distance to the goal
                    agent_shaping = dist_to_goal * goal_approach_shaping_factor
                    
                    # Update the reward based on the difference between the previous shaping value and the current one
                    self.rew += a.global_shaping - agent_shaping
                    
                    # Store the current shaping value for future reward calculations
                    a.global_shaping = agent_shaping
        else:
            # If rewards are individual (not shared), then calculate reward for the provided agent only
            self.rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)
            dist_to_goal = torch.linalg.vector_norm(agent.state.pos - agent.goal.state.pos, dim=1)
            agent_shaping = dist_to_goal * goal_approach_shaping_factor
            self.rew += agent.global_shaping - agent_shaping
            agent.global_shaping = agent_shaping

            # Check if agents are currently at their goal positions
            is_at_goal_position = dist_to_goal <= goal_threshold

            # Goal achievement reward
            is_first_time_enter_goal_position = is_at_goal_position &  ~self.agent_at_goal[:,agent_index]
            is_already_at_goal_position = is_at_goal_position & self.agent_at_goal[:,agent_index]
            self.rew[is_first_time_enter_goal_position] += goal_first_time_reward # The first time enter goal positions
            self.rew[is_already_at_goal_position] += goal_staying_reward # Already at goal positions

            # Penalize movement after reaching the goal
            is_leave_goal = (dist_to_goal > goal_threshold) & self.agent_at_goal[:,agent_index] # An anget is penalized if it reached its goal in the previous time step but leaves it in the current time step 
            self.rew[is_leave_goal] += goal_leave_penalty

            # Reward for moving with high speeds
            # self.rew += torch.norm(agent.state.vel, dim=1) / agent_max_speed * high_speed_reward_factor

            if torch.all(is_first_time_enter_goal_position):
                print("[DEBUG] First time enter goal positions")

            if torch.all(is_already_at_goal_position):
                print("[DEBUG] Stay at goal positions")

            if torch.all(is_leave_goal):
                print("[DEBUG] Leave goal positions")

            # Update the status whether agents are at their goal positions
            self.agent_at_goal[is_at_goal_position,agent_index] = True
            
        # Check for collisions between each pair of agents in the environment
        if agent_index == 0: # Avoid repeated check
            self.collision_with_agents = torch.zeros(self.world.batch_dim, self.n_agents, self.n_agents, dtype=torch.bool) # Default to no collision
            self.collision_with_obstacles = torch.zeros(self.world.batch_dim, self.n_agents, self.n_obstacles, dtype=torch.bool) # Default to no collision
            for i in range(self.n_agents):
                for j in range(i+1, self.n_agents):
                    agent_i = self.world.agents[i]
                    agent_j = self.world.agents[j]
                    # Get the rectangle corners for each agent
                    corners_i = get_rectangle_corners(agent_i.state.pos, agent_i.state.rot, agent_i.shape.width, agent_i.shape.length)
                    corners_j = get_rectangle_corners(agent_j.state.pos, agent_j.state.rot, agent_j.shape.width, agent_j.shape.length)
                    # Check for intersection using the interX function
                    collision_batch_index = interX(corners_i, corners_j, False)
                    self.collision_with_agents[collision_batch_index, i, j] = True
                    self.collision_with_agents[collision_batch_index, j, i] = True

                # Check for collisions between agents and obstacles
                for k in range(len(self.world.landmarks)):
                    landmark = self.world.landmarks[k]
                    if "obstacle" in landmark.name: # We are only interested in obstacles hier
                        landmark_index = int(landmark.name.split('_')[1])
                        corners_landmark_j = get_rectangle_corners(landmark.state.pos, landmark.state.rot, landmark.shape.width, landmark.shape.length) 
                        collision_with_obstacles_index = interX(corners_i, corners_landmark_j, False) # [batch_dim]
                        self.collision_with_obstacles[collision_with_obstacles_index, i, landmark_index] = True

        # Penalize collisions 
        is_collide_with_agents = self.collision_with_agents[:, agent_index].sum(dim=1) > 0 # [batch_dim] Summing over the second dimension to see if agent i has collided with any agent
        is_collide_with_obstacles = self.collision_with_obstacles[:, agent_index].sum(dim=1) > 0
        # if torch.all(is_collide_with_agents):
        #     print("[DEBUG] Collide with other agents!!!")
        # if torch.all(is_collide_with_obstacles):
        #     print("[DEBUG] Collide with obstacles!!!")


        # Apply penalty to each batch where collisions occur
        self.rew[is_collide_with_agents] += collision_with_agents_penalty

        # Apply penalty to each batch where collisions occur
        self.rew[is_collide_with_obstacles] += collision_with_obstacles_penalty 

        # Subtract the time penalty from the reward
        self.rew += time_penalty

        return self.rew


    def observation(self, agent: Agent):
        # print("[DEBUG] observation()")
        """
        Generate an observation for the given agent.

        Parameters:
        - agent (Agent): The agent for which the observation is to be generated.

        Returns:
        - torch.Tensor: The observation tensor for the given agent.
        """


        # Normalized goal position relative to the agent
        self_goal_rel_norm = (agent.goal.state.pos - agent.state.pos) / pos_normalizer

        # Self state
        self_state_norm = []
        self_pos_norm = agent.state.pos / pos_normalizer # TODO Evaluate if self position is of interest
        self_v_norm = agent.state.vel / agent.max_speed
        self_rot_norm = (agent.state.rot % (2 * torch.pi)) / (2 * torch.pi)

        self_state_norm = torch.cat(
            [
                self_goal_rel_norm, # Instead of observing self position, we observe the goal position here
                self_v_norm,
                self_rot_norm
            ],
            dim=1
        ) # [batch_dim, 5]


        # Observations of all obstacles (Simplified version, as we only have the for boundaries of the rectangle world as obstacles; therefore, the most effective observation is the distance to the boundaries)
        obstacles_pos_rel_norm = []
        # Skipping landmarks indicating goal positions
        obstacles = self.world.landmarks[self.n_agents:]

        for obstacle in obstacles:
            obstacles_pos_rel_norm_i = (obstacle.state.pos - agent.state.pos) / pos_normalizer
            if obstacle.state.rot[0,0] == 0: # Horizontal rectangle; in case of batch environments, [0,0] simply selects the first environment 
                obstacles_pos_rel_norm.append(obstacles_pos_rel_norm_i[:,1].reshape(-1,1)) # Relative position in y-direction
            else: # Verticle rectangle
                obstacles_pos_rel_norm.append(obstacles_pos_rel_norm_i[:,0].reshape(-1,1)) # Relative position in x-direction

        # Observations of all other agents
        other_agent_states_rel_norm = []
        for other_agent in self.world.agents:
            if other_agent != agent: # Exclude the current agent
                other_agents_pos_rel_norm = (other_agent.state.pos - agent.state.pos) / pos_normalizer
                other_agents_v_norm = other_agent.state.vel / agent.max_speed
                other_agents_rot_norm = (other_agent.state.rot % (2 * torch.pi)) / (2 * torch.pi)
                other_agent_states_rel_norm.append(
                    torch.cat(
                        [other_agents_pos_rel_norm, 
                         other_agents_v_norm, 
                         other_agents_rot_norm], 
                        dim=1)
                )

        # Concatenate the agent's position, velocity, relative position to its goal, 
        # all the relative positions of obstacles, and the relative positions of all
        # other agents to generate the complete observation.
        return torch.cat(
            [
                self_state_norm,
                *obstacles_pos_rel_norm,
                *other_agent_states_rel_norm
            ],
            dim=-1,
        ) # [batch_dim, 5+4+5*(n_agent-1)]

    def done(self):
        # print("[DEBUG] done()")
        is_collision_with_agents_occur = torch.any(self.collision_with_agents.view(self.world.batch_dim,-1), dim=-1) # [batch_dim]
        is_collision_with_obstacles_occur = torch.any(self.collision_with_obstacles.view(self.world.batch_dim,-1), dim=-1) # [batch_dim]
        is_all_agents_at_goal = torch.all(self.agent_at_goal, dim=1) # [batch_dim]
        is_done = is_collision_with_agents_occur | is_collision_with_obstacles_occur | is_all_agents_at_goal

        return is_done


    def extra_render(self, env_index: int = 0):
        # print("[DEBUG] extra_render()")
        from vmas.simulator import rendering

        geoms = []

        for i in range(4):
            # For the lines' length, adjust based on world dimensions
            if i % 2:  # horizontal lines
                geom = Line(length=self.world.x_semidim).get_geometry()
            else:  # vertical lines
                geom = Line(length=self.world.y_semidim).get_geometry()

            xform = rendering.Transform()
            geom.add_attr(xform)

            # Set the positions of the centers of the boundaries
            xform.set_translation(
                0.0 if i % 2 else (self.world.x_semidim / 2 if i == 0 else -self.world.x_semidim / 2),
                0.0 if not i % 2 else (self.world.y_semidim / 2 if i == 1 else -self.world.y_semidim / 2)
            )
            
            # Set orientations
            xform.set_rotation(torch.pi / 2 if not i % 2 else 0.0)
            
            color = Color.BLACK.value
            if isinstance(color, torch.Tensor) and len(color.shape) > 1:
                color = color[env_index]
            geom.set_color(*color)
            geoms.append(geom)

        return geoms


if __name__ == "__main__":
    render_interactively(
        __file__, control_two_agents=False, shared_reward=False,
    )
