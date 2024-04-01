from matplotlib import pyplot as plt
import torch

from utilities.colors import Color


##################################################
## Custom Classes
##################################################
class Normalizers:
    def __init__(self, pos = None, pos_world = None, v = None, rot = None, action_steering = None, action_vel = None, distance_lanelet = None, distance_agent = None, distance_ref = None, priority = None,):
        self.pos = pos
        self.pos_world = pos_world
        self.v = v
        self.rot = rot
        self.action_steering = action_steering
        self.action_vel = action_vel
        self.distance_lanelet = distance_lanelet
        self.distance_agent = distance_agent
        self.distance_ref = distance_ref
        self.priority = priority

class Rewards:
    def __init__(self, progress = None, weighting_ref_directions = None, higth_v = None, reach_goal = None, reach_intermediate_goal = None,):
        self.progress = progress
        self.weighting_ref_directions = weighting_ref_directions
        self.higth_v = higth_v
        self.reach_goal = reach_goal
        self.reach_intermediate_goal = reach_intermediate_goal


class Penalties:
    def __init__(self, deviate_from_ref_path = None, deviate_from_goal = None, weighting_deviate_from_ref_path = None, near_boundary = None, near_other_agents = None, collide_with_agents = None, collide_with_boundaries = None, collide_with_obstacles = None, leave_world = None, time = None, change_steering = None):
        self.deviate_from_ref_path = deviate_from_ref_path  # Penalty for deviating from reference path
        self.deviate_from_goal = deviate_from_goal          # Penalty for deviating from goal position 
        self.weighting_deviate_from_ref_path = weighting_deviate_from_ref_path
        self.near_boundary = near_boundary                  # Penalty for being too close to lanelet boundaries
        self.near_other_agents = near_other_agents          # Penalty for being too close to other agents
        self.collide_with_agents = collide_with_agents      # Penalty for colliding with other agents
        self.collide_with_boundaries = collide_with_boundaries  # Penalty for colliding with lanelet boundaries
        self.collide_with_obstacles = collide_with_obstacles  # Penalty for colliding with obstacles
        self.leave_world = leave_world  # Penalty for leaving the world
        self.time = time                                    # Penalty for losing time
        self.change_steering = change_steering # Penalty for changing steering direction
        
class Thresholds:
    def __init__(self, deviate_from_ref_path = None, near_boundary_low = None, near_boundary_high = None, near_other_agents_low = None, near_other_agents_high = None, reach_goal = None, reach_intermediate_goal = None, change_steering = None, no_reward_if_too_close_to_boundaries = None, no_reward_if_too_close_to_other_agents = None, distance_mask_agents = None):
        self.deviate_from_ref_path = deviate_from_ref_path
        self.near_boundary_low = near_boundary_low
        self.near_boundary_high = near_boundary_high
        self.near_other_agents_low = near_other_agents_low
        self.near_other_agents_high = near_other_agents_high
        self.reach_goal = reach_goal                              # Threshold less than which agents are considered at their goal positions
        self.reach_intermediate_goal = reach_intermediate_goal    # Threshold less than which agents are considered at their intermediate goal positions
        self.change_steering = change_steering
        self.no_reward_if_too_close_to_boundaries = no_reward_if_too_close_to_boundaries # Agents get no reward if they are too close to lanelet boundaries
        self.no_reward_if_too_close_to_other_agents = no_reward_if_too_close_to_other_agents # Agents get no reward if they are too close to other agents
        self.distance_mask_agents = distance_mask_agents # Threshold above which nearing agents will be masked

class ReferencePathsMapRelated:
    def __init__(self, long_term_all = None, long_term_intersection = None, long_term_merge_in = None, long_term_merge_out = None, point_extended_all = None, point_extended_intersection = None, point_extended_merge_in = None, point_extended_merge_out = None, long_term_vecs_normalized = None, point_extended = None, sample_interval = None):
        self.long_term_all = long_term_all                              # All long-term reference paths
        self.long_term_intersection = long_term_intersection            # Long-term reference paths for the intersection scenario
        self.long_term_merge_in = long_term_merge_in                    # Long-term reference paths for the mergin in scenario
        self.long_term_merge_out = long_term_merge_out                  # Long-term reference paths for the merge out scenario
        self.point_extended_all = point_extended_all                    # Extend the long-term reference paths by one point at the end
        self.point_extended_intersection = point_extended_intersection  # Extend the long-term reference paths by one point at the end
        self.point_extended_merge_in = point_extended_merge_in          # Extend the long-term reference paths by one point at the end
        self.point_extended_merge_out = point_extended_merge_out        # Extend the long-term reference paths by one point at the end

        self.long_term_vecs_normalized = long_term_vecs_normalized      # Normalized vectors of the line segments on the long-term reference path
        self.point_extended = point_extended                            # Extended point for a non-loop reference path (address the oscillations near the goal)
        self.sample_interval = sample_interval                          # Integer, sample interval from the long-term reference path for the short-term reference paths
                                                                        # TODO Instead of an integer, sample with a fixed distance from the long-term reference paths when reading the map data

        
class ReferencePathsAgentRelated:
    def __init__(self, long_term: torch.Tensor = None, long_term_vec_normalized: torch.Tensor = None, point_extended: torch.Tensor = None, left_boundary: torch.Tensor = None, right_boundary: torch.Tensor = None, entry: torch.Tensor = None, exit: torch.Tensor = None, n_points_long_term: torch.Tensor = None, n_points_left_b: torch.Tensor = None, n_points_right_b: torch.Tensor = None, is_loop: torch.Tensor = None, n_points_nearing_boundary: torch.Tensor = None, nearing_points_left_boundary: torch.Tensor = None, nearing_points_right_boundary: torch.Tensor = None, short_term: torch.Tensor = None, short_term_indices: torch.Tensor = None, scenario_id: torch.Tensor = None, path_id: torch.Tensor = None, point_id: torch.Tensor = None):
        self.long_term = long_term                # Actual long-term reference paths of agents
        self.long_term_vec_normalized = long_term_vec_normalized # Normalized vectories on the long-term trajectory
        self.point_extended = point_extended
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.entry = entry                      # [for non-loop path only] Line segment of entry
        self.exit = exit                        # [for non-loop path only] Line segment of exit
        self.is_loop = is_loop                  # Whether the reference path is a loop
        self.n_points_long_term = n_points_long_term            # The number of points on the long-term reference paths
        self.n_points_left_b = n_points_left_b                  # The number of points on the left boundary of the long-term reference paths 
        self.n_points_right_b = n_points_right_b                # The number of points on the right boundary of the long-term reference paths 
        self.short_term = short_term                            # Short-term reference path
        self.short_term_indices = short_term_indices            # Indices that indicate which part of the long-term reference path is used to build the short-term reference path
        self.n_points_nearing_boundary = n_points_nearing_boundary              # Number of points on nearing boundaries to be observed
        self.nearing_points_left_boundary = nearing_points_left_boundary      # Nearing left boundary
        self.nearing_points_right_boundary = nearing_points_right_boundary    # Nearing right boundary

        self.scenario_id = scenario_id  # Which scenarios agents are (current implementation includes (1) intersection, (2) merge-in, and (3) merge-out)
        self.path_id = path_id          # Which paths agents are
        self.point_id = point_id        # Which points agents are

class Observations:
    def __init__(self, is_partial = None, is_global_coordinate_sys = None, n_nearing_agents = None, nearing_agents_indices = None, n_nearing_obstacles_observed = None, is_add_noise = None, noise_level = None, n_stored_steps = None, n_observed_steps = None, is_observe_corners = None, past_pri: torch.Tensor = None, past_pos: torch.Tensor = None, past_rot: torch.Tensor = None, past_corners: torch.Tensor = None, past_vel: torch.Tensor = None, past_short_term_ref_points: torch.Tensor = None, past_action_vel: torch.Tensor = None, past_action_steering: torch.Tensor = None, past_distance_to_ref_path: torch.Tensor = None, past_distance_to_boundaries: torch.Tensor = None, past_distance_to_left_boundary: torch.Tensor = None, past_distance_to_right_boundary: torch.Tensor = None, past_distance_to_agents: torch.Tensor = None, past_left_boundary: torch.Tensor = None, past_right_boundary: torch.Tensor = None):
        self.is_partial = is_partial    # Local observation
        self.is_global_coordinate_sys = is_global_coordinate_sys
        self.n_nearing_agents = n_nearing_agents
        self.nearing_agents_indices = nearing_agents_indices
        self.n_nearing_obstacles_observed = n_nearing_obstacles_observed
        self.is_add_noise = is_add_noise            # Whether to add noise to observations
        self.noise_level = noise_level              # Whether to add noise to observations
        self.n_stored_steps = n_stored_steps        # Number of past steps to store
        self.n_observed_steps = n_observed_steps    # Number of past steps to observe
        self.is_observe_corners = is_observe_corners

        self.past_pri = past_pri            # Past priorities
        self.past_pos = past_pos            # Past positions
        self.past_rot = past_rot            # Past rotations
        self.past_corners = past_corners    # Past corners
        self.past_vel = past_vel            # Past velocites
        
        self.past_short_term_ref_points = past_short_term_ref_points    # Past short-term reference points
        self.past_left_boundary = past_left_boundary  # Past left lanelet boundary
        self.past_right_boundary = past_right_boundary  # Past right lanelet boundary

        self.past_action_vel = past_action_vel                          # Past velocity action
        self.past_action_steering = past_action_steering                # Past steering action
        self.past_distance_to_ref_path = past_distance_to_ref_path      # Past distance to refrence path
        self.past_distance_to_boundaries = past_distance_to_boundaries  # Past distance to lanelet boundaries
        self.past_distance_to_left_boundary = past_distance_to_left_boundary  # Past distance to left lanelet boundary
        self.past_distance_to_right_boundary = past_distance_to_right_boundary  # Past distance to right lanelet boundary
        self.past_distance_to_agents = past_distance_to_agents  # Past mutual distance between agents
                
class Distances:
    def __init__(self, type = None, agents = None, left_boundaries = None, right_boundaries = None, boundaries = None, ref_paths = None, closest_point_on_ref_path = None, closest_point_on_left_b = None, closest_point_on_right_b = None, goal = None, obstacles = None):
        if (type is not None) & (type not in ['c2c', 'MTV']):
            raise ValueError("Invalid distance type. Must be 'c2c' or 'MTV'.")
        self.type = type                            # Distances between agents
        self.agents = agents                        # Distances between agents
        self.left_boundaries = left_boundaries      # Distances between agents and the left boundaries of their current lanelets (for each corner of each agent)
        self.right_boundaries = right_boundaries    # Distances between agents and the right boundaries of their current lanelets (for each corner of each agent)
        self.boundaries = boundaries                # The minimum distances between agents and the boundaries of their current lanelets
        self.ref_paths = ref_paths                  # Distances between agents and the center line of their current lanelets
        self.closest_point_on_ref_path = closest_point_on_ref_path  # Index of the closest point on reference path
        self.closest_point_on_left_b = closest_point_on_left_b      # Index of the closest point on left boundary
        self.closest_point_on_right_b = closest_point_on_right_b    # Index of the closest point on right boundary
        self.goal = goal                            # Distances to goal positions
        self.obstacles = obstacles                  # Distances to obstacles

class Evaluation:
    # This class stores the data relevant to evaluation of the system-wide performance, which necessitates the information being in the gloabl coordinate system.
    def __init__(self, pos_traj = None, v_traj = None, rot_traj = None, deviation_from_ref_path = None, path_tracking_error_mean = None):
        self.pos_traj = pos_traj    # Position trajectory
        self.v_traj = v_traj        # Velocity trajectory
        self.rot_traj = rot_traj    # Rotation trajectory
        self.deviation_from_ref_path = deviation_from_ref_path
        self.path_tracking_error_mean = path_tracking_error_mean # [relevant to path-tracking scenarios] Calculated when an agent reached its goal. The goal reward could be adjusted according to this variable

class Obstacles:
    # This class stores the data relevant to static and dynamic obstacles.
    def __init__(self, n = None, pos = None, corners = None, vel = None, rot = None, length = None, width = None, center_points = None, lengths = None, yaws = None):
        self.n = n              # The number of obstacles
        self.pos = pos          # Position
        self.corners = corners  # Corners
        self.vel = vel          # Velocity
        self.rot = rot          # Rotation
        self.length = length    # Length of the dynamic obstacle
        self.width = width      # Width of the dynamic obstacle

class Timer:
    # This class stores the data relevant to static and dynamic obstacles.
    def __init__(self, step = None, start = None, end = None, step_duration = None, step_begin = None, render_begin = None):
        self.step = step              # Count the current time step
        self.start = start              # Time point of simulation start
        self.end = end              # Time point of simulation end
        self.step_duration = step_duration # Duration of each time step
        self.step_begin = step_begin        # Time when the current time step begins
        self.render_begin = render_begin    # Time when the rendering of the current time step begins

class Collisions:
    def __init__(self, with_obstacles: torch.Tensor = None, with_agents: torch.Tensor = None, with_lanelets: torch.Tensor = None, with_entry_segments: torch.Tensor = None, with_exit_segments: torch.Tensor = None):
        self.with_agents = with_agents              # Whether collide with agents
        self.with_obstacles = with_obstacles        # Whether collide with obstacles
        self.with_lanelets = with_lanelets          # Whether collide with lanelet boundaries
        self.with_entry_segments = with_entry_segments # Whether collide with entry segments
        self.with_exit_segments = with_exit_segments   # Whether collide with exit segments
        
class Constants:
    # Predefined constants that may be used during simulations
    def __init__(self, env_idx_broadcasting: torch.Tensor = None, empty_actions: torch.Tensor = None, mask_pos: torch.Tensor = None, mask_vel: torch.Tensor = None, mask_rot: torch.Tensor = None, mask_zero: torch.Tensor = None, mask_one: torch.Tensor = None, reset_agent_min_distance: torch.Tensor = None, reset_scenario_probabilities: torch.Tensor = None):
        self.env_idx_broadcasting = env_idx_broadcasting
        self.empty_actions = empty_actions
        self.mask_pos = mask_pos
        self.mask_zero = mask_zero
        self.mask_one = mask_one
        self.mask_vel = mask_vel
        self.mask_rot = mask_rot
        self.reset_agent_min_distance = reset_agent_min_distance   # The minimum distance between agents when being reset
        self.reset_scenario_probabilities = reset_scenario_probabilities   # Reset the agents to different scenarios with different probability
        
class Prioritization:
    def __init__(self, values: torch.Tensor = None,):
        self.values = values

class CircularBuffer:
    def __init__(self, buffer_size: torch.Tensor = None, buffer: torch.Tensor = None,):
        """Initializes a circular buffer to store initial states. """
        self.buffer_size = buffer_size # Number of entries to be recorded
        self.buffer = buffer # Buffer
        self.pointer = 0  # Point to the index where the new recording should be stored
        self.valid_size = 0 # Valid size of the buffer, maximum being `buffer_size`

    def add(self, recording: torch.Tensor = None):
        """Adds a new recording to the buffer, overwriting the oldest recording if the buffer is full.

        Args:
            recording: A recording tensor to add to the buffer.
        """
        self.buffer[self.pointer] = recording
        self.pointer = (self.pointer + 1) % self.buffer_size # Increment, loop back to 0 if full
        self.valid_size = min(self.valid_size + 1, self.buffer_size) # Increment up to the maximum size

    def get_latest(self, n=1):
        """Returns the n-th latest recording from the buffer.

        Args:
            n: Specifies which latest recording to retrieve (1-based index: 1 is the most recent).

        Return:
            The n-th latest recording. If n is larger than valid_size, returns the first recording.
        """
        if n > self.valid_size:
            index = 0
        else:
            index = (self.pointer - n) % self.buffer_size

        return self.buffer[index]
    
    def reset(self):
        """Reset the buffer.
        """
        self.buffer[:] = 0
        self.pointer = 0
        self.valid_size = 0

class StateBuffer(CircularBuffer):
    def __init__(self, buffer_size: torch.Tensor = None, buffer: torch.Tensor = None):
        """ Initializes a circular buffer to store initial states. """
        super().__init__(buffer_size=buffer_size, buffer=buffer)  # Properly initialize the parent class
        self.idx_scenario = 5
        self.idx_path = 6
        self.idx_point = 7
        
class InitialStateBuffer(CircularBuffer):
    def __init__(self, buffer_size: torch.Tensor = None, buffer: torch.Tensor = None, probability_record: torch.Tensor = None, probability_use_recording: torch.Tensor = None,):
        """ Initializes a circular buffer to store initial states. """
        super().__init__(buffer_size=buffer_size, buffer=buffer)  # Properly initialize the parent class
        self.probability_record = probability_record
        self.probability_use_recording = probability_use_recording
        self.idx_scenario = 5
        self.idx_path = 6
        self.idx_point = 7

    def get_random(self):
        """ Returns a randomly selected recording from the buffer.

        Return: 
            A randomly selected recording tensor. If the buffer is empty, returns None.
        """
        if self.valid_size == 0:
            return None
        else:
            # Random index based on the current size of the buffer
            random_index = torch.randint(0, self.valid_size, ())

            return self.buffer[random_index]

class Noise:
    def __init__(self, vel: torch.Tensor = None, ref: torch.Tensor = None, dis_ref: torch.Tensor = None, dis_lanelets: torch.Tensor = None, other_agents_pri: torch.Tensor = None, other_agents_pos: torch.Tensor = None, other_agents_rot: torch.Tensor = None, other_agents_vel: torch.Tensor = None, other_agents_dis: torch.Tensor = None, level_vel: torch.Tensor = None, level_pos: torch.Tensor = None, level_rot: torch.Tensor = None, level_dis: torch.Tensor = None, level_pri: torch.Tensor = None):
        self.vel = vel
        self.ref = ref
        self.dis_ref = dis_ref
        self.dis_lanelets = dis_lanelets
        self.other_agents_pri = other_agents_pri
        self.other_agents_pos = other_agents_pos
        self.other_agents_rot = other_agents_rot
        self.other_agents_vel = other_agents_vel
        self.other_agents_dis = other_agents_dis
        self.level_vel = level_vel
        self.level_pos = level_pos
        self.level_rot = level_rot
        self.level_dis = level_dis
        self.level_pri = level_pri


##################################################
## Helper Functions
##################################################
def get_rectangle_corners(center: torch.Tensor, yaw, width, length, is_close_shape: bool = True):
    """ Compute the corners of rectangles for a batch of agents given their centers, yaws (rotations),
    widths, and lengths, using PyTorch tensors.

    Args:
        center: [batch_dim, 2] or [2] center positions of the rectangles. In the case of the latter, batch_dim is deemed to be 1.
        yaw: [batch_dim, 1] or [1] or [] Rotation angles in radians.
        width: [scalar] Width of the rectangles.
        length: [scalar] Length of the rectangles.
        
    Return: 
        [batch_dim, 4, 2] Corner points of the rectangles for each agent.
    """
    if center.ndim == 1:
        center = center.unsqueeze(0)
        
    if yaw.ndim == 0:
        yaw = yaw.unsqueeze(0).unsqueeze(0)
    elif yaw.ndim == 1:
        yaw = yaw.unsqueeze(0)
    
    batch_dim = center.shape[0]
        
    width_half = width / 2
    length_half = length / 2

    # Corner points relative to the center
    if is_close_shape:
        corners = torch.tensor([[length_half, width_half], [length_half, -width_half], [-length_half, -width_half], [-length_half, width_half], [length_half, width_half]], dtype=center.dtype, device=center.device) # Repeat the first vertex to close the shape
    else:
        corners = torch.tensor([[length_half, width_half], [length_half, -width_half], [-length_half, -width_half], [-length_half, width_half]], dtype=center.dtype, device=center.device)         

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
    corners_rotated = torch.matmul(rot_matrix, corners.transpose(1, 2)).transpose(1, 2)

    # Add center positions to the rotated corners
    corners_global = corners_rotated + center.unsqueeze(1)

    return corners_global

def get_perpendicular_distances(point: torch.Tensor, polyline: torch.Tensor, n_points_long_term: torch.Tensor = None):
    """ Calculate the minimum perpendicular distance from the given point(s) to the given polyline.
    
    Args:
        point: torch.Size([batch_size, 2]) or torch.Size([2]), position of the point. In the case of the latter, the batch_size is deemed to be 1.
        polyline: torch.Size([num_points, 2]) or torch.Size([batch_size, num_points, 2]) x- and y-coordinates of the points on the polyline.
    """
    
    if point.ndim == 1:
        point = point.unsqueeze(0)
    
    batch_size = point.shape[0]
    
    # Expand the polyline points to match the batch size        
    if polyline.ndim == 2:
        polyline_expanded = polyline.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, n_points, 2]
    else:
        polyline_expanded = polyline

    # Split the polyline into line segments
    line_starts = polyline_expanded[:, :-1, :]
    line_ends = polyline_expanded[:, 1:, :]

    # Create vectors for each line segment and for the point to the start of each segment
    point_expanded = point.unsqueeze(1)  # Shape: [batch_size, 1, 2]
    line_vecs = line_ends - line_starts
    point_vecs = point_expanded - line_starts

    # Project point_vecs onto line_vecs
    line_lens_squared = torch.sum(line_vecs ** 2, dim=2)
    projected_lengths = torch.sum(point_vecs * line_vecs, dim=2) / line_lens_squared

    # Clamp the projections to lie within the line segments
    clamped_lengths = torch.clamp(projected_lengths, 0, 1)

    # Find the closest points on the line segments to the given points
    closest_points = line_starts + (line_vecs * clamped_lengths.unsqueeze(2))

    # Calculate the distances from the given points to these closest points
    distances = torch.norm(closest_points - point_expanded, dim=2)
    
    if n_points_long_term is not None:
        if n_points_long_term.ndim == 0:
            n_points_long_term = n_points_long_term.unsqueeze(0)
        for env_idx, n_long_term_point in enumerate(n_points_long_term):
            distance_to_end_point = distances[env_idx, n_long_term_point-2]
            distances[env_idx, n_long_term_point-1:] = distance_to_end_point
    
    assert not distances.isnan().any()
    
    perpendicular_distances, indices_closest_points = torch.min(distances, dim=1)
    
    indices_closest_points[:] += 1 # Force the nearest point to lie always in the future

    return perpendicular_distances, indices_closest_points


def get_short_term_reference_path(polyline: torch.Tensor, index_closest_point: torch.Tensor, n_points_to_return: int, device = torch.device("cpu"), is_polyline_a_loop: torch.Tensor = False, n_points_long_term: torch.Tensor = None, sample_interval: int = 2, n_points_shift: int = 1):
    """
    
    Args:
        polyline:                   [batch_size, num_points, 2] or [num_points, 2]. In the case of the latter, batch_dim is deemed as 1.
        index_closest_point:        [batch_size, 1] or [1] or []. In the case of the latter, batch_dim is deemed as 1.
        n_points_to_return:         [1] or []. In the case of the latter, batch_dim is deemed as 1.
        is_polyline_a_loop:         [batch_size] or []. In the case of the latter, batch_dim is deemed as 1.
        n_points_long_term:         [batch_size] or []. In the case of the latter, batch_dim is deemed as 1.
        sample_interval:            Sample interval to match specific purposes;
                                    set to 2 when using this function to get the short-term reference path;
                                    set to 1 when using this function to get the nearing boundary points.
        n_points_shift:             Number of points to be shifted to match specific purposes;
                                    set to 1 when using this function to get the short-term reference path to "force" the first point of the short-term reference path being in front of the agent;
                                    set to -2 when using this function to get the nearing boundary points to consider the points behind the agent.
    """
    if polyline.ndim == 2:
        polyline = polyline.unsqueeze(0)
    if index_closest_point.ndim == 1:
        index_closest_point = index_closest_point.unsqueeze(1)
    elif index_closest_point.ndim == 0:
        index_closest_point = index_closest_point.unsqueeze(0).unsqueeze(1)
    if is_polyline_a_loop.ndim == 0:
        is_polyline_a_loop = is_polyline_a_loop.unsqueeze(0)
    if n_points_long_term.ndim == 0:
        n_points_long_term = n_points_long_term.unsqueeze(0)
        
    batch_size = index_closest_point.shape[0]        

    future_points_idx = torch.arange(n_points_to_return, device=device) * sample_interval + index_closest_point + n_points_shift

    if n_points_long_term is None:
        n_points_long_term = polyline.shape[-2]
    
    for env_i in range(batch_size):
        n_long_term_point = n_points_long_term[env_i]
        if is_polyline_a_loop[env_i]:
            # Apply modulo to handle the case that each agent's reference path is a loop
            future_points_idx[env_i] = torch.where(future_points_idx[env_i] >= n_long_term_point - 1, (future_points_idx[env_i] + 1) % n_long_term_point, future_points_idx[env_i]) # Use "+ 1" to skip the last point since it overlaps with the first point

    # Extract
    short_term_path = polyline[
        torch.arange(batch_size, device=device, dtype=torch.int).unsqueeze(1), # For broadcasting
        future_points_idx
    ] 
        
    return short_term_path, future_points_idx

def get_nearing_boundaries(left_boundary: torch.Tensor = None, right_boundary: torch.Tensor = None, closest_point_on_ref_path: torch.Tensor = None, n_points_nearing_boundary: torch.Tensor = None, device = torch.device("cpu"), is_ref_path_loop: torch.Tensor = False, n_points_long_term: torch.Tensor = None, sample_interval: int = 1,):
    """
    
    Args:
        left_boundary:              [batch_size, num_points, 2] or [num_points, 2]. In the case of the latter, batch_dim is deemed as 1.
        right_boundary:             [batch_size, num_points, 2] or [num_points, 2]. In the case of the latter, batch_dim is deemed as 1.
        closest_point_on_ref_path:  [batch_size, 1] or [1] or []. In the case of the latter, batch_dim is deemed as 1.
        n_points_nearing_boundary:        [1] or []. In the case of the latter, batch_dim is deemed as 1.
        is_ref_path_loop:           [batch_size] or []. In the case of the latter, batch_dim is deemed as 1.
        n_points_long_term:         [batch_size] or []. In the case of the latter, batch_dim is deemed as 1.
    """
    if left_boundary.ndim == 2:
        left_boundary = left_boundary.unsqueeze(0)
    if right_boundary.ndim == 2:
        right_boundary = right_boundary.unsqueeze(0)
    if closest_point_on_ref_path.ndim == 1:
        closest_point_on_ref_path = closest_point_on_ref_path.unsqueeze(1)
    elif closest_point_on_ref_path.ndim == 0:
        closest_point_on_ref_path = closest_point_on_ref_path.unsqueeze(0).unsqueeze(1)
    if is_ref_path_loop.ndim == 0:
        is_ref_path_loop = is_ref_path_loop.unsqueeze(0)
    if n_points_long_term.ndim == 0:
        n_points_long_term = n_points_long_term.unsqueeze(0)
        
    batch_size = closest_point_on_ref_path.shape[0]        

    # Create a tensor that represents the indices for n_points_nearing_boundary for each agent
    future_points_idx = torch.arange(n_points_nearing_boundary, device=device) * sample_interval + closest_point_on_ref_path - 2 # Add one to push the short-term reference path one point further

    if n_points_long_term is None:
        n_points_long_term = left_boundary.shape[-2]
    
    for env_i in range(batch_size):
        n_long_term_point = n_points_long_term[env_i]
        if is_ref_path_loop[env_i]:
            # Apply modulo to handle the case that each agent's reference path is a loop
            future_points_idx[env_i] = torch.where(future_points_idx[env_i] >= n_long_term_point - 1, (future_points_idx[env_i] + 1) % n_long_term_point, future_points_idx[env_i]) # Use "+ 1" to skip the last point since it overlaps with the first point

    # Extract the nearing points
    nearing_points_left_boundary = left_boundary[
        torch.arange(batch_size, device=device, dtype=torch.int).unsqueeze(1), # For broadcasting
        future_points_idx
    ]
    nearing_points_right_boundary = right_boundary[
        torch.arange(batch_size, device=device, dtype=torch.int).unsqueeze(1), # For broadcasting
        future_points_idx
    ]
        
    return nearing_points_left_boundary, nearing_points_right_boundary


def calculate_projected_movement(agent_pos_cur, agent_pos_next, line_segments):
    """
    Calculate the minimum perpendicular distance from the agent to line segments,
    and the projected movement of the agent along these line segments.
    """
    batch_size = agent_pos_cur.shape[0]
    num_segments = line_segments.shape[1] - 1

    # Expand line segments to match the batch size
    line_segments_expanded = line_segments.unsqueeze(0).expand(batch_size, -1, -1)

    # Split the line segments
    line_starts = line_segments_expanded[:, :-1, :]
    line_ends = line_segments_expanded[:, 1:, :]

    # Create vectors for each line segment
    line_vecs = line_ends - line_starts

    # Calculate agent movement vector
    agent_movement_vec = agent_pos_next - agent_pos_cur
    agent_movement_vec_expanded = agent_movement_vec.unsqueeze(1)

    # Project agent movement vector onto line segment vectors
    line_lens_squared = torch.sum(line_vecs ** 2, dim=2)
    projected_movement_lengths = torch.sum(agent_movement_vec_expanded * line_vecs, dim=2) / line_lens_squared

    # Expand agent positions for perpendicular distance calculation
    point_expanded = agent_pos_cur.unsqueeze(1)  # Shape: [batch_size, 1, 2]

    # Vectors from agent positions to the start of each line segment
    point_vecs = point_expanded - line_starts

    # Project point_vecs onto line_vecs
    projected_lengths = torch.sum(point_vecs * line_vecs, dim=2) / line_lens_squared

    # Clamp the projections to lie within the line segments
    clamped_lengths = torch.clamp(projected_lengths, 0, 1)

    # Find the closest points on the line segments to each agent position
    closest_points = line_starts + (line_vecs * clamped_lengths.unsqueeze(2))

    # Calculate the distances from each agent position to these closest points
    distances = torch.norm(closest_points - point_expanded, dim=2)

    # Return the minimum distance for each agent and projected movements
    return torch.min(distances, dim=1), projected_movement_lengths

def exponential_decreasing_fcn(x, x0, x1):
    """
    Exponential function y(x) = (e^( -(x-x0) / (x1-x0) ) - e^-1) / (1 - e^-1), so that y decreases exponentially from 1 to 0 when x increases from x0 to x1, where 
    x = max(min(x, x1), x0), 
    x1 = threshold_near_boundary, and 
    x0 = agent.shape.width/2.
    """
    x = torch.clamp(x, min=x0, max=x1) # x stays inside [x0, x1]
    y = (torch.exp(-(x-x0)/(x1-x0)) - 1/torch.e) / (1 - 1/torch.e)
    
    return y

def get_distances_between_agents(self, distance_type, is_set_diagonal = False):
    """ This function calculates the mutual distances between agents. 
        Currently, the calculation of two types of distances is supported ('c2c' and 'MTV'): 
            c2c: center-to-center distance
            MTV: minimum translation vector (MTV)-based distance
    Args:
        distance_type: one of {c2c, MTV}
        is_set_diagonal: whether to set the diagonal elements (distance from an agent to this agent itself) from zero to a high value 
    TODO: Add the posibility to calculate the mutual distances between agents in a single env (`reset_world` sometime only needs to resets a single env)
    """
    if distance_type == 'c2c':
        # Collect positions for all agents across all batches, shape [n_agents, batch_size, 2]
        positions = torch.stack([self.world.agents[i].state.pos for i in range(self.n_agents)])
        
        # Reshape from [n_agents, batch_size, 2] to [batch_size, n_agents, 2]
        positions_reshaped = positions.transpose(0, 1)

        # Reshape for broadcasting: shape becomes [batch_size, n_agents, 1, 2] and [batch_size, 1, n_agents, 2]
        pos1 = positions_reshaped.unsqueeze(2)
        pos2 = positions_reshaped.unsqueeze(1)

        # Calculate squared differences, shape [batch_size, n_agents, n_agents, 2]
        squared_diff = (pos1 - pos2) ** 2

        # Sum over the last dimension to get squared distances, shape [batch_size, n_agents, n_agents]
        squared_distances = squared_diff.sum(-1)

        # Take the square root to get actual distances, shape [batch_size, n_agents, n_agents]
        mutual_distances = torch.sqrt(squared_distances)
    elif distance_type == 'MTV':
        # Initialize
        mutual_distances = torch.zeros((self.world.batch_dim, self.n_agents, self.n_agents), device=self.world.device, dtype=torch.float32)
        
        # Calculate the normal axes of the four edges of each rectangle (Note that each rectangle has two normal axes)
        axes_all = torch.diff(self.corners[:,:,0:3,:], dim=2)
        axes_norm_all = axes_all / torch.norm(axes_all, dim=-1).unsqueeze(-1) # Normalize

        for i in range(self.n_agents):
            corners_i = self.corners[:,i,0:4,:]
            axes_norm_i = axes_norm_all[:,i]
            for j in range(i+1, self.n_agents):
                corners_j = self.corners[:,j,0:4,:]
                axes_norm_j = axes_norm_all[:,j]
                
                # 1. Project each of the four corners of rectangle i and all the four corners of rectangle j to each of the two axes of rectangle j. 
                # 2. The distance from a corner of rectangle i to rectangle j is calculated by taking the Euclidean distance of the "gaps" on the two axes of rectangle j between the projected point of this corner on the axes and the projected points of rectangle j. If the projected point of this corner lies inside the projection of rectangle j, the gap is consider zero.
                # 3. Steps 1 and 2 give us four distances. Repeat these two step for rectangle j, which give us another four distances.
                # 4. The MTV-based distance between the two rectangles is the smallest distance among the eight distances.
                        
                # Project rectangle j to its own axes
                projection_jj = (corners_j.unsqueeze(2) * axes_norm_j.unsqueeze(1)).sum(dim=3)
                max_jj, _ = torch.max(projection_jj, dim=1)
                min_jj, _ = torch.min(projection_jj, dim=1)
                # Project rectangle i to the axes of rectangle j
                projection_ij = (corners_i.unsqueeze(2) * axes_norm_j.unsqueeze(1)).sum(dim=3)
                max_ij, _ = torch.max(projection_ij, dim=1)
                min_ij, _ = torch.min(projection_ij, dim=1)
                
                MTVs_ij = (projection_ij - min_jj.unsqueeze(1))*(projection_ij <= min_jj.unsqueeze(1)) + (max_jj.unsqueeze(1) - projection_ij)*(projection_ij >= max_jj.unsqueeze(1))
                
                MTVs_ij_Euclidean = torch.norm(MTVs_ij, dim=2)
                
                # Project rectangle i to its own axes
                projection_ii = (corners_i.unsqueeze(2) * axes_norm_i.unsqueeze(1)).sum(dim=3)
                max_ii, _ = torch.max(projection_ii, dim=1)
                min_ii, _ = torch.min(projection_ii, dim=1)
                # Project rectangle j to the axes of rectangle i
                projection_ji = (corners_j.unsqueeze(2) * axes_norm_i.unsqueeze(1)).sum(dim=3)
                max_ji, _ = torch.max(projection_ji, dim=1)
                min_ji, _ = torch.min(projection_ji, dim=1)
                MTVs_ji = (projection_ji - min_ii.unsqueeze(1))*(projection_ji <= min_ii.unsqueeze(1)) + (max_ii.unsqueeze(1) - projection_ji)*(projection_ji >= max_ii.unsqueeze(1))
                MTVs_ij_Euclidean = torch.norm(MTVs_ji, dim=2)
                
                # The distance from rectangle j to rectangle i is calculated as the Euclidean distance of the two lengths of the MTVs on the two axes of rectangle i   
                distance_ji, _ = torch.min(torch.hstack((MTVs_ij_Euclidean, MTVs_ij_Euclidean)), dim=1)
                
                # Check rectangles are overlapping
                is_projection_overlapping = torch.hstack(
                    ((max_ii >= min_ji) & (min_ii <= max_ji), 
                    (max_jj >= min_ij) & (min_jj <= max_ij))
                )
                is_overlapping = torch.all(is_projection_overlapping, dim=1)
                distance_ji[is_overlapping] = 0 # Rectangles are overlapping

                mutual_distances[:,i,j] = distance_ji # The smaller one among the distance from rectangle i to j and the distance from rectangle j to i is define as the distance between two rectangles 
                mutual_distances[:,j,i] = distance_ji
                
    if is_set_diagonal:
        mutual_distances.diagonal(dim1=-2, dim2=-1).fill_(torch.sqrt(self.world.x_semidim ** 2 + self.world.y_semidim ** 2))
        
    return mutual_distances


def interX(L1, L2, is_return_points=False):
    """ Calculate the intersections of batches of curves. 
        Adapted from https://www.mathworks.com/matlabcentral/fileexchange/22441-curve-intersections
    Args:
        L1: [batch_size, num_points, 2]
        L2: [batch_size, num_points, 2]
        is_return_points: bool. Whether to return the intersecting points.
    """
    # L1[:,:,0] -= 0.35
    # L1[:,:,1] -= 0.05
    batch_dim = L1.shape[0]
    collision_index = torch.zeros(batch_dim, dtype=torch.bool) # Initialize

    # Handle empty inputs
    if L1.numel() == 0 or L2.numel() == 0:
        return torch.empty((0, 2), device=L1.device) if is_return_points else False

    # Extract x and y coordinates
    x1, y1 = L1[..., 0], L1[..., 1]
    x2, y2 = L2[..., 0], L2[..., 1]

    # Compute differences
    dx1, dy1 = torch.diff(x1, dim=1), torch.diff(y1, dim=1)
    dx2, dy2 = torch.diff(x2, dim=1), torch.diff(y2, dim=1)

    # Determine 'signed distances'
    S1 = dx1 * y1[..., :-1] - dy1 * x1[..., :-1]
    S2 = dx2 * y2[..., :-1] - dy2 * x2[..., :-1]

    # Helper function for computing D
    def D(x, y):
        return (x[..., :-1] - y) * (x[..., 1:] - y)

    C1 = D(dx1.unsqueeze(2) * y2.unsqueeze(1) - dy1.unsqueeze(2) * x2.unsqueeze(1), S1.unsqueeze(2)) < 0
    C2 = (D((y1.unsqueeze(2) * dx2.unsqueeze(1) - x1.unsqueeze(2) * dy2.unsqueeze(1)).transpose(1, 2), S2.unsqueeze(2)) < 0).transpose(1,2)

    # Obtain the segments where an intersection is expected
    batch_indices, i, j = torch.where(C1 & C2)
    batch_indices_pruned = torch.sort(torch.unique(batch_indices))[0]
    collision_index[batch_indices_pruned] = True

    if is_return_points:
        # In case of collisions, return collision points; else return empty points 
        if batch_indices.numel() == 0:
            return torch.empty((0, 2), device=L1.device)
        else:
            # Process intersections for each batch item
            intersections = []
            for b in batch_indices.unique():
                L = dy2[b, j] * dx1[b, i] - dy1[b, i] * dx2[b, j]
                nonzero = L != 0
                i_nz, j_nz, L_nz = i[nonzero], j[nonzero], L[nonzero]

                P = torch.stack(((dx2[b, j_nz] * S1[b, i_nz] - dx1[b, i_nz] * S2[b, j_nz]) / L_nz,
                                (dy2[b, j_nz] * S1[b, i_nz] - dy1[b, i_nz] * S2[b, j_nz]) / L_nz), dim=-1)
                intersections.append(P)
            # Combine intersections from all batches
            return torch.cat(intersections, dim=0)
    else:
        # Simply return whether collisions occur or not
        return collision_index

def get_point_line_distance(points: torch.Tensor, lines_start_points: torch.Tensor, lines_end_points: torch.Tensor):
    """
    Calculate the distance from multiple points (or a single point) to a line.

    Args:
        points: [batch_size, num_points, 2] representing the x- and y-coordinates of the points. Both `num_points` and `batch_size` could potentially be 1.
        lines_start_points: [batch_size, 2] representing the start points of the lines
        lines_end_points: [batch_size, 2] representing the end points of the lines

    Returns:
        torch.Tensor: [num_points] containing distances from the points to the line.
    """
    batch_size = max(points.shape[0], lines_start_points.shape[0])
    num_distances = max(points.shape[1], 1)
    distances = torch.zeros((batch_size, num_distances), device=points.device, dtype=torch.float32)
    
    # Match dimension
    lines_start_points = lines_start_points.unsqueeze(1)
    lines_end_points = lines_end_points.unsqueeze(1)

    # Compute the vectors
    line_vec = lines_end_points - lines_start_points
    points_vec = points - lines_start_points

    # Calculate the projection of points_vec onto line_vec
    line_len = (line_vec * line_vec).sum(dim=2)
    projected = (points_vec * line_vec).sum(dim=2) / line_len

    # Clamp the projection between 0 and 1 to find the nearest point on the line
    nearest = torch.clamp(projected, 0, 1)
    is_projection_inside_line = (projected >= 0) & (projected <= 1)

    # Find the nearest point on the line to the point
    nearest_point = lines_start_points + nearest.unsqueeze(2) * line_vec

    # Calculate the distance from the point to the nearest point on the line
    distances = (points - nearest_point).norm(dim=2)
    
    are_two_points_overlapping = ((lines_start_points - lines_end_points).norm(dim=2) == 0).squeeze(1)
    distances[are_two_points_overlapping] = (points - lines_start_points)[are_two_points_overlapping].norm(dim=2)

    return distances, is_projection_inside_line


def visualize_path(tracking_path, start_pos, goal_pos, start_rot, agent_width, agent_length, is_ref_path_loop: bool = False, is_save_fig: bool = False, path_save_fig: str = "fig.pdf", obstacles = None):
    plt.plot(tracking_path[:,0], tracking_path[:,1], color=Color.black100, linewidth=0.5)

    corners = get_rectangle_corners(
        center=start_pos,
        yaw=start_rot,
        width=agent_width,
        length=agent_length,
        is_close_shape=True
    )
    if corners.shape[0] == 1:
        corners = corners.squeeze(0)

    plt.fill(corners[:, 0], corners[:, 1], color=Color.blue100, linewidth=0.2, edgecolor='black') #  , alpha=0.5, 
    
    # Calculate the end point of the arrow based on the rotation angle
    dx = agent_length / 2 * torch.cos(start_rot)
    dy = agent_length / 2 * torch.sin(start_rot)
    
    # Draw an arrow from the center of the agent to the calculated end point
    plt.arrow(start_pos[0], start_pos[1], dx, dy, width=0.01, head_width=0.03, color=Color.black100)

    if not is_ref_path_loop:
        # Goal only exists if the reference path is not a loop
        plt.scatter(goal_pos[0], goal_pos[1], s=12, color=Color.red100, edgecolors="black", linewidths=0.2)

    # Visualize obstacles if any
    if obstacles is not None:
        if "dynamic" in path_save_fig:
            # Many dynamics obstacles
            max_steps = obstacles.shape[1]
            for i_obs in range(obstacles.shape[0]):
                plt.fill(obstacles[i_obs, -1, :, 0], obstacles[i_obs, -1, :, 1], color=Color.green100, linestyle="-")
        else:
            # One static obstacle
            plt.fill(obstacles[:, 0], obstacles[:, 1], color=Color.green100, linestyle="-")
        
    plt.axis("equal")
    plt.xlabel(r"$x$ [m]")
    plt.ylabel(r"$y$ [m]")
    
    if is_save_fig:
        plt.tight_layout(rect=[0, 0, 1, 1]) # left, bottom, right, top in normalized (0,1) figure coordinates
        plt.savefig(path_save_fig, format="pdf", bbox_inches="tight")
        print(f"An visualization of the path is saved under {path_save_fig}.")
        
    plt.show()
       

def remove_overlapping_points(polyline: torch.Tensor, threshold: float = 1e-4):
    remove = polyline.diff(dim=0).norm(dim=1) <= threshold
    remove = torch.hstack((remove, torch.zeros(1, dtype=torch.bool))) # Always keep the last point
    # Filter out overlapping points
    return polyline[~remove]

def generate_sine_path(start_pos, path_length_x, amplitude, num_points, device):
    # Generate linearly spaced x coordinates
    x_coords = torch.linspace(start_pos[0], start_pos[0] + path_length_x, num_points, device=device)
    # Generate sine y coordinates
    y_coords = start_pos[1] + amplitude * torch.sin(2 * torch.pi * (x_coords - start_pos[0]) / path_length_x)
    
    tracking_path = torch.stack((x_coords, y_coords), dim=1)
    return tracking_path

     

def get_ref_path_for_tracking_scenarios(path_tracking_type, agent_width: float = 0.1, agent_length: float = 0.2, point_interval: float = 0.1, max_speed: float = 0.5, device = torch.device("cpu"), center_point = None, is_visualize: bool = False, is_save_fig: bool = False, max_ref_path_points = None):
    if path_tracking_type == "line":
        first_point = torch.tensor([-1, 0], device=device, dtype=torch.float32)

        path_length = 3 # [m]
        
        last_point = first_point.clone()
        last_point[0] += path_length
        
        num_points = int(path_length / point_interval)  # Total number of points to discretize the reference path
        tracking_path = torch.stack(
            [torch.linspace(first_point[i], last_point[i], num_points, device=device, dtype=torch.float32) for i in range(2)], dim=1
        )
        
        start_rot = torch.tensor(45, device=device, dtype=torch.float32).deg2rad()
        goal_rot = torch.tensor(0, device=device, dtype=torch.float32).deg2rad()

    elif path_tracking_type == "turning":
        first_point = torch.tensor([-1, 0], device=device, dtype=torch.float32)

        horizontal_length = 3  # [m] Length of the horizontal part. Default 3 m
        vertical_length = 2  # [m] Length of the vertical part. Default 2 m
        num_points = int((horizontal_length + vertical_length) / point_interval)  # Total number of points to discretize the reference path

        # Number of points for each segment
        num_points_horizontal = int(num_points * horizontal_length / (horizontal_length + vertical_length))
        num_points_vertical = num_points - num_points_horizontal

        # Generate horizontal segment
        x_coords_horizontal = torch.linspace(first_point[0], first_point[0] + horizontal_length, num_points_horizontal, device=device)
        y_coords_horizontal = torch.full((num_points_horizontal,), first_point[1], device=device)

        # Generate vertical segment
        x_coords_vertical = torch.full((num_points_vertical,), first_point[0] + horizontal_length, device=device)
        y_coords_vertical = torch.linspace(first_point[1], first_point[1] + vertical_length, num_points_vertical, device=device)

        # Combine segments
        x_coords = torch.cat((x_coords_horizontal, x_coords_vertical))
        y_coords = torch.cat((y_coords_horizontal, y_coords_vertical))
        tracking_path = torch.stack((x_coords, y_coords), dim=1)

        start_rot = torch.tensor(0, device=device, dtype=torch.float32).deg2rad()
        goal_rot = torch.tensor(90, device=device, dtype=torch.float32).deg2rad()
        
    elif path_tracking_type == "circle":
        first_point = torch.tensor([-1, 0], device=device, dtype=torch.float32)
        
        circle_radius = 1.5 # [m] default: 1.5
        
        # Calculate the number of discrete points that consist of the whole path such that the interval between points is roughly `point_interval`
        path_length = 2 * torch.pi * circle_radius # [m]
        num_points = int(path_length / point_interval) 
        
        circle_origin = first_point.clone()
        circle_origin[0] += circle_radius
                
        # Generate angles for each point on the tracking path
        angles = torch.linspace(torch.pi, -torch.pi, num_points, device=device)

        # Calculate x and y coordinates for each point 
        x_coords = circle_origin[0] + circle_radius * torch.cos(angles)
        y_coords = circle_origin[1] + circle_radius * torch.sin(angles)

        tracking_path = torch.stack((x_coords, y_coords), dim=1)

        start_rot = torch.tensor(90, device=device, dtype=torch.float32).deg2rad()
        goal_rot = start_rot.clone()
    elif path_tracking_type == "sine":
        first_point = torch.tensor([-1, 0], device=device, dtype=torch.float32)

        path_length_x = 2.5  # [m] Length along x-axis. Default: 3
        num_points_tmp = 100  # Will be used to calculate the sine wave in a numerical way. Should be sufficient but not overly large
        amplitude = 0.8  # Amplitude of the sine wave. Default: 1

        # Calculate the actual needed number of discrete points
        tracking_path_tmp = generate_sine_path(first_point, path_length_x, amplitude, num_points_tmp, device)
        path_length = ((tracking_path_tmp.diff(dim=0) ** 2).sum(dim=1) ** 0.5).sum()
        num_points = int(path_length / point_interval) # Total number of points to discretize the reference path
        
        tracking_path = generate_sine_path(first_point, path_length_x, amplitude, num_points, device)

        start_rot = torch.tensor(90, device=device, dtype=torch.float32).deg2rad()
        goal_rot = torch.tensor(90, device=device, dtype=torch.float32).deg2rad()        
    elif path_tracking_type == "horizontal_8":
        # Use lemniscate of Bernoulli to generate a horizontal "8" path (inspired by https://mathworld.wolfram.com/Lemniscate.html)
        first_point = torch.tensor([-1, 0], device=device, dtype=torch.float32)

        center_point_8 = first_point.clone() # Center point of the lemniscate
        a = 1.5  # half-width of the lemniscate
        center_point_8[0] += a
        num_points = 100  # Number of points to discretize the reference path

        # Generate parameter t
        t = torch.linspace(-torch.pi, torch.pi, num_points, device=device)

        # Parametric equations for the lemniscate
        x_coords = first_point[0] + (a * torch.cos(t)) / (1 + torch.sin(t)**2)
        y_coords = first_point[1] + (a * torch.sin(t) * torch.cos(t)) / (1 + torch.sin(t)**2)

        # Combine x and y coordinates
        tracking_path = torch.stack((x_coords, y_coords), dim=1)

        start_rot = torch.tensor(90, device=device, dtype=torch.float32).deg2rad()
        goal_rot = torch.tensor(90, device=device, dtype=torch.float32).deg2rad()

    else:
        raise ValueError("Invalid path tracking type provided. Must be one of 'line', 'turning', 'circle', 'sine', and 'turning'.")
    
    tracking_path = remove_overlapping_points(tracking_path)
    num_points = tracking_path.shape[0] # Update
    # Initial velocity is set as the haf of the maximum velocity
    start_vel = torch.tensor([0.5*max_speed*torch.cos(start_rot), 0.5*max_speed*torch.sin(start_rot)], device=device, dtype=torch.float32) 
    
    # Mean length of the line segments on the path
    mean_length_line_segments = tracking_path.diff(dim=0).norm(dim=1).mean()
    # print(f"The mean length of the line segments of the tracking path is {mean_length_line_segments}.")

    # Check is the reference path is a loop
    is_ref_path_loop = (tracking_path[0, :] - tracking_path[-1, :]).norm() <= 1e-4

    # Determine the x- and y-range of the reference path, which will later be used to determine the x- and y-dimensions of the world
    x_min = torch.min(tracking_path[:, 0])
    x_max = torch.max(tracking_path[:, 0])
    y_min = torch.min(tracking_path[:, 1])
    y_max = torch.max(tracking_path[:, 1])
    ranges = torch.hstack((x_max - x_min, y_max - y_min))
    center_point_path = torch.hstack(
        (
            (x_min + x_max) / 2,
            (y_min + y_max) / 2
        )
    )
    
    tracking_path -= center_point_path # Move the path center to the origin
    
    if center_point is not None:
        tracking_path += center_point # Move the path center to the given center point
        
    start_pos = tracking_path[0, :].clone()
    goal_pos = tracking_path[-1, :].clone()

    # Extend an additional point (with the same direction) at the end of the path to workaround the phenomenon that the agent oscilltes near the final goal. Only used in non-loop path-tracking scenarios.
    point_extended = 2 * tracking_path[-1, :] - tracking_path[-2, :]

        
    if is_visualize:
        is_save_fig = True
        path_save_fig = "outputs/path_tracking_" + path_tracking_type + ".pdf"
        visualize_path(tracking_path, start_pos, goal_pos, start_rot, agent_width, agent_length, is_ref_path_loop, is_save_fig, path_save_fig)
        
    if max_ref_path_points is not None:
        # Repeated the last point
        assert max_ref_path_points > num_points, "The number of points on the reference path should be smaller than the defined maximum number."
        last_point = tracking_path[-1, :].unsqueeze(0).repeat(max_ref_path_points - num_points, 1)
        tracking_path_last_point_repeated = torch.cat((tracking_path, last_point), dim=0)
    
    return tracking_path_last_point_repeated, num_points, ranges, start_pos, start_rot, start_vel, goal_pos, goal_rot, is_ref_path_loop, point_extended

def get_ref_path_for_obstacle_avoidance_scenarios(agent_width: float = 0.1, agent_length: float = 0.2, point_interval: float = 0.1, max_speed: float = 0.5, device = torch.device("cpu"), center_point = True, is_visualize: bool = False, is_save_fig: bool = False, obstacles = None):
    
    first_point = torch.tensor([-1, 0], device=device, dtype=torch.float32)

    path_length = 3 # [m]
    
    last_point = first_point.clone()
    last_point[0] += path_length
    
    num_points = int(path_length / point_interval)  # Total number of points to discretize the reference path
    tracking_path = torch.stack(
        [torch.linspace(first_point[i], last_point[i], num_points, device=device, dtype=torch.float32) for i in range(2)], dim=1
    )
    
    start_rot = torch.tensor(0, device=device, dtype=torch.float32).deg2rad()
    goal_rot = torch.tensor(0, device=device, dtype=torch.float32).deg2rad()
    
    tracking_path = remove_overlapping_points(tracking_path)
    # Initial velocity is set as the haf of the maximum velocity
    start_vel = torch.tensor([0.5*max_speed*torch.cos(start_rot), 0.5*max_speed*torch.sin(start_rot)], device=device, dtype=torch.float32) 
    
    # Mean length of the line segments on the path
    mean_length_line_segments = tracking_path.diff(dim=0).norm(dim=1).mean()
    # print(f"The mean length of the line segments of the tracking path is {mean_length_line_segments}.")

    # Check is the reference path is a loop
    is_ref_path_loop = (tracking_path[0, :] - tracking_path[-1, :]).norm() <= 1e-4
        
    # Determine the x- and y-range of the reference path, which will later be used to determine the x- and y-dimensions of the world
    x_min = torch.min(tracking_path[:, 0])
    x_max = torch.max(tracking_path[:, 0])
    y_min = torch.min(tracking_path[:, 1])
    y_max = torch.max(tracking_path[:, 1])
    ranges = torch.hstack((x_max - x_min, y_max - y_min))
    center_point_path = torch.hstack(
        (
            (x_min + x_max) / 2,
            (y_min + y_max) / 2
        )
    )

    tracking_path -= center_point_path # Move the path center to the origin
    
    if center_point is not None:
        tracking_path += center_point # Move the path center to the given center point
        
    start_pos = tracking_path[0, :].clone()
    goal_pos = tracking_path[-1, :].clone()

    if not is_ref_path_loop:
        # Extend an additional point (with the same direction) at the end of the path to workaround the phenomenon that the agent oscilltes near the final goal
        point_extended = 2 * tracking_path[-1, :] - tracking_path[-2, :]
    else:
        point_extended = None
        
    if is_visualize:
        is_save_fig = True
        if obstacles.ndim == 4:
            path_save_fig = "outputs/obstacle_avoidance_dynamics.pdf"
        else:
            path_save_fig = "outputs/obstacle_avoidance_static.pdf"
        visualize_path(tracking_path, start_pos, goal_pos, start_rot, agent_width, agent_length, is_ref_path_loop, is_save_fig, path_save_fig, obstacles=obstacles)
    
    return tracking_path, ranges, start_pos, start_rot, start_vel, goal_pos, goal_rot, is_ref_path_loop, point_extended


def transform_from_global_to_local_coordinate(pos_i: torch.Tensor, pos_j: torch.Tensor, rot_i):
    """
    Args:
        pos_i: torch.Size([batch_size, 2])
        pos_j: torch.Size([batch_size, num_points, 2]) or torch.Size([num_points, 2])
        rot_i: torch.Size([batch_size, 1])
    """
    # Prepare for vectorized ccomputation
    if pos_j.ndim == 3:
        pos_i_extended = pos_i.unsqueeze(1)
    else:
        pos_i_extended = pos_i.unsqueeze(1)
        # Check if the last point overlaps with the first point
        if (pos_j[0, :] - pos_j[-1, :]).norm() == 0:
            pos_j = pos_j[0:-1, :].unsqueeze(0)
        else:
            pos_j = pos_j.unsqueeze(0)
                        
    pos_vec = pos_j - pos_i_extended
    pos_vec_abs = pos_vec.norm(dim=2)
    rot_rel = torch.atan2(pos_vec[:, :, 1], pos_vec[:, :, 0]) - rot_i
    
    pos_rel = torch.stack(
        (
            torch.cos(rot_rel) * pos_vec_abs,
            torch.sin(rot_rel) * pos_vec_abs,
        ), dim=2
    )
    
    return pos_rel


def angle_eliminate_two_pi(angle):
    """
    Normalize an angle to be within the range -pi to pi.

    Args:
        angle (torch.Tensor): The angle to normalize in radians. Can be a tensor of any shape.

    Returns:
        torch.Tensor: Normalized angle between -pi and pi.
    """
    two_pi = 2 * torch.pi
    angle = angle % two_pi  # Normalize angle to be within 0 and 2*pi
    angle[angle > torch.pi] -= two_pi  # Shift to -pi to pi range
    return angle