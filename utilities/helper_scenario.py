import torch


##################################################
## Custom Classes
##################################################
class Normalizers:
    def __init__(self, pos, v, rot):
        self.pos = pos
        self.v = v
        self.rot = rot

class Rewards:
    def __init__(self, progress = None, weighting_ref_directions = None, higth_v = None, reaching_goal = None):
        self.progress = progress
        self.weighting_ref_directions = weighting_ref_directions
        self.higth_v = higth_v
        self.reaching_goal = reaching_goal

class Penalties:
    def __init__(self, deviate_from_ref_path = None, deviate_from_goal = None, weighting_deviate_from_ref_path = None, near_boundary = None, near_other_agents = None, collide_with_agents = None, collide_with_boundaries = None, time = None):
        self.deviate_from_ref_path = deviate_from_ref_path  # Penalty for deviating from reference path
        self.deviate_from_goal = deviate_from_goal          # Penalty for deviating from goal position 
        self.weighting_deviate_from_ref_path = weighting_deviate_from_ref_path
        self.near_boundary = near_boundary                  # Penalty for being too close to lanelet boundaries
        self.near_other_agents = near_other_agents          # Penalty for being too close to other agents
        self.collide_with_agents = collide_with_agents      # Penalty for colliding with other agents
        self.collide_with_boundaries = collide_with_boundaries  # Penalty for colliding with lanelet boundaries
        self.time = time                                    # Penalty for losing time
        
class Thresholds:
    def __init__(self, deviate_from_ref_path = None, near_boundary_low = None, near_boundary_high = None, near_other_agents_low = None, near_other_agents_high = None, reaching_goal = None):
        self.deviate_from_ref_path = deviate_from_ref_path
        self.near_boundary_low = near_boundary_low
        self.near_boundary_high = near_boundary_high
        self.near_other_agents_low = near_other_agents_low
        self.near_other_agents_high = near_other_agents_high
        self.reaching_goal = reaching_goal                      # Threshold less than which agents are considered at their goal positions

class ReferencePaths:
    def __init__(self, long_term = None, long_term_yaws = None, long_term_center_points = None, long_term_lengths = None, long_term_vecs_normalized = None, n_short_term_points = None, short_term = None, short_term_indices = None, left_boundary_repeated = None, right_boundary_repeated = None):
        self.long_term = long_term                              # Long-term reference path
        self.long_term_yaws = long_term_yaws                    # Yaws of the line segments on the long-term reference path
        self.long_term_center_points = long_term_center_points  # Center points of the line segments on the long-term reference path
        self.long_term_lengths = long_term_lengths              # Lengths of the line segments on the long-term reference path
        self.long_term_vecs_normalized = long_term_vecs_normalized  # Normalized vectors of the line segments on the long-term reference path
        self.n_short_term_points = n_short_term_points          # Number of points used to build a short-term reference path
        self.short_term = short_term                            # Short-term reference path
        self.short_term_indices = short_term_indices            # Indices that indicate which part of the long-term reference path is used to build the short-term reference path
        self.left_boundary_repeated = left_boundary_repeated    # Just to allocate memory for a specific purpose 
        self.right_boundary_repeated = right_boundary_repeated  # Just to allocate memory for a specific purpose 
        
class Observations:
    def __init__(self, is_local = None, is_global_coordinate_sys = None, n_nearing_agents = None):
        self.is_local = is_local # Local observation
        self.is_global_coordinate_sys = is_global_coordinate_sys
        self.n_nearing_agents = n_nearing_agents
        
class Distances:
    def __init__(self, type = None, agents = None, left_boundaries = None, right_boundaries = None, ref_paths = None, closest_point_on_ref_path = None, goal = None):
        if (type is not None) & (type not in ['c2c', 'MTV']):
            raise ValueError("Invalid distance type. Must be 'c2c' or 'MTV'.")
        self.type = type                            # Distances between agents
        self.agents = agents                        # Distances between agents
        self.left_boundaries = left_boundaries      # Distances between agents and the left boundaries of their current lanelets
        self.right_boundaries = right_boundaries    # Distances between agents and the right boundaries of their current lanelets
        self.ref_paths = ref_paths                  # Distances between agents and the center line of their current lanelets
        self.closest_point_on_ref_path = closest_point_on_ref_path
        self.goal = goal                            # Distances to goal positions

class Evaluation:
    # This class stores the data relevant to evaluation of the system-wide performance, which necessitates the information being in the gloabl coordinate system.
    def __init__(self, pos_traj = None, v_traj = None, rot_traj = None, deviation_from_ref_path = None, path_tracking_error_mean = None):
        self.pos_traj = pos_traj    # Position trajectory
        self.v_traj = v_traj        # Velocity trajectory
        self.rot_traj = rot_traj    # Rotation trajectory
        self.deviation_from_ref_path = deviation_from_ref_path
        self.path_tracking_error_mean = path_tracking_error_mean # [relevant to path-tracking scenarios] Calculated when an agent reached its goal. The goal reward could be adjusted according to this variable

##################################################
## Helper Functions
##################################################
def get_rectangle_corners(center, yaw, width, length, is_close_shape: bool = True):
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

def find_short_term_trajectory(pos, reference_path, n_short_term_points=6):
    n_points = reference_path.shape[0]

    # Expand dimensions for vectorized computation of distances
    # pos shape becomes [n_agents, 1, 2] and reference_path shape becomes [1, n_points, 2]
    expanded_pos = pos.unsqueeze(1)
    expanded_ref_traj = reference_path.unsqueeze(0)

    # Compute squared distances (broadcasting is used here)
    distances = torch.sum((expanded_pos - expanded_ref_traj) ** 2, dim=2)

    # Find the indices of the closest points
    closest_indices = torch.argmin(distances, dim=1)

    # Create a range of indices for the next n_short_term_points points
    future_idx_range = torch.arange(0, n_short_term_points)
    future_indices = (closest_indices.unsqueeze(1) + future_idx_range.unsqueeze(0)) % n_points

    # Gather the short-term trajectory points for each agent
    short_term_path = torch.gather(reference_path.expand(pos.shape[0], -1, -1), 1, future_indices.unsqueeze(2).expand(-1, -1, 2))

    return short_term_path

def get_perpendicular_distances(point, boundary):
    """
    Calculate the minimum perpendicular distance from the given point to the given boundary.
    """
    
    # Expand the boundary points to match the batch size
    batch_size = point.shape[0]
    boundary_expanded = boundary.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: [batch_size, n_points, 2]

    # Split the boundary into line segments
    line_starts = boundary_expanded[:, :-1, :]
    line_ends = boundary_expanded[:, 1:, :]

    # Create vectors for each line segment and for the point to the start of each segment
    agent_positions_expanded = point.unsqueeze(1)  # Shape: [batch_size, 1, 2]
    line_vecs = line_ends - line_starts
    point_vecs = agent_positions_expanded - line_starts

    # Project point_vecs onto line_vecs
    line_lens_squared = torch.sum(line_vecs ** 2, dim=2)
    projected_lengths = torch.sum(point_vecs * line_vecs, dim=2) / line_lens_squared

    # Clamp the projections to lie within the line segments
    clamped_lengths = torch.clamp(projected_lengths, 0, 1)

    # Find the closest points on the line segments to the given points
    closest_points = line_starts + (line_vecs * clamped_lengths.unsqueeze(2))

    # Calculate the distances from the given points to these closest points
    distances = torch.norm(closest_points - agent_positions_expanded, dim=2)
    
    perpendicular_distances, indices_closest_points = torch.min(distances, dim=1)
    
    return perpendicular_distances, indices_closest_points

def get_short_term_reference_path(reference_path, closest_point_on_ref_path, n_short_term_points, device = torch.device("cpu")):
    # Create a tensor that represents the indices for n_short_term_points for each agent
    future_points_idx_tmp = torch.arange(n_short_term_points, device=device).unsqueeze(0) + closest_point_on_ref_path.unsqueeze(1)
    
    len_reference_path = len(reference_path)
    
    # Check if the reference path is a loop
    if torch.allclose(reference_path[0, :], reference_path[-1, :], rtol=1e-4):
        # Apply modulo to handle the fact that each agent's reference path is a loop
        future_points_idx = torch.where(future_points_idx_tmp >= len_reference_path - 1, (future_points_idx_tmp + 1) % len_reference_path, future_points_idx_tmp) # Use "+ 1" to skip the last point since it overlaps with the first point
    else:
        future_points_idx = torch.where(future_points_idx_tmp >= len_reference_path - 1, len_reference_path - 1, future_points_idx_tmp) # Set all the remaining points to the last point 

    # Extract the short-term reference path from the reference path
    short_term_path = reference_path[future_points_idx] # Note that the agent's current position is between the first and second points (may overlap with the second point)
    
    return short_term_path, future_points_idx

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
    agent_positions_expanded = agent_pos_cur.unsqueeze(1)  # Shape: [batch_size, 1, 2]

    # Vectors from agent positions to the start of each line segment
    point_vecs = agent_positions_expanded - line_starts

    # Project point_vecs onto line_vecs
    projected_lengths = torch.sum(point_vecs * line_vecs, dim=2) / line_lens_squared

    # Clamp the projections to lie within the line segments
    clamped_lengths = torch.clamp(projected_lengths, 0, 1)

    # Find the closest points on the line segments to each agent position
    closest_points = line_starts + (line_vecs * clamped_lengths.unsqueeze(2))

    # Calculate the distances from each agent position to these closest points
    distances = torch.norm(closest_points - agent_positions_expanded, dim=2)

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

def get_distances_between_agents(self):
    """
    This function calculates the mutual distances between agents. 
    Currently, the calculation of two types of distances is supported ('c2c' and 'MTV'): 
        c2c: center-to-center distance
        MTV: minimum translation vector (MTV)-based distance
    TODO: Add the posibility to calculate the mutual distances between agents in a single env (`reset_world` sometime only needs to resets a single env)
    """
    if self.distances.type == 'c2c':
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
    elif self.distances.type == 'MTV':
        # Initialize
        mutual_distances = torch.zeros((self.world.batch_dim, self.n_agents, self.n_agents), device=self.world.device, dtype=torch.float32)
        
        # Calculate the normal axes of the four edges of each rectangle (Note that each rectangle has two normal axes)
        axes_all = torch.diff(self.corners_gloabl[:,:,0:3,:], dim=2)
        axes_norm_all = axes_all / torch.norm(axes_all, dim=-1).unsqueeze(-1) # Normalize

        for i in range(self.n_agents):
            corners_i = self.corners_gloabl[:,i,0:4,:]
            axes_norm_i = axes_norm_all[:,i]
            for j in range(i+1, self.n_agents):
                corners_j = self.corners_gloabl[:,j,0:4,:]
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
    return mutual_distances


def interX(L1, L2, is_return_points=False):
    """
    Calculate the intersections of batches of curves using PyTorch tensors.
    Each curve in the batches should be a tensor of shape (batch_size, points, 2), 
    where points is the number of points in the curve.
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
