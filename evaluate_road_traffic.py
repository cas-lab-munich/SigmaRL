from datetime import datetime
import torch
import numpy as np
from termcolor import colored, cprint

from tensordict import TensorDict

from utilities.colors import Color

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Scientific plotting
import scienceplots # Do not remove (https://github.com/garrettj403/SciencePlots)
plt.rcParams.update({'figure.dpi': '100'}) # Avoid DPI problem (https://github.com/garrettj403/SciencePlots/issues/60)
plt.style.use(['science','ieee']) # The science + ieee styles for IEEE papers (can also be one of 'ieee' and 'science' )
# print(plt.style.available) # List all available style

import time
import json

import os
import sys
# # !Important: Add project root to system path
# current_dir = os.path.dirname(os.path.realpath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, '..'))
# sys.path.append(project_root)

from mappo_cavs import mappo_cavs
from utilities.helper_training import Parameters, SaveData, find_the_highest_reward_among_all_models, get_model_name

colors = [
    Color.blue100, Color.purple100, Color.violet100, Color.bordeaux100, Color.red100, Color.orange100, Color.maygreen100, Color.green100, Color.turquoise100, Color.petrol100, Color.yellow100, Color.magenta100, Color.black100,
    Color.blue50, Color.purple50, Color.violet50, Color.bordeaux50, Color.red50, Color.orange50, Color.maygreen50, Color.green50, Color.turquoise50, Color.petrol50, Color.yellow50, Color.magenta50, Color.black50,
] # Each agent will get a different color

def evaluate_outputs():

    positions = out_td["agents","info","pos"]
    velocities = out_td["agents","info","vel"]
    is_collision_with_agents = out_td["agents","info","is_collision_with_agents"].bool()
    is_collision_with_lanelets = out_td["agents","info","is_collision_with_lanelets"].bool()
    distance_ref = out_td["agents","info","distance_ref"]
    
    is_collide = is_collision_with_agents | is_collision_with_lanelets
    
    num_steps = positions.shape[1]

    velocity_average[i_model, :] = velocities.norm(dim=-1).mean(dim=(-2, -1))
    collision_rate_with_agents[i_model, :] = is_collision_with_agents.squeeze(-1).any(dim=-1).sum(dim=-1) / num_steps
    collision_rate_with_lanelets[i_model, :] = is_collision_with_lanelets.squeeze(-1).any(dim=-1).sum(dim=-1) / num_steps
    distance_ref_average[i_model, :] = distance_ref.squeeze(-1).mean(dim=(-2, -1))

    path_eval_out_td = parameters.where_to_save + parameters.mode_name + "_out_td.pth"
    if parameters.is_save_eval_results:
        # Save the input TensorDict
        torch.save(out_td, path_eval_out_td)

def remove_max_min_per_row(tensor):
    """
    Remove the maximum and minimum values from each row of the tensor.
    
    Args:
        tensor: A 2D tensor with shape [a, b]
    
    Returns:
        A 2D tensor with the max and min values removed from each row.
    """
    # Find the indices of the max and min in each row
    max_vals, max_indices = torch.max(tensor, dim=1, keepdim=True)
    min_vals, min_indices = torch.min(tensor, dim=1, keepdim=True)
    
    # Deal with the case in which one whole row has the same value
    is_row_same_value = max_indices == min_indices
    is_row_different_values = max_indices != min_indices
    
    row_same_value, _ = torch.where(is_row_same_value)
    row_different_values, _ = torch.where(is_row_different_values)
    
    # Replace max and min values with inf and -inf
    tensor[row_same_value.unsqueeze(-1), 0] = float('inf')
    tensor[row_same_value.unsqueeze(-1), 1] = float('-inf')

    tensor[row_different_values.unsqueeze(-1), max_indices[is_row_different_values].unsqueeze(-1)] = float('inf')
    tensor[row_different_values.unsqueeze(-1), min_indices[is_row_different_values].unsqueeze(-1)] = float('-inf')

    # Remove the inf and -inf values
    mask = (tensor != float('inf')) & (tensor != float('-inf'))
    filtered_tensor = tensor[mask].view(tensor.size(0), -1)
    
    return filtered_tensor

# Function to add custom median markers
def custom_violinplot_color(parts, color_face, color_lines, alpha):
    for pc in parts['bodies']:
        pc.set_facecolor(color_face)
        pc.set_edgecolor(color_face)
        pc.set_alpha(alpha)

    parts["cmedians"].set_colors(color_lines)
    parts["cmedians"].set_alpha(alpha)
    parts["cmaxes"].set_colors(color_lines)
    parts["cmaxes"].set_alpha(alpha)
    parts["cmins"].set_colors(color_lines)
    parts["cmins"].set_alpha(alpha)
    parts["cbars"].set_colors(color_lines)
    parts["cbars"].set_alpha(alpha)


def smooth_data(data, window_size=5):
    """
    Smooths the data using a simple moving average.
    
    Args:
        data: The input data to smooth.
        window_size (int): The size of the smoothing window.
        
    Returns:
        numpy.ndarray: The smoothed data.
    """
    smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    return np.concatenate((data[:window_size-1], smoothed))


scenario_name = "road_traffic" # road_traffic, path_tracking, obstacle_avoidance

parameters = Parameters(
    n_agents=16,
    dt=0.05, # [s] sample time 
    device="cpu" if not torch.cuda.is_available() else "cuda:0",  # The divice where learning is run
    scenario_name=scenario_name,
    
    # Training parameters
    n_iters=200, # Number of sampling and training iterations (on-policy: rollouts are collected during sampling phase, which will be immediately used in the training phase of the same iteration),
    frames_per_batch=1200*8, # Number of team frames collected per training iteration 
                            # num_envs = frames_per_batch / max_steps
                            # total_frames = frames_per_batch * n_iters
                            # sub_batch_size = frames_per_batch // minibatch_size    num_epochs=30, # Number of optimization steps per training iteration,
    minibatch_size=2*9, # Size of the mini-batches in each optimization step (2**9 - 2**12?),
    lr=2e-4, # Learning rate,
    max_grad_norm=1.0, # Maximum norm for the gradients,
    clip_epsilon=0.2, # clip value for PPO loss,
    gamma=0.99, # discount factor (empirical formula: 0.1 = gamma^t, where t is the number of future steps that you want your agents to predict {0.96 -> 56 steps, 0.98 -> 114 steps, 0.99 -> 229 steps, 0.995 -> 459 steps})
    lmbda=0.9, # lambda for generalised advantage estimation,
    entropy_eps=1e-4, # coefficient of the entropy term in the PPO loss,
    max_steps=1200, # Episode steps before done
    training_strategy='4', # One of {'1', '2', '3', '4'}
    
    is_save_intermidiate_model=True, # Is this is true, the model with the highest mean episode reward will be saved,
    
    episode_reward_mean_current=0.00,
    
    is_load_model=True, # Load offline model if available. The offline model in `where_to_save` whose name contains `episode_reward_mean_current` will be loaded
    is_load_final_model=False,

    is_continue_train=False, # If offline models are loaded, whether to continue to train the model
    mode_name=None, 
    episode_reward_intermidiate=-1e3, # The initial value should be samll enough
    
    where_to_save="", # folder where to save the trained models, fig, data, etc.

    # Scenario parameters
    is_partial_observation=True, 
    is_ego_view=True,
    n_points_short_term=3,
    is_use_intermediate_goals=False,
    n_nearing_agents_observed=2,
    n_nearing_obstacles_observed=4,
    
    is_testing_mode=True,
    is_visualize_short_term_path=True,
    
    is_save_eval_results=True,
    
    is_observe_distance_to_boundaries=True,
    is_apply_mask=True,
    is_use_mtv_distance=True,
    
    # Visualization
    is_real_time_rendering=False,
    
    # Evaluation
    is_load_out_td=False,
)

model_paths = [
    "outputs/road_traffic_ppo/bird-view/",
    "outputs/road_traffic_ppo/no mask/",
    "outputs/road_traffic_ppo/obs. corners of surr. ag./",
    "outputs/road_traffic_ppo/not obs. dist. to surr. ag./",
    "outputs/road_traffic_ppo/obs. boundary points/",
    "outputs/road_traffic_ppo/not obs. dist. to center line/",
    "outputs/road_traffic_ppo/our/",
    "outputs/road_traffic_ppo/vanilla/",
    "outputs/road_traffic_ppo/chall. initial state buffer/",
    "outputs/road_traffic_ppo/PER/",
]

num_models = len(model_paths)

labels = [m.split('/')[-2] for m in model_paths]
labels_short = [f"M{idx + 1}" for idx in range(num_models)]

idx_our_model = next((i for i, s in enumerate(model_paths) if "our" in s), None)
labels_short[idx_our_model] += " (our)" 

labels_with_numbers = [labels_short[idx] + " (" + l + ")" for idx, l in enumerate(labels)]
labels_with_numbers[idx_our_model] = labels_with_numbers[idx_our_model][0:8]

fontsize = 8

for i_model in range(num_models):
    print("------------------------------------------")
    print(colored("-- [INFO] Model ", "black"), colored(f"{i_model + 1}", "blue"), colored(f"({labels[i_model]})", color="grey"))
    print("------------------------------------------")
    
    model_path = model_paths[i_model]
    
    # Load parameters
    try:
        path_to_json_file = next(os.path.join(model_path, file) for file in os.listdir(model_path) if file.endswith('.json')) # Find the first json file in the folder
        # Load parameters from the saved json file
        with open(path_to_json_file, 'r') as file:
            data = json.load(file)
            saved_data = SaveData.from_dict(data)
            parameters = saved_data.parameters
            
            parameters.is_testing_mode = True
            parameters.is_real_time_rendering = False
            
            parameters.is_save_eval_results = True
            parameters.is_load_model = True
            parameters.is_load_final_model = False
            parameters.is_load_out_td  = True
            
            parameters.n_agents = 12
            parameters.max_steps = 1200 # 1200 -> 1 min
            parameters.num_vmas_envs = 32
            parameters.frames_per_batch = parameters.max_steps * parameters.num_vmas_envs
            parameters.training_strategy = "1"
    except StopIteration:
        print(colored("No json file found. Manually defined parameters will be used.", "red"))
    
    if i_model == 0:
        # Initialize
        velocity_average = torch.zeros((num_models, parameters.num_vmas_envs), device=parameters.device, dtype=torch.float32)
        # collision_rate_sum = torch.zeros((num_models, parameters.num_vmas_envs), device=parameters.device, dtype=torch.float32)
        collision_rate_with_agents = torch.zeros((num_models, parameters.num_vmas_envs), device=parameters.device, dtype=torch.float32)
        collision_rate_with_lanelets = torch.zeros((num_models, parameters.num_vmas_envs), device=parameters.device, dtype=torch.float32)
        distance_ref_average = torch.zeros((num_models, parameters.num_vmas_envs), device=parameters.device, dtype=torch.float32)
        episode_reward = torch.zeros((num_models, parameters.n_iters), device=parameters.device, dtype=torch.float32)
        
    episode_reward[i_model, :] = torch.tensor([saved_data.episode_reward_mean_list])

    if parameters.is_load_out_td:
        # Load the model with the highest reward
        parameters.episode_reward_mean_current = find_the_highest_reward_among_all_models(model_path)
        parameters.mode_name = get_model_name(parameters=parameters)
        path_eval_out_td = parameters.where_to_save + parameters.mode_name + "_out_td.pth"
        out_td = torch.load(path_eval_out_td)
        
        evaluate_outputs()
    else:
        env, policy, parameters = mappo_cavs(parameters=parameters)
        
        sim_begin = time.time()
        with torch.no_grad():
            out_td = env.rollout(
                max_steps=parameters.max_steps-1,
                policy=policy,
                callback=(lambda env, _: env.render()) if parameters.num_vmas_envs == 1 else None,
                auto_cast_to_device=True,
                break_when_any_done=False,
            )
        sim_end = time.time() - sim_begin
        
        print(colored(f"[INFO] Total execution time for {parameters.num_vmas_envs} simulations (each has {parameters.max_steps} steps): {sim_end:.3f} sec.", "blue"))
        print(colored(f"[INFO] One-step execution time {(sim_end / parameters.num_vmas_envs / parameters.max_steps):.4f} sec.", "blue"))

        evaluate_outputs()
        
        plt.close('all')


collision_rate_with_agents = remove_max_min_per_row(collision_rate_with_agents)
collision_rate_with_lanelets = remove_max_min_per_row(collision_rate_with_lanelets)
collision_rate_sum = collision_rate_with_agents[:] + collision_rate_with_lanelets[:]


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


# Define the directory path
where_to_save_eva_results = "outputs/road_traffic_ppo/eva"

# Check if the directory exists
if not os.path.exists(where_to_save_eva_results):
    os.makedirs(where_to_save_eva_results)
    print(f"[INFO] Directory '{where_to_save_eva_results}' was created.")
else:
    print(f"[INFO] Directory '{where_to_save_eva_results}' already exists.")


torch.set_printoptions(precision=3)
# Logs
print(colored(f"[LOG] Median velocity [m/s]: {velocity_average.median(dim=-1)[0]}", "black"))
print(colored(f"[LOG] Median collision rate [%]: {collision_rate_sum.median(dim=-1)[0] * 100}", "black"))
print(colored(f"[LOG] Median deviation from center line [m]: {distance_ref_average.median(dim=-1)[0]}", "black"))



###############################
## Fig 1 - Episode reward
###############################
data_np = episode_reward.numpy()
plt.clf()
plt.figure(figsize=(3.5, 2.3))

for i in range(data_np.shape[0]):
    # Original data with transparency
    plt.plot(data_np[i, :], color=colors[i], alpha=0.2, linestyle="-", linewidth=0.15)
    
    # Smoothed data
    smoothed_reward = smooth_data(data_np[i, :])
    plt.plot(smoothed_reward, label=labels_short[i], color=colors[i], linestyle="-", linewidth=0.7)

plt.xlim([0, data_np.shape[1]])
plt.xlabel('Episode')
plt.ylabel('Episode mean reward')
plt.legend(loc='lower right', fontsize="small", ncol=4)
# plt.legend(bbox_to_anchor=(1, 0.5), loc='center right', fontsize=fontsize)
# plt.legend(bbox_to_anchor=(0.5, 1.0), loc='upper center', fontsize='x-small', ncol=5)

plt.ylim([-1.5, 6])

plt.tight_layout()
# Save figure
if parameters.is_save_eval_results:
    path_save_eval_fig = f"{where_to_save_eva_results}/{timestamp}_episode_reward.pdf"
    plt.savefig(path_save_eval_fig, format="pdf", bbox_inches="tight", pad_inches=0, dpi=300)
    print(colored(f"[INFO] A fig has been saved under", "black"), colored(f"{path_save_eval_fig}", "blue"))


###############################
## Fig 2 - collision rate
###############################
plt.clf()
fig, ax = plt.subplots(figsize=(3.5, 2.0))
data_np = collision_rate_sum.numpy() * 100
data_with_agents_np = collision_rate_with_agents.numpy() * 100
data_with_lanelets_np = collision_rate_with_lanelets.numpy() * 100

# Positions of the violin plots (adjust as needed to avoid overlap)
positions = np.arange(1, len(labels) + 1)
offset = 0.2  # Offset for positioning the violins side by side

# Plotting each dataset with different colors
parts1 = ax.violinplot(dataset=data_np.T, positions=positions - offset, showmeans=False, showmedians=True, widths=0.2)
parts2 = ax.violinplot(dataset=data_with_agents_np.T, positions=positions, showmeans=False, showmedians=True, widths=0.2)
parts3 = ax.violinplot(dataset=data_with_lanelets_np.T, positions=positions + offset, showmeans=False, showmedians=True, widths=0.2)

# Set colors for each violin plot
custom_violinplot_color(parts1, Color.red100, Color.black100, 0.5)
custom_violinplot_color(parts2, Color.blue100, Color.black100, 0.15)
custom_violinplot_color(parts3, Color.green100, Color.black100, 0.15)

# Setting ticks and labels
ax.set_xticks(positions)
ax.set_xticklabels(labels_short, rotation=45, ha="right", fontsize='small')
ax.set_ylabel(r'Collision rate [$\%$]')
ax.set_ylim([0, 2.5]) # [%]
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f')) # Set y-axis tick labels to have two digits after the comma
ax.xaxis.set_minor_locator(ticker.NullLocator()) # Make minor x-ticks invisible
ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.1)
ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.1)

# Adding legend
ax.legend([parts1["bodies"][0], parts2["bodies"][0], parts3["bodies"][0]], ['Total', 'Agent-agent', 'Agent-boundary'], loc='upper right', fontsize=fontsize)

plt.tight_layout()
# Save figure
if parameters.is_save_eval_results:
    path_save_eval_fig = f"{where_to_save_eva_results}/{timestamp}_collision_rate.pdf"
    plt.savefig(path_save_eval_fig, format="pdf", bbox_inches="tight", pad_inches=0, dpi=300)
    print(colored(f"[INFO] A fig has been saved under", "black"), colored(f"{path_save_eval_fig}", "blue"))
    
    
###############################
## Fig 3 - center line deviation
###############################
plt.clf()
fig, ax = plt.subplots(figsize=(3.5, 2.0))

data_np = distance_ref_average.numpy()
ax.violinplot(dataset = data_np.T, showmeans=False, showmedians=True)
ax.set_xticks(np.arange(1, len(labels) + 1))
ax.set_xticklabels(labels_short, rotation=45, ha="right", fontsize='small')
ax.set_ylabel(r'Center line deviation [m]')
ax.set_ylim([0.04, 0.1]) # [m]
ax.xaxis.set_minor_locator(ticker.NullLocator()) # Make minor x-ticks invisible
ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.1)
ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.1)

# Save figure
plt.tight_layout()
if parameters.is_save_eval_results:
    path_save_eval_fig = f"{where_to_save_eva_results}/{timestamp}_deviation_average.pdf"
    plt.savefig(path_save_eval_fig, format="pdf", bbox_inches="tight", pad_inches=0, dpi=300)
    print(colored(f"[INFO] A fig has been saved under", "black"), colored(f"{path_save_eval_fig}", "blue"))

    
    
###############################
## Fig 4 - average velocity 
###############################
plt.clf()
fig, ax = plt.subplots(figsize=(3.5, 2.0))
data_np = velocity_average.numpy()
ax.violinplot(dataset = data_np.T, showmeans=False, showmedians=True)
ax.set_xticks(np.arange(1, len(labels) + 1))
ax.set_xticklabels(labels_with_numbers, rotation=45, ha="right", fontsize='small')
ax.set_ylabel(r'Velocity [m/s]')
ax.set_ylim([0.7, 0.8]) # [m/s]
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f')) # Set y-axis tick labels to have two digits after the comma
ax.xaxis.set_minor_locator(ticker.NullLocator()) # Make minor x-ticks invisible
ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.1)
ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.1)
# Save figure
plt.tight_layout()
if parameters.is_save_eval_results:
    path_save_eval_fig = f"{where_to_save_eva_results}/{timestamp}_velocity_average.pdf"
    plt.savefig(path_save_eval_fig, format="pdf", bbox_inches="tight", pad_inches=0, dpi=300)
    print(colored(f"[INFO] A fig has been saved under", "black"), colored(f"{path_save_eval_fig}", "cyan"))
# plt.show()