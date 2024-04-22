# !Important: Add project root to system path
import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from datetime import datetime
import torch
import numpy as np
from termcolor import colored, cprint

from operator import itemgetter

from tensordict import TensorDict

from utilities.colors import Color

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from vmas.simulator.utils import save_video


# Scientific plotting
import scienceplots # Do not remove (https://github.com/garrettj403/SciencePlots)
plt.rcParams.update({'figure.dpi': '100'}) # Avoid DPI problem (https://github.com/garrettj403/SciencePlots/issues/60)
plt.style.use(['science','ieee']) # The science + ieee styles for IEEE papers (can also be one of 'ieee' and 'science' )
# print(plt.style.available) # List all available style

import time
import json

from mappo_cavs import mappo_cavs
from utilities.helper_training import Parameters, SaveData, find_the_highest_reward_among_all_models, get_model_name

colors = [
    Color.blue100, Color.purple100, Color.violet100, Color.bordeaux100, Color.red100, Color.orange100, Color.maygreen100, Color.green100, Color.turquoise100, Color.petrol100, Color.yellow100, Color.magenta100, Color.black100,
    Color.blue50, Color.purple50, Color.violet50, Color.bordeaux50, Color.red50, Color.orange50, Color.maygreen50, Color.green50, Color.turquoise50, Color.petrol50, Color.yellow50, Color.magenta50, Color.black50,
] # Each agent will get a different color


class Evaluation:
    """
    A class for handling the evaluation of simulation outputs.
    """
    def __init__(self, model_paths, where_to_save_eva_results, models_selected = None, render_titles = None):
        """
        Initializes the Evaluation object with model paths and titles for rendering.

        Args:
        model_paths (list): List of paths to the model directories.
        render_titles (list): Titles for rendering the plots.
        """
        self.model_paths = model_paths
        
        self.where_to_save_eva_results = where_to_save_eva_results
        
        self.models_selected = models_selected
        self.num_models_selected = len(self.models_selected)

        self.render_titles = render_titles
        
        self.parameters = None  # This will be set when loading model parameters

        self.saved_data = None

        self.velocity_average = None
        self.collision_rate_with_agents = None
        self.collision_rate_with_lanelets = None
        self.distance_ref_average = None
        self.episode_reward = None
        
        
        num_models = len(model_paths)

        self.labels = [m.split('/')[-2] for m in model_paths]
        self.labels_short = [f"M{idx + 1}" for idx in range(num_models)]

        idx_our_model = next((i for i, s in enumerate(model_paths) if "our" in s), None)
        # labels_short[idx_our_model] += " (our)" 

        self.labels_with_numbers = [self.labels_short[idx] + " (" + l + ")" for idx, l in enumerate(self.labels)]
        self.labels_with_numbers[idx_our_model] = self.labels_with_numbers[idx_our_model][0:8]        

    def load(self, model_path):
        """
        Loads parameters from a JSON file located in the model path.

        Args:
            model_path (str): Path to the model directory.
        """
        try:
            path_to_json_file = next(os.path.join(model_path, file) for file in os.listdir(model_path) if file.endswith('.json')) # Find the first json file in the folder
            # Load parameters from the saved json file
            with open(path_to_json_file, 'r') as file:
                data = json.load(file)
                self.saved_data = SaveData.from_dict(data)
                self.parameters = self.saved_data.parameters
        except StopIteration:
            raise FileNotFoundError("No json file found.")
        

    def evaluate_outputs(self, model_path, i_model_selected, i_model):
        """
        Evaluate the outputs for the given TensorDict.
        """
        out_td = self.get_simulation_outputs(model_path, i_model)
        
        positions = out_td["agents","info","pos"]
        velocities = out_td["agents","info","vel"]
        is_collision_with_agents = out_td["agents","info","is_collision_with_agents"].bool()
        is_collision_with_lanelets = out_td["agents","info","is_collision_with_lanelets"].bool()
        distance_ref = out_td["agents","info","distance_ref"]
        
        is_collide = is_collision_with_agents | is_collision_with_lanelets
        
        num_steps = positions.shape[1]

        self.velocity_average[i_model_selected, :] = velocities.norm(dim=-1).mean(dim=(-2, -1))
        self.collision_rate_with_agents[i_model_selected, :] = is_collision_with_agents.squeeze(-1).any(dim=-1).sum(dim=-1) / num_steps
        self.collision_rate_with_lanelets[i_model_selected, :] = is_collision_with_lanelets.squeeze(-1).any(dim=-1).sum(dim=-1) / num_steps
        self.distance_ref_average[i_model_selected, :] = distance_ref.squeeze(-1).mean(dim=(-2, -1))

        path_eval_out_td = self.parameters.where_to_save + self.parameters.mode_name + "_out_td.pth"
        if self.parameters.is_save_eval_results and (not self.parameters.is_save_simulation_video) and (self.parameters.num_vmas_envs > 1) and (not self.parameters.is_load_out_td):
            # Save the input TensorDict
            torch.save(out_td, path_eval_out_td)   
        
    def get_simulation_outputs(self, model_path: str, i_model):
        if self.parameters.is_load_out_td:
            # Load the model with the highest reward
            self.parameters.episode_reward_mean_current = find_the_highest_reward_among_all_models(model_path)
            self.parameters.mode_name = get_model_name(parameters=self.parameters)
            path_eval_out_td = self.parameters.where_to_save + self.parameters.mode_name + "_out_td.pth"
            out_td = torch.load(path_eval_out_td)
        else:
            env, policy, self.parameters = mappo_cavs(parameters=self.parameters)
            
            sim_begin = time.time()
            with torch.no_grad():
                if self.parameters.is_save_simulation_video:    
                    out_td, frame_list = env.rollout(
                        max_steps=self.parameters.max_steps-1,
                        policy=policy,
                        callback=lambda env, _: env.render(mode="rgb_array", visualize_when_rgb=True), # mode \in {"human", "rgb_array"}
                        auto_cast_to_device=True,
                        break_when_any_done=False,
                        is_save_simulation_video=self.parameters.is_save_simulation_video,
                    )
                    sim_end = time.time() - sim_begin
                    save_video(f"{model_path}{render_titles[i_model]}", frame_list, fps=1 / self.parameters.dt)
                else:
                    out_td = env.rollout(
                        max_steps=self.parameters.max_steps-1,
                        policy=policy,
                        callback=(lambda env, _: env.render(mode="human")) if self.parameters.num_vmas_envs == 1 else None, # mode \in {"human", "rgb_array"}
                        auto_cast_to_device=True,
                        break_when_any_done=False,
                        is_save_simulation_video=False,
                    )
                    sim_end = time.time() - sim_begin
            
            print(colored(f"[INFO] Total execution time for {self.parameters.num_vmas_envs} simulations (each has {self.parameters.max_steps} steps): {sim_end:.3f} sec.", "blue"))
            print(colored(f"[INFO] One-step execution time {(sim_end / self.parameters.num_vmas_envs / self.parameters.max_steps):.4f} sec.", "blue"))
        return out_td
    
    @staticmethod
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
    @staticmethod
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

    @staticmethod
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

    def plot(self):
        num_models_selected = len(self.models_selected)
        
        fontsize = 8
        
        if not self.parameters.is_save_simulation_video:
            self.collision_rate_with_agents = self.remove_max_min_per_row(self.collision_rate_with_agents) # Remove the best and the worst to eliminate the influence of the stochastic nature of the randomness
            self.collision_rate_with_lanelets = self.remove_max_min_per_row(self.collision_rate_with_lanelets)
            collision_rate_sum = self.collision_rate_with_agents[:] + self.collision_rate_with_lanelets[:]


            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Check if the directory exists
            if not os.path.exists(self.where_to_save_eva_results):
                os.makedirs(self.where_to_save_eva_results)
                print(f"[INFO] Directory '{self.where_to_save_eva_results}' was created.")
            else:
                print(f"[INFO] Directory '{self.where_to_save_eva_results}' already exists.")


            torch.set_printoptions(precision=3)
            # Logs
            print(colored(f"[LOG] Median velocity [m/s]: {self.velocity_average.median(dim=-1)[0]}", "black"))
            print(colored(f"[LOG] Median collision rate [%]: {collision_rate_sum.median(dim=-1)[0] * 100}", "black"))
            print(colored(f"[LOG] Median deviation from center line [m]: {self.distance_ref_average.median(dim=-1)[0]}", "black"))

            ###############################
            ## Fig 1 - Episode reward
            ###############################
            data_np = self.episode_reward.numpy()
            plt.clf()
            plt.figure(figsize=(3.5, 2.3))

            for i in range(data_np.shape[0]):
                # Original data with transparency
                plt.plot(data_np[i, :], color=colors[i], alpha=0.2, linestyle="-", linewidth=0.15)
                
                # Smoothed data
                smoothed_reward = self.smooth_data(data_np[i, :])
                plt.plot(smoothed_reward, label=self.labels_short[models_selected[i]], color=colors[i], linestyle="-", linewidth=0.7)

            plt.xlim([0, data_np.shape[1]])
            plt.xlabel('Episode')
            plt.ylabel('Episode mean reward')
            plt.legend(loc='lower right', fontsize="small", ncol=4)
            # plt.legend(bbox_to_anchor=(1, 0.5), loc='center right', fontsize=fontsize)
            # plt.legend(bbox_to_anchor=(0.5, 1.0), loc='upper center', fontsize='x-small', ncol=5)

            plt.ylim([-1.5, 6])

            plt.tight_layout()
            # Save figure
            if self.parameters.is_save_eval_results:
                path_save_eval_fig = f"{self.where_to_save_eva_results}/{timestamp}_episode_reward.pdf"
                plt.savefig(path_save_eval_fig, format="pdf", bbox_inches="tight", pad_inches=0, dpi=300)
                path_save_eval_fig = f"{self.where_to_save_eva_results}/{timestamp}_episode_reward.png"
                plt.savefig(path_save_eval_fig, format="png", bbox_inches="tight", pad_inches=0, dpi=600)
                print(colored(f"[INFO] A fig has been saved under", "black"), colored(f"{path_save_eval_fig}", "blue"))


            ###############################
            ## Fig 2 - collision rate
            ###############################
            plt.clf()
            fig, ax = plt.subplots(figsize=(3.5, 2.0))
            data_np = collision_rate_sum.numpy() * 100
            data_with_agents_np = self.collision_rate_with_agents.numpy() * 100
            data_with_lanelets_np = self.collision_rate_with_lanelets.numpy() * 100

            # Positions of the violin plots (adjust as needed to avoid overlap)
            positions = np.arange(1, num_models_selected + 1)
            offset = 0.2  # Offset for positioning the violins side by side

            # Plotting each dataset with different colors
            parts1 = ax.violinplot(dataset=data_np.T, positions=positions - offset, showmeans=False, showmedians=True, widths=0.2)
            parts2 = ax.violinplot(dataset=data_with_agents_np.T, positions=positions, showmeans=False, showmedians=True, widths=0.2)
            parts3 = ax.violinplot(dataset=data_with_lanelets_np.T, positions=positions + offset, showmeans=False, showmedians=True, widths=0.2)

            # Set colors for each violin plot
            self.custom_violinplot_color(parts1, Color.red100, Color.black100, 0.5)
            self.custom_violinplot_color(parts2, Color.blue100, Color.black100, 0.15)
            self.custom_violinplot_color(parts3, Color.green100, Color.black100, 0.15)

            # Setting ticks and labels
            ax.set_xticks(positions)
            ax.set_xticklabels(itemgetter(*models_selected)(self.labels_short), rotation=45, ha="right", fontsize='small')
            ax.set_ylabel(r'Collision rate [$\%$]')
            ax.set_ylim([0, 2.5]) # [%]
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f')) # Set y-axis tick labels to have two digits after the comma
            ax.xaxis.set_minor_locator(ticker.NullLocator()) # Make minor x-ticks invisible
            ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.1)
            ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.1)

            # Adding legend
            if (num_models_selected == 7) or (num_models_selected == 4):
                ax.legend([parts1["bodies"][0], parts2["bodies"][0], parts3["bodies"][0]], ['Total', 'Agent-agent', 'Agent-boundary'], loc='upper center', bbox_to_anchor=(0.5, 1.2), fontsize=fontsize, ncol=3)
            else:
                ax.legend([parts1["bodies"][0], parts2["bodies"][0], parts3["bodies"][0]], ['Total', 'Agent-agent', 'Agent-boundary'], loc='upper right', fontsize=fontsize)
                

            plt.tight_layout()
            # Save figure
            if self.parameters.is_save_eval_results:
                path_save_eval_fig = f"{self.where_to_save_eva_results}/{timestamp}_collision_rate.pdf"
                plt.savefig(path_save_eval_fig, format="pdf", bbox_inches="tight", pad_inches=0, dpi=300)
                path_save_eval_fig = f"{self.where_to_save_eva_results}/{timestamp}_collision_rate.png"
                plt.savefig(path_save_eval_fig, format="png", bbox_inches="tight", pad_inches=0, dpi=600)
                print(colored(f"[INFO] A fig has been saved under", "black"), colored(f"{path_save_eval_fig}", "blue"))
                
                
            ###############################
            ## Fig 3 - center line deviation
            ###############################
            plt.clf()
            fig, ax = plt.subplots(figsize=(3.5, 2.0))

            data_np = self.distance_ref_average.numpy()
            ax.violinplot(dataset = data_np.T, showmeans=False, showmedians=True)
            ax.set_xticks(np.arange(1,num_models_selected + 1))
            ax.set_xticklabels(itemgetter(*models_selected)(self.labels_short), rotation=45, ha="right", fontsize='small')
            ax.set_ylabel(r'Center line deviation [m]')
            ax.set_ylim([0.04, 0.1]) # [m]
            ax.xaxis.set_minor_locator(ticker.NullLocator()) # Make minor x-ticks invisible
            ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.1)
            ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.1)

            # Save figure
            plt.tight_layout()
            if self.parameters.is_save_eval_results:
                path_save_eval_fig = f"{self.where_to_save_eva_results}/{timestamp}_deviation_average.pdf"
                plt.savefig(path_save_eval_fig, format="pdf", bbox_inches="tight", pad_inches=0, dpi=300)
                path_save_eval_fig = f"{self.where_to_save_eva_results}/{timestamp}_deviation_average.png"
                plt.savefig(path_save_eval_fig, format="png", bbox_inches="tight", pad_inches=0, dpi=600)
                print(colored(f"[INFO] A fig has been saved under", "black"), colored(f"{path_save_eval_fig}", "blue"))

            ###############################
            ## Fig 4 - average velocity 
            ###############################
            plt.clf()
            fig, ax = plt.subplots(figsize=(3.5, 2.0))
            data_np = self.velocity_average.numpy()
            ax.violinplot(dataset = data_np.T, showmeans=False, showmedians=True)
            ax.set_xticks(np.arange(1, num_models_selected + 1))
            ax.set_xticklabels(itemgetter(*models_selected)(self.labels_with_numbers), rotation=45, ha="right", fontsize='small')
            ax.set_ylabel(r'Velocity [m/s]')
            ax.set_ylim([0.7, 0.8]) # [m/s]
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f')) # Set y-axis tick labels to have two digits after the comma
            ax.xaxis.set_minor_locator(ticker.NullLocator()) # Make minor x-ticks invisible
            ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.1)
            ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.1)
            # Save figure
            plt.tight_layout()
            if self.parameters.is_save_eval_results:
                path_save_eval_fig = f"{self.where_to_save_eva_results}/{timestamp}_velocity_average.pdf"
                plt.savefig(path_save_eval_fig, format="pdf", bbox_inches="tight", pad_inches=0, dpi=300)
                path_save_eval_fig = f"{self.where_to_save_eva_results}/{timestamp}_velocity_average.png"
                plt.savefig(path_save_eval_fig, format="png", bbox_inches="tight", pad_inches=0, dpi=600)
                print(colored(f"[INFO] A fig has been saved under", "black"), colored(f"{path_save_eval_fig}", "cyan"))
            # plt.show()

    def run_evaluation(self):
        """
        Main method to run evaluation over all provided model paths.
        """
        i_model_selected = 0
        
        for i_model in models_selected:
            print("------------------------------------------")
            print(colored("-- [INFO] Model ", "black"), colored(f"{i_model + 1}", "blue"), colored(f"({self.labels[i_model]})", color="grey"))
            print("------------------------------------------")
            
            model_path = model_paths[i_model]
            
            self.load(model_path)
            
            # Adjust parameters 
            self.parameters.is_testing_mode = True
            self.parameters.is_real_time_rendering = False
            self.parameters.is_save_eval_results = True
            self.parameters.is_load_model = True
            self.parameters.is_load_final_model = False
            self.parameters.is_load_out_td  = True
            self.parameters.n_agents = 12
            self.parameters.max_steps = 1200 # 1200 -> 1 min
            if self.parameters.is_load_out_td:
                self.parameters.num_vmas_envs = 32
            else:
                self.parameters.num_vmas_envs = 1
            self.parameters.frames_per_batch = self.parameters.max_steps * self.parameters.num_vmas_envs
            self.parameters.training_strategy = "1"
            self.parameters.is_save_simulation_video = False
            self.parameters.is_visualize_short_term_path = False
            self.parameters.is_visualize_lane_boundary = False
            self.parameters.is_visualize_extra_info = True
            self.parameters.render_title = self.render_titles[i_model]

            if self.velocity_average is None:
                # Initialize
                self.velocity_average = torch.zeros((self.num_models_selected, self.parameters.num_vmas_envs), device=self.parameters.device, dtype=torch.float32)
                # collision_rate_sum = torch.zeros((num_models_selected, parameters.num_vmas_envs), device=parameters.device, dtype=torch.float32)
                self.collision_rate_with_agents = torch.zeros((self.num_models_selected, self.parameters.num_vmas_envs), device=self.parameters.device, dtype=torch.float32)
                self.collision_rate_with_lanelets = torch.zeros((self.num_models_selected, self.parameters.num_vmas_envs), device=self.parameters.device, dtype=torch.float32)
                self.distance_ref_average = torch.zeros((self.num_models_selected, self.parameters.num_vmas_envs), device=self.parameters.device, dtype=torch.float32)
                self.episode_reward = torch.zeros((self.num_models_selected, self.parameters.n_iters), device=self.parameters.device, dtype=torch.float32)
                
            self.episode_reward[i_model_selected, :] = torch.tensor([self.saved_data.episode_reward_mean_list])
            
            self.evaluate_outputs(model_path, i_model_selected, i_model)
            
            i_model_selected += 1
            
            
if __name__ == "__main__":
    scenario_name = "road_traffic" # road_traffic, path_tracking, obstacle_avoidance

    # parameters = Parameters()

    model_paths = [
        "outputs/road_traffic_ppo/bird-view/",
        "outputs/road_traffic_ppo/no mask/",
        "outputs/road_traffic_ppo/obs. vertices of surr. ag./",
        "outputs/road_traffic_ppo/not obs. dist. to surr. ag./",
        "outputs/road_traffic_ppo/obs. boundary points/",
        "outputs/road_traffic_ppo/not obs. dist. to center line/",
        "outputs/road_traffic_ppo/our/",
        "outputs/road_traffic_ppo/vanilla/",
        "outputs/road_traffic_ppo/chall. initial state buffer/",
        "outputs/road_traffic_ppo/PER/",
    ]

    render_titles = [
        "M1 (bird's-eye view)",
        "M2 (no mask)",
        "M3 (observe vertices of surrounding agents)",
        "M4 (not observe distance to surrounding agents)",
        "M5 (observe boundary points)",
        "M6 (not observe distance to center line)",
        "M7 (our)",
        "M8 (vanilla)",
        "M9 (challenge initial state buffer)",
        "M10 (prioritized experience replay)",
    ]

    models_selected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # All 
    # models_selected = [0, 1, 2, 3, 4, 5, 6] # Ablation studies
    # models_selected = [6, 7, 8, 9] # Comparisons with state of the arts
    
    # Define the directory path
    where_to_save_eva_results = "outputs/road_traffic_ppo/eva"

    evaluator = Evaluation(model_paths, where_to_save_eva_results, models_selected, render_titles)
    
    evaluator.run_evaluation()
    evaluator.plot()