# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Add project root to system path
import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from datetime import datetime
import torch
import numpy as np
from termcolor import colored, cprint

from operator import itemgetter

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import matplotlib

# Use Type 1 fonts (vector fonts) for IEEE paper submission
matplotlib.rcParams["pdf.fonttype"] = 42  # Use Type 42 (TrueType) fonts

# DejaVu fonts are embedded as vector fonts by default.
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["DejaVu Serif"]  # Use a serif font

from vmas.simulator.utils import save_video

# Scientific plotting
import scienceplots  # Do not remove (https://github.com/garrettj403/SciencePlots)

plt.rcParams.update(
    {"figure.dpi": "100"}
)  # Avoid DPI problem (https://github.com/garrettj403/SciencePlots/issues/60)
plt.style.use(
    ["science", "ieee"]
)  # The science + ieee styles for IEEE papers (can also be one of 'ieee' and 'science' )
# print(plt.style.available) # List all available style

import time
import json

from utilities.mappo_cavs import mappo_cavs
from utilities.helper_training import (
    SaveData,
    find_the_highest_reward_among_all_models,
    get_model_name,
)

from utilities.constants import SCENARIOS

from utilities.colors import Color, colors


class Evaluation:
    """
    A class for handling the evaluation of simulation outputs.
    """

    def __init__(self, **kwargs):
        """
        Initializes the Evaluation object with model paths and titles for rendering.

        Args:
        model_paths (list): List of paths to the model directories.
        where_to_save_eva_results (str): Where to save evaluation results.
        render_titles (list): Titles for rendering the plots.
        video_names (list): Video names.
        """
        self.scenario_type = kwargs.pop("scenario_type", None)

        self.model_paths = kwargs.pop("model_paths", None)

        self.where_to_save_eva_results = kwargs.pop(
            "where_to_save_eva_results", "outputs"
        )

        self.where_to_save_logging = kwargs.pop("where_to_save_logging", "log.txt")

        self.models_selected = kwargs.pop("models_selected", None)
        if (self.models_selected is None) or (self.models_selected == []):
            # If not specified, select all
            self.models_selected = list(range(len(self.model_paths)))

        self.legends = kwargs.pop("legends", None)
        self.render_titles = kwargs.pop("render_titles", None)
        self.num_simulations_per_model = kwargs.pop("num_simulations_per_model", 32)
        self.num_agents = kwargs.pop("num_agents", 15)
        self.simulation_steps = kwargs.pop("simulation_steps", 15)

        self.is_render = kwargs.pop("is_render", True)
        self.is_save_simulation_video = kwargs.pop("is_save_simulation_video", False)
        self.video_names = kwargs.pop("video_names", None)

        self.num_models_selected = len(self.models_selected)

        self.idx_our = next(
            (i for i, path in enumerate(self.model_paths) if "our" in path), -1
        )  # Index of our model
        print(f"Index of our model: {self.idx_our}")

        self.parameters = None  # This will be set when loading model parameters

        self.saved_data = None

        num_models = len(self.model_paths)

        self.labels = [m.split("/")[-2] for m in self.model_paths]
        self.labels_short = [f"M{idx}" for idx in range(num_models)]

        idx_our_model = next(
            (i for i, s in enumerate(self.model_paths) if "our" in s), None
        )
        # labels_short[idx_our_model] += " (our)"

        self.labels_with_numbers = [
            self.labels_short[idx] + " (" + l + ")" for idx, l in enumerate(self.labels)
        ]
        self.labels_with_numbers[idx_our_model] = self.labels_with_numbers[
            idx_our_model
        ][0:8]

    def _load_parameters(self):
        """
        Loads parameters from a JSON file located in the model path.
        """
        try:
            path_to_json_file = next(
                os.path.join(self.model_i_path, file)
                for file in os.listdir(self.model_i_path)
                if file.endswith(".json")
            )  # Find the first json file in the folder
            # Load parameters from the saved json file
            with open(path_to_json_file, "r") as file:
                data = json.load(file)
                self.saved_data = SaveData.from_dict(data)
                self.parameters = self.saved_data.parameters
        except StopIteration:
            raise FileNotFoundError("No json file found.")

    def _adjust_parameters(self):
        # Adjust parameters
        self.parameters.scenario_type = self.scenario_type
        self.parameters.is_testing_mode = True
        self.parameters.is_real_time_rendering = False
        self.parameters.is_save_eval_results = True
        self.parameters.is_load_model = True
        self.parameters.is_load_final_model = False
        self.parameters.is_load_out_td = True
        self.parameters.n_agents = self.num_agents
        self.parameters.max_steps = self.simulation_steps

        self.parameters.is_save_simulation_video = self.is_save_simulation_video

        if self.is_render or self.parameters.is_save_simulation_video:
            # Only one env is needed if video should be shown or saved
            self.parameters.num_vmas_envs = 1
            self.parameters.is_load_out_td = False
        else:
            # Otherwise run multiple parallel envs and average the evaluation results
            self.parameters.num_vmas_envs = self.num_simulations_per_model
        self.parameters.frames_per_batch = (
            self.parameters.max_steps * self.parameters.num_vmas_envs
        )

        # The two parameters below are only used in training. Set them to False so that they do not effect testing
        self.parameters.is_prb = False
        self.parameters.is_challenging_initial_state_buffer = False

        self.parameters.is_visualize_short_term_path = False
        self.parameters.is_visualize_lane_boundary = False
        self.parameters.is_visualize_extra_info = False
        self.parameters.render_title = self.render_titles[self.model_i]

    def _evaluate_model_i(self):
        """
        Evaluate a model.
        """
        out_td = self._get_simulation_outputs()

        positions = out_td["agents", "info", "pos"]
        velocities = out_td["agents", "info", "vel"]
        is_collision_with_agents = out_td[
            "agents", "info", "is_collision_with_agents"
        ].bool()
        is_collision_with_lanelets = out_td[
            "agents", "info", "is_collision_with_lanelets"
        ].bool()
        distance_ref = out_td["agents", "info", "distance_ref"]

        is_collide = is_collision_with_agents | is_collision_with_lanelets

        num_steps = positions.shape[1]

        self.velocity_average[self.model_idx, :] = velocities.norm(dim=-1).mean(
            dim=(-2, -1)
        )
        self.collision_rate_with_agents[self.model_idx, :] = (
            is_collision_with_agents.squeeze(-1).any(dim=-1).sum(dim=-1) / num_steps
        )
        self.collision_rate_with_lanelets[self.model_idx, :] = (
            is_collision_with_lanelets.squeeze(-1).any(dim=-1).sum(dim=-1) / num_steps
        )
        self.distance_ref_average[self.model_idx, :] = distance_ref.squeeze(-1).mean(
            dim=(-2, -1)
        )

    def _get_simulation_outputs(self):
        # Load the model with the highest reward
        self.parameters.episode_reward_mean_current = (
            find_the_highest_reward_among_all_models(self.model_i_path)
        )
        self.parameters.model_name = get_model_name(parameters=self.parameters)
        path_eval_out_td = (
            self.parameters.where_to_save
            + self.parameters.model_name
            + f"_out_td_{self.parameters.scenario_type}.pth"
        )
        # Load simulation outputs if exist; otherwise run simulation
        if os.path.exists(path_eval_out_td) & (not self.is_render):
            cprint("[INFO] Simulation outputs exist and will be loaded.", "grey")
            out_td = torch.load(path_eval_out_td)
        else:
            if self.parameters.is_using_prioritized_marl:
                env, policy, priority_module, _ = mappo_cavs(parameters=self.parameters)
            else:
                env, policy, _ = mappo_cavs(parameters=self.parameters)

            cprint("[INFO] Run simulation...", "grey")
            sim_begin = time.time()

            if self.parameters.is_save_simulation_video:
                with torch.no_grad():
                    if self.parameters.is_using_prioritized_marl:
                        out_td, frame_list = env.rollout(
                            max_steps=self.parameters.max_steps - 1,
                            policy=policy,
                            callback=lambda env, _: env.render(
                                mode="rgb_array", visualize_when_rgb=False
                            ),  # mode \in {"human", "rgb_array"}
                            auto_cast_to_device=True,
                            break_when_any_done=False,
                            is_save_simulation_video=self.parameters.is_save_simulation_video,
                            priority_module=priority_module,
                        )
                    else:
                        out_td, frame_list = env.rollout(
                            max_steps=self.parameters.max_steps - 1,
                            policy=policy,
                            callback=lambda env, _: env.render(
                                mode="rgb_array", visualize_when_rgb=False
                            ),  # mode \in {"human", "rgb_array"}
                            auto_cast_to_device=True,
                            break_when_any_done=False,
                            is_save_simulation_video=self.parameters.is_save_simulation_video,
                        )

                sim_end = time.time() - sim_begin

                save_video(
                    f"{self.model_i_path}{self.video_names[self.model_i]}_{self.scenario_type}",
                    frame_list,
                    fps=1 / self.parameters.dt,
                )
                print(
                    colored(f"[INFO] Video saved under ", "grey"),
                    colored(f"{self.model_i_path}.", "blue"),
                )
            else:
                with torch.no_grad():
                    out_td = env.rollout(
                        max_steps=self.parameters.max_steps - 1,
                        policy=policy,
                        callback=(
                            (lambda env, _: env.render(mode="human"))
                            if self.parameters.num_vmas_envs == 1
                            else None
                        ),  # mode should be one of {"human", "rgb_array"}
                        auto_cast_to_device=True,
                        break_when_any_done=False,
                        is_save_simulation_video=False,
                    )
                sim_end = time.time() - sim_begin

            # Save simulation outputs
            torch.save(out_td, path_eval_out_td)
            print(colored("[INFO] Simulation outputs saved.", "grey"))

            print(
                colored(
                    f"[INFO] Total execution time for {self.parameters.num_vmas_envs} simulations (each has {self.parameters.max_steps} steps): {sim_end:.3f} sec.",
                    "blue",
                )
            )
            print(
                colored(
                    f"[INFO] One-step execution time {(sim_end / self.parameters.num_vmas_envs / self.parameters.max_steps):.4f} sec.",
                    "blue",
                )
            )

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
        tensor[row_same_value.unsqueeze(-1), 0] = float("inf")
        tensor[row_same_value.unsqueeze(-1), 1] = float("-inf")

        tensor[
            row_different_values.unsqueeze(-1),
            max_indices[is_row_different_values].unsqueeze(-1),
        ] = float("inf")
        tensor[
            row_different_values.unsqueeze(-1),
            min_indices[is_row_different_values].unsqueeze(-1),
        ] = float("-inf")

        # Remove the inf and -inf values
        mask = (tensor != float("inf")) & (tensor != float("-inf"))
        filtered_tensor = tensor[mask].view(tensor.size(0), -1)

        return filtered_tensor

    # Function to add custom median markers
    @staticmethod
    def custom_violinplot_color(parts, color_face, color_lines, alpha):
        for pc in parts["bodies"]:
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
        smoothed = np.convolve(data, np.ones(window_size) / window_size, mode="valid")
        return np.concatenate((data[: window_size - 1], smoothed))

    def plot(self):
        num_models_selected = len(self.models_selected)

        self.collision_rate_with_agents = self.remove_max_min_per_row(
            self.collision_rate_with_agents
        )  # Remove the best and the worst to eliminate the influence of the stochastic nature of the randomness
        self.collision_rate_with_lanelets = self.remove_max_min_per_row(
            self.collision_rate_with_lanelets
        )
        collision_rate_sum = (
            self.collision_rate_with_agents[:] + self.collision_rate_with_lanelets[:]
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Check if the directory exists
        if not os.path.exists(self.where_to_save_eva_results):
            os.makedirs(self.where_to_save_eva_results)
            print(f"[INFO] Directory '{self.where_to_save_eva_results}' was created.")
        else:
            print(
                f"[INFO] Directory '{self.where_to_save_eva_results}' already exists."
            )

        # torch.set_printoptions(precision=3)
        # Logs
        CR_AA_avg = self.collision_rate_with_agents.mean(dim=-1)
        CR_AL_avg = self.collision_rate_with_lanelets.mean(dim=-1)
        CR_total_avg = collision_rate_sum.mean(dim=-1)

        CD_avg = self.distance_ref_average.mean(dim=-1)
        AS_avg = self.velocity_average.mean(dim=-1)

        # Log messages
        log_CR_AA_avg = f"[LOG] Mean agent-agent collision rate [%]: {self.format_array(CR_AA_avg.numpy() * 100)}"
        log_CR_AL_avg = f"[LOG] Mean agent-lanelet collision rate [%]: {self.format_array(CR_AL_avg.numpy() * 100)}"
        log_CR_total_avg = f"[LOG] Mean collision rate [%]: {self.format_array(CR_total_avg.numpy() * 100)}"
        log_CD_avg = f"[LOG] Mean center line deviation [cm]: {self.format_array(CD_avg.numpy() * 100)}"
        log_AS_avg = f"[LOG] Mean speed [m/s]: {self.format_array(AS_avg.numpy())}"

        # Compute composite score
        w1 = 1 / CR_total_avg.mean()
        w2 = 1 / CD_avg.mean()
        w3 = 1 / AS_avg.mean()
        # composite score = - w1 * CR_total_avg - w2 * CD_avg + w3 * AS_avg
        composite_score = -w1 * CR_total_avg - w2 * CD_avg + w3 * AS_avg

        log_CS = f"[LOG] Composite scores: {self.format_array(composite_score.numpy())}"

        cprint(log_CR_total_avg, "black")
        cprint(log_CR_AA_avg, "black")
        cprint(log_CR_AL_avg, "black")
        cprint(log_CD_avg, "black")
        cprint(log_AS_avg, "black")
        cprint(log_CS)

        # Save the evaluation results to a txt file
        with open(self.where_to_save_logging, "a") as file:
            file.write(f"Scenario: {self.parameters.scenario_type}\n")
            file.write(f"{log_CR_AA_avg}\n")
            file.write(f"{log_CR_AL_avg}\n")
            file.write(f"{log_CR_total_avg}\n")
            file.write(f"{log_CD_avg}\n")
            file.write(f"{log_AS_avg}\n")
            file.write(f"{log_CS}\n")
            file.write("=========================================")
            file.write("\n")

        ###############################
        ## Fig 1 - Episode reward
        ###############################
        data_np = self.episode_reward.numpy()
        plt.clf()
        plt.figure(figsize=(3.8, 4.3))

        for i in range(data_np.shape[0]):
            # Original data with transparency
            plt.plot(
                data_np[i, :], color=colors[i], alpha=0.2, linestyle="-", linewidth=0.2
            )

            # Smoothed data
            smoothed_reward = self.smooth_data(data_np[i, :])
            plt.plot(
                smoothed_reward,
                label=self.legends[self.models_selected[i]],
                color=colors[i],
                linestyle="-",
                linewidth=0.8,
            )

        plt.xlim([0, data_np.shape[1]])
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        plt.legend(
            bbox_to_anchor=(0.5, 1.88),
            loc="upper center",
            fontsize=9,
            ncol=1,
            handleheight=0.7,
            handletextpad=0.5,
            labelspacing=0.3,
        )
        # plt.legend(loc='lower right', fontsize="small", ncol=4)
        # plt.legend(bbox_to_anchor=(1, 0.5), loc='center right', fontsize=fontsize)

        plt.ylim([-1.0, 6.5])
        plt.tight_layout(
            rect=[0, 0, 1, 1]
        )  # Adjust the layout to make space for the legend on the right
        plt.subplots_adjust(top=0.4)  # Adjust top value to make space for the legend

        # Save figure
        if self.parameters.is_save_eval_results:
            path_save_eval_fig = f"{self.where_to_save_eva_results}/{timestamp}_{self.parameters.scenario_type}_episode_reward.pdf"
            plt.savefig(
                path_save_eval_fig,
                format="pdf",
                bbox_inches="tight",
                pad_inches=0,
                dpi=300,
            )
            path_save_eval_fig = f"{self.where_to_save_eva_results}/{timestamp}_{self.parameters.scenario_type}_episode_reward.png"
            plt.savefig(
                path_save_eval_fig,
                format="png",
                bbox_inches="tight",
                pad_inches=0,
                dpi=600,
            )
            print(
                colored(f"[INFO] A fig has been saved under", "black"),
                colored(f"{path_save_eval_fig}", "blue"),
            )

        plt.clf()

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

        # Plot a horizontal line indicating the median of our model
        median_our = np.median(data_np[self.idx_our])
        max_our = np.max(data_np[self.idx_our])
        plt.axhline(y=median_our, color="black", linestyle="--", alpha=0.3)
        ax.text(
            self.idx_our + 1,
            max_our,
            f"Our: {median_our:.2f}$\%$",
            ha="center",
            va="bottom",
            fontsize="small",
            color="grey",
        )

        # Plot each dataset with different colors
        parts1 = ax.violinplot(
            dataset=data_np.T,
            positions=positions - offset,
            showmeans=False,
            showmedians=True,
            widths=0.2,
        )
        parts2 = ax.violinplot(
            dataset=data_with_agents_np.T,
            positions=positions,
            showmeans=False,
            showmedians=True,
            widths=0.2,
        )
        parts3 = ax.violinplot(
            dataset=data_with_lanelets_np.T,
            positions=positions + offset,
            showmeans=False,
            showmedians=True,
            widths=0.2,
        )

        # Set colors for each violin plot
        self.custom_violinplot_color(parts1, Color.red100, Color.black100, 0.5)
        self.custom_violinplot_color(parts2, Color.blue100, Color.black100, 0.15)
        self.custom_violinplot_color(parts3, Color.green100, Color.black100, 0.15)

        # Setting ticks and labels
        ax.set_xticks(positions)
        ax.set_xticklabels(
            itemgetter(*self.models_selected)(self.labels_short),
            rotation=45,
            ha="right",
            fontsize="small",
        )
        ax.set_ylabel(r"Collision rate [$\%$]")
        # ax.set_ylim([0, 2.0]) # [%]
        ax.yaxis.set_major_formatter(
            ticker.FormatStrFormatter("%.2f")
        )  # Set y-axis tick labels to have two digits after the comma
        ax.xaxis.set_minor_locator(ticker.NullLocator())  # Make minor x-ticks invisible
        ax.grid(True, which="major", axis="x", linestyle="--", linewidth=0.1)
        ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.1)

        # # Add text to shown the median of M1
        # median_M1 = np.median(data_np[0])
        # ax.text(positions[0], 0.2, f'Median\n(total):\n{median_M1:.2f}$\%$', ha='center', va='bottom', fontsize='small', color='grey')

        # Adding legend
        if (num_models_selected == 7) or (num_models_selected == 4):
            ax.legend(
                [parts1["bodies"][0], parts2["bodies"][0], parts3["bodies"][0]],
                ["Total", "Agent-agent", "Agent-boundary"],
                loc="upper center",
                bbox_to_anchor=(0.5, 1.2),
                fontsize=8,
                ncol=3,
            )
        else:
            ax.legend(
                [parts1["bodies"][0], parts2["bodies"][0], parts3["bodies"][0]],
                ["Total", "Agent-agent", "Agent-boundary"],
                loc="upper right",
                fontsize=8,
            )

        plt.tight_layout()
        # Save figure
        if self.parameters.is_save_eval_results:
            path_save_eval_fig = f"{self.where_to_save_eva_results}/{timestamp}_{self.parameters.scenario_type}_collision_rate.pdf"
            plt.savefig(
                path_save_eval_fig,
                format="pdf",
                bbox_inches="tight",
                pad_inches=0,
                dpi=300,
            )
            path_save_eval_fig = f"{self.where_to_save_eva_results}/{timestamp}_{self.parameters.scenario_type}_collision_rate.png"
            plt.savefig(
                path_save_eval_fig,
                format="png",
                bbox_inches="tight",
                pad_inches=0,
                dpi=600,
            )
            print(
                colored(f"[INFO] A fig has been saved under", "black"),
                colored(f"{path_save_eval_fig}", "blue"),
            )

        ###############################
        ## Fig 3 - center line deviation
        ###############################
        plt.clf()
        fig, ax = plt.subplots(figsize=(3.5, 2.0))

        data_np = self.distance_ref_average.numpy()

        # Plot a horizontal line indicating the median of our model
        median_our = np.median(data_np[self.idx_our])
        max_our = np.max(data_np[self.idx_our])
        plt.axhline(y=median_our, color="black", linestyle="--", alpha=0.3)
        ax.text(
            self.idx_our + 1,
            max_our,
            f"Our: {median_our:.3f} m",
            ha="center",
            va="bottom",
            fontsize="small",
            color="grey",
        )

        ax.violinplot(dataset=data_np.T, showmeans=False, showmedians=True)
        ax.set_xticks(np.arange(1, num_models_selected + 1))
        ax.set_xticklabels(
            itemgetter(*self.models_selected)(self.labels_short),
            rotation=45,
            ha="right",
            fontsize="small",
        )
        ax.set_ylabel(r"Center line deviation [m]")
        # ax.set_ylim([0.04, 0.1]) # [m]
        ax.xaxis.set_minor_locator(ticker.NullLocator())  # Make minor x-ticks invisible
        ax.grid(True, which="major", axis="x", linestyle="--", linewidth=0.1)
        ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.1)

        # Add text to shown the median of M1
        if (self.num_models_selected == 7) or (self.num_models_selected == 10):
            median_M1 = np.median(data_np[0])
            ax.text(
                1,
                0.043,
                f"Median:\n{median_M1:.2f} m",
                ha="center",
                va="bottom",
                fontsize="small",
                color="grey",
            )

        # Save figure
        plt.tight_layout()
        if self.parameters.is_save_eval_results:
            path_save_eval_fig = f"{self.where_to_save_eva_results}/{timestamp}_{self.parameters.scenario_type}_deviation_average.pdf"
            plt.savefig(
                path_save_eval_fig,
                format="pdf",
                bbox_inches="tight",
                pad_inches=0,
                dpi=300,
            )
            path_save_eval_fig = f"{self.where_to_save_eva_results}/{timestamp}_{self.parameters.scenario_type}_deviation_average.png"
            plt.savefig(
                path_save_eval_fig,
                format="png",
                bbox_inches="tight",
                pad_inches=0,
                dpi=600,
            )
            print(
                colored(f"[INFO] A fig has been saved under", "black"),
                colored(f"{path_save_eval_fig}", "blue"),
            )

        ###############################
        ## Fig 4 - average velocity
        ###############################
        plt.clf()
        fig, ax = plt.subplots(figsize=(3.5, 1.6))
        data_np = self.velocity_average.numpy()

        # Plot a horizontal line indicating the median of our model
        median_our = np.median(data_np[self.idx_our])
        max_our = np.max(data_np[self.idx_our])
        plt.axhline(y=median_our, color="black", linestyle="--", alpha=0.3)
        ax.text(
            self.idx_our + 1,
            max_our,
            f"Our: {median_our:.2f} m/s",
            ha="center",
            va="bottom",
            fontsize="small",
            color="grey",
        )

        ax.violinplot(dataset=data_np.T, showmeans=False, showmedians=True)
        ax.set_xticks(np.arange(1, num_models_selected + 1))
        ax.set_xticklabels(
            itemgetter(*self.models_selected)(self.labels_short),
            rotation=45,
            ha="right",
            fontsize="small",
        )
        ax.set_ylabel(r"Speed [m/s]")
        # ax.set_ylim([0.6, 0.8]) # [m/s]
        ax.yaxis.set_major_formatter(
            ticker.FormatStrFormatter("%.2f")
        )  # Set y-axis tick labels to have two digits after the comma
        ax.xaxis.set_minor_locator(ticker.NullLocator())  # Make minor x-ticks invisible
        ax.grid(True, which="major", axis="x", linestyle="--", linewidth=0.1)
        ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.1)

        # # Add text to shown the median of M1
        # median_M1 = np.median(data_np[0])
        # ax.text(1, 0.71, f'Median:\n{median_M1:.2f} m/s', ha='center', va='bottom', fontsize='small', color='grey')

        # Save figure
        plt.tight_layout()
        if self.parameters.is_save_eval_results:
            path_save_eval_fig = f"{self.where_to_save_eva_results}/{timestamp}_{self.parameters.scenario_type}_velocity_average.pdf"
            plt.savefig(
                path_save_eval_fig,
                format="pdf",
                bbox_inches="tight",
                pad_inches=0,
                dpi=300,
            )
            path_save_eval_fig = f"{self.where_to_save_eva_results}/{timestamp}_{self.parameters.scenario_type}_velocity_average.png"
            plt.savefig(
                path_save_eval_fig,
                format="png",
                bbox_inches="tight",
                pad_inches=0,
                dpi=600,
            )
            print(
                colored(f"[INFO] A fig has been saved under", "black"),
                colored(f"{path_save_eval_fig}", "cyan"),
            )
        # plt.show()

        print(
            colored(f"Logging had been saved under ", "grey"),
            colored(self.where_to_save_logging, "blue"),
        )

    def _init_eva_matrices(self):
        """
        Initialize evaluation matrices.
        """
        self.velocity_average = torch.zeros(
            (self.num_models_selected, self.parameters.num_vmas_envs),
            device=self.parameters.device,
            dtype=torch.float32,
        )
        # collision_rate_sum = torch.zeros((num_models_selected, parameters.num_vmas_envs), device=parameters.device, dtype=torch.float32)
        self.collision_rate_with_agents = torch.zeros(
            (self.num_models_selected, self.parameters.num_vmas_envs),
            device=self.parameters.device,
            dtype=torch.float32,
        )
        self.collision_rate_with_lanelets = torch.zeros(
            (self.num_models_selected, self.parameters.num_vmas_envs),
            device=self.parameters.device,
            dtype=torch.float32,
        )
        self.distance_ref_average = torch.zeros(
            (self.num_models_selected, self.parameters.num_vmas_envs),
            device=self.parameters.device,
            dtype=torch.float32,
        )
        self.episode_reward = torch.zeros(
            (self.num_models_selected, self.parameters.n_iters),
            device=self.parameters.device,
            dtype=torch.float32,
        )

    def run_evaluation(self):
        """
        Main method to run evaluation over all provided model paths.
        """
        self.model_idx = 0

        for model_i in self.models_selected:
            print("------------------------------------------")
            print(
                colored("-- [INFO] Model ", "black"),
                colored(f"{model_i + 1}", "blue"),
                colored(f"({self.labels[model_i]})", color="grey"),
            )
            print("------------------------------------------")

            self.model_i = model_i
            self.model_i_path = self.model_paths[model_i]

            self._load_parameters()
            self._adjust_parameters()

            if self.model_idx == 0:
                self._init_eva_matrices()  # Only need to be done once

            self.episode_reward[self.model_idx, :] = torch.tensor(
                [self.saved_data.episode_reward_mean_list]
            )

            self._evaluate_model_i()

            self.model_idx += 1

        if not self.parameters.is_save_simulation_video:
            self.plot()

    @staticmethod
    def format_array(arr):
        return ", ".join([f"{x:.2f}" for x in arr])
