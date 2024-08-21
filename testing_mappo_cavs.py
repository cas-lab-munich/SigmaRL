# Copyright (c) 2024, Chair of Embedded Software (Informatik 11), RWTH Aachen University.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

from utilities.helper_training import Parameters, SaveData
import os

from vmas.simulator.utils import save_video
import json

from training_mappo_cavs import mappo_cavs

from utilities.constants import SCENARIOS

path = "checkpoints/ITSC24/M0 (our)"

try:
    path_to_json_file = next(
        os.path.join(path, file) for file in os.listdir(path) if file.endswith(".json")
    )  # Find the first json file in the folder
    # Load parameters from the saved json file
    with open(path_to_json_file, "r") as file:
        data = json.load(file)
        saved_data = SaveData.from_dict(data)
        parameters = saved_data.parameters

        # Adjust parameters
        parameters.is_testing_mode = True
        parameters.is_real_time_rendering = True
        parameters.is_save_eval_results = False
        parameters.is_load_model = True
        parameters.is_load_final_model = False
        parameters.is_load_out_td = False
        parameters.max_steps = 1200  # 1200 -> 1 min
        if parameters.is_load_out_td:
            parameters.num_vmas_envs = 32
        else:
            parameters.num_vmas_envs = 1

        parameters.scenario_type = (
            "CPM_entire"  # on_ramp_1, roundabout_1, intersection_1/2/3, CPM_mixed
        )
        parameters.n_agents = SCENARIOS[parameters.scenario_type]["n_agents"]

        parameters.is_save_simulation_video = False
        parameters.is_visualize_short_term_path = False
        parameters.is_visualize_lane_boundary = False
        parameters.is_visualize_extra_info = True

        env, policy, parameters = mappo_cavs(parameters=parameters)

        out_td, frame_list = env.rollout(
            max_steps=parameters.max_steps - 1,
            policy=policy,
            callback=lambda env, _: env.render(
                mode="rgb_array", visualize_when_rgb=True
            ),  # mode \in {"human", "rgb_array"}
            auto_cast_to_device=True,
            break_when_any_done=False,
            is_save_simulation_video=parameters.is_save_simulation_video,
        )
        save_video(f"{path}video", frame_list, fps=1 / parameters.dt)
except StopIteration:
    raise FileNotFoundError("No json file found.")
