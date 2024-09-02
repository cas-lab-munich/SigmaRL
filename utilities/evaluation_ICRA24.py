# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

script_dir = os.path.dirname(__file__)  # Directory of the current script
project_root = os.path.dirname(script_dir)  # Project root directory
if project_root not in sys.path:
    sys.path.append(project_root)

from utilities.constants import SCENARIOS
from utilities.evaluation_base import Evaluation

model_paths = [
    "outputs/our_not_OM/",
    "outputs/test_OM/",
]

legends = [
    r"$M_{" + path.split("/")[-2][1] + r"}$ " + path.split("/")[-2][2:]
    for path in model_paths
]

render_titles = [path.rsplit("/", 2)[-2] for path in model_paths]

video_names = [path.rsplit("/", 2)[-2][0:2] for path in model_paths]

scenario_types = [
    "CPM_entire",
    "intersection_2",
    "on_ramp_1",
    "roundabout_1",
]

for i_scenario in scenario_types:
    print("*****************************************")
    print("*****************************************")
    print(f"[INFO] Scenario: {i_scenario}")
    print("*****************************************")
    print("*****************************************")

    n_agents = SCENARIOS[i_scenario]["n_agents"]

    evaluator = Evaluation(
        scenario_type=i_scenario,  # Specify which scenario should be used to do the evaluation. One of {"CPM_entire", "intersection_2", "on_ramp_1", "roundabout_1"}
        model_paths=model_paths,
        num_agents=n_agents,  # Number of agents to be used in the evaluation
        simulation_steps=1200,  # Number of time steps of each simulation. 1200 -> 1 min if sample time is 50 ms
        where_to_save_eva_results=f"outputs/ICRA24/eva_{i_scenario}",
        where_to_save_logging=f"outputs/ICRA24/log.txt",
        models_selected=[],  # Leave empty if all the models should be evaluated
        legends=legends,
        render_titles=render_titles,
        num_simulations_per_model=32,
        is_render=False,
        is_save_simulation_video=False,
        video_names=video_names,
    )

    evaluator.run_evaluation()
