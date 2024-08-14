# Multi-Agent Reinforcement Learning for Connected and Automated Vehicles
<div>
<img src="assets/figs/cpm_entire.gif" width="360" height="320" />
<br>
<img src="assets/figs/intersection_2.gif" height="160"/>
<img src="assets/figs/on_ramp_1.gif" height="160"/>
<img src="assets/figs/roundabout_1.gif" height="160"/>
<br>
Fig. 1: Video demonstrations (speed x2). All scenarios are listed in the variable `SCENARIOS` in `utilities/constants.py`.
</div>

## Install
Open a terminal and navigate to where you want to clone this repo. Then run the following commands (assuming <a href="https://conda.io/projects/conda/en/latest/index.html" target="_blank">conda</a> is installed):
```
git clone git@git.rwth-aachen.de:CPM/Project/jianye/software/marl_for_cavs.git
cd marl_for_cavs/
conda create --name your-env-name python=3.9
conda activate your-env-name
pip install -r requirements.txt
```
If you are a developer of this repo, set your username and email in terminal using
```
git config --global user.name "first_name last_name"
git config --global user.email "you-email-address"
```

## How to Use
- Run `/training_mappo_cavs.py`. During training, all the intermediate models that have higher performance than the saved one will be saved.
- After training, run `/testing_mappo_cavs.py` to test your model. Adjust the parameter `path` therein to tell which folder the target model was saved.

`/scenarios/road_traffic.py` defines the training environment, such as observation function and reward function. Besides, it provides an interactive interface, which also visualizes the environment. Use `arrow keys` to control agents and use the `tab key` to switch between agents. Adjust the parameter `scenario_type` to choose a scenario. All available scenarios are listed in the variable `SCENARIOS` in `utilities/constants.py`.



## OpenStreetMap Support
We support maps customized in <a href="https://josm.openstreetmap.de/" target="_blank">JOSM</a>. Follow these steps:
- Install and open JOSM, click the green download button
- Zoom in and find an empty area (as empty as possible)
- Select the area by drawing a rectangle
- Click "Download"
- Now you will see a new window. Make sure there is no element. Otherwise, redo the above steps.
- Customize lanes. Note that all lanes you draw are considered center lines. You do not need to draw left and right boundaries, since they will be determined automatically later by our script with a given width.
- Save the osm file and store it at `assets/maps`. Give it a name.
- Go to `utilities/constants.py` and create a new dictionary for it. You should at least give the value for the key "map_path".
- Go to `utilities/parse_osm.py`. Adjust the parameters `scenario_type` (the name of the new map), `width` (lane width), and `scale` (a scale to convert GPS system) to meet your requirements.

## Important
- Do not work directly on `main` branch. Instead, create your own branch using `git checkout -b ab-dev`, where "a" is the first letter of your first name and "b" is the first letter of your last name. You can also replace "dev" with another keyword (or other keywords) that reflects the purpose of your branch.
- The `main` branch must be a stable branch, meaning that you must **make sure** your commits will not break it before pushing them to it. At least `training_mappo_cavs.py` can be ran without any issues in your own branch.
- Before you push commits to `main` branch, get a permission from your advisor.
- Write code comments! Write code comments!! Write code comments!!!

## References
We would be grateful if you would refer to the paper below if you find this repository helpful.


<summary>
Jianye Xu, Pan Hu, Bassam Alrifaee, "SigmaRL: A Sample-Efficient and Generalizable Multi-Agent Reinforcement Learning Framework for Motion Planning," ResearchGate, Preprint, 2024, doi: 10.13140/RG.2.2.24505.17769
<br>

<!-- icons from https://simpleicons.org/ -->
<a href="http://dx.doi.org/10.13140/RG.2.2.24505.17769" target="_blank"><img src="https://img.shields.io/badge/Preprint-Paper-00629B"></a>
<a href="https://youtu.be/36gCamoqEcA" target="_blank"><img src="https://img.shields.io/badge/-Video-FF0000?logo=YouTube"></a>
<a href="https://github.com/cas-lab-munich/generalizable-marl" target="_blank"><img src="https://img.shields.io/badge/-GitHub-181717?logo=GitHub"></a>

</summary>

<summary>
References in Bibtex format
</summary>
<p>

```bibtex
@article{xu2024sample,
    author  = {Jianye Xu and Pan Hu and Bassam Alrifaee},
    title   = {SigmaRL: A Sample-Efficient and Generalizable Multi-Agent Reinforcement Learning Framework for Motion Planning},
    year    = {2024},
    doi     = {10.13140/RG.2.2.24505.17769},
    note    = {Accepted by 27th IEEE International Conference on Intelligent Transportation Systems (IEEE ITSC 2024)},
}
```

### Reproduce Experiment Results
To reprodece the experiment results of the paper, run `utilities/evaluation_ITSC24.py`. Checkpoints for trained models will be loaded automatically.

You can also run `/testing_mappo_cavs.py` to intuitively evaluate the trained models. Adjust the parameter `path` therein to tell which folder the target model was saved.

</p>

## Logs
- [2024-08-14] ITSC 24 final version. See branch `ITSC24-final`.
- [2024-08-14] We support customized maps in OpenStreetMap now!
- [2024-07-10] Our scenario is now available as a MARL benchmark scenario in VMAS (see <a href="https://github.com/proroklab/VectorizedMultiAgentSimulator/releases/tag/1.4.2" target="_blank">here</a>)! 
- [2024-05-07] Version for the submission to ITSC 2024 on May 1, 2024. See branch `ITSC24`.
- [2024-04-10] Traffic-road scenario works. See branch `traffic-road` and save files in directory `outputs_saved/traffic_road`.
- [2024-01-26] Line-path-tracking scenario works. See branch `path-tracking-scenarios` and saved files in directory `outputs_saved/line`.
- [2024-01-26] Sine-path-tracking scenario works. See branch `path-tracking-scenarios` and saved files in directory `outputs_saved/sine`.
- [2024-01-25] Circle-path-tracking scenario works. See branch `circle-path-tracking` and saved files in directory `outputs_saved/circle`.
- [2024-01-22] Sine-path-tracking scenario works. See branch `sine-path-tracking` and saved files in directory `outputs_saved/sine`.
