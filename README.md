# SigmaRL: A Sample-Efficient and Generalizable Multi-Agent Reinforcement Learning Framework for Motion Planning
<!-- icons from https://simpleicons.org/ -->
<a href="http://dx.doi.org/10.13140/RG.2.2.24505.17769" target="_blank"><img src="https://img.shields.io/badge/Preprint-Paper-00629B"></a>
<a href="https://youtu.be/36gCamoqEcA" target="_blank"><img src="https://img.shields.io/badge/-Video-FF0000?logo=YouTube"></a>
<a href="https://github.com/cas-lab-munich/generalizable-marl" target="_blank"><img src="https://img.shields.io/badge/-GitHub-181717?logo=GitHub"></a>

This repository provides the full code for the paper "SigmaRL: A Sample-Efficient and Generalizable Multi-Agent Reinforcement Learning Framework for Motion Planning".
This paper was accepted by the 27th IEEE International Conference on Intelligent Transportation Systems (IEEE ITSC 2024).

We use <a href="https://github.com/proroklab/VectorizedMultiAgentSimulator" target="_blank">VMAS</a>, a vectorized differentiable simulator designed for efficient Multi-Agent Reinforcement Learning benchmarking, as our simulator und customize a reinforcement learning environment to suit the case of our Cyber-Physical Mobility Lab (<a href="https://cpm.embedded.rwth-aachen.de/">CPM Lab</a>).
Besides, we also support maps handcrafted in <a href="https://josm.openstreetmap.de/" target="_blank">OpenStreetMap</a>. Below you will find detailed guidance to create your **OWN** map.

<div>
<img src="assets/figs/generalizable-MARL.gif" width="180" height="160" />
<img src="assets/figs/roundabout_1.gif" height="160"/>

<img src="assets/figs/intersection_2.gif" height="160"/>
<img src="assets/figs/on_ramp_1.gif" height="160"/>

<p>Fig. 1: Video demonstrations (speed x2). All scenarios are listed in the variable `SCENARIOS` in `utilities/constants.py`.</p>
</div>

## Install (tested in macOS and Windows)
Open a terminal and navigate to where you want to clone this repo. Then run the following commands:
```
git clone https://github.com/cas-lab-munich/generalizable-marl.git
cd generalizable-marl/
conda create --name your-env-name python=3.9
conda activate your-env-name
pip install -r requirements.txt
```

## How to Use
- Run `/training_mappo_cavs.py`. During training, all the intermediate models that have higher performance than the saved one will be saved.
- After training, run `/testing_mappo_cavs.py` to test your model. Adjust the parameter `path` therein to tell which folder the target model was saved.

`/scenarios/road_traffic.py` defines the training environment, such as observation function and reward function. Besides, it provides an interactive interface, which also visualizes the environment. Use `arrow keys` to control agents and use the `tab key` to switch between agents. Adjust the parameter `scenario_type` to choose a scenario. All available scenarios are listed in the variable `SCENARIOS` in `utilities/constants.py`.

### Reproduce Experiment Results
To reprodece the experiment results of the paper, run `utilities/evaluation_ITSC24.py`. Checkpoints for trained models will be loaded automatically.

You can also run `/testing_mappo_cavs.py` to intuitively evaluate the trained models. Adjust the parameter `path` therein to tell which folder the target model was saved.

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

### News
- [2024-08-14] We support customized maps in OpenStreetMap now!
- [2024-07-10] Our scenario is now available as a MARL benchmark scenario in VMAS for Connected and Automated Vehicles (CAVs) (see <a href="https://github.com/proroklab/VectorizedMultiAgentSimulator/releases/tag/1.4.2">here</a>)!

## References
We would be grateful if you would refer to the paper below if you find this repository helpful.

<summary>
Jianye Xu, Pan Hu, Bassam Alrifaee, "SigmaRL: A Sample-Efficient and Generalizable Multi-Agent Reinforcement Learning Framework for Motion Planning," ResearchGate, Preprint, 2024, doi: 10.13140/RG.2.2.24505.17769
<br>
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
</p>