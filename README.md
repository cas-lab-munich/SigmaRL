# MARL for CAVs

## Install
Open a terminal and navigate to where you want to clone this repo. Then run the following commands:
```
git clone git@git.rwth-aachen.de:CPM/Project/jianye/software/marl_for_cavs.git
cd marl_for_cavs/
conda create --name your-env-name python=3.9
conda activate your-env-name
pip install -r requirements.txt
conda list
```
If you are a developer of this repo, set your username and email in terminal using
```
git config --global user.name "first_name last_name"
git config --global user.email "you-email-address"
```

<img src="assets/generalizable-MARL.gif" width="360" height="320" />

## How to Use
- Run `/training_mappo_cavs.py`. During training, all the intermediate models that have higher performance than the saved one will be saved.
- After training, run `/testing_mappo_cavs.py` to test your model. Adjust the parameter `path` therein to tell which folder the target model was saved.

`/scenarios/road_traffic.py` defines the training environment, such as observation function, reward function, etc. Besides, it provides an interactive interface, which also visualizes the environment. Use `arrow keys` to control agents and use the `tab key` to switch between agents.

## Important
- Do not work directly on `main` branch. Instead, create your own branch using `git checkout -b ab-dev`, where "a" is the first letter of your first name and "b" is the first letter of your last name. You can also replace "dev" with another keyword (or other keywords) that reflects the purpose of your branch.
- The `main` branch must be a stable branch, meaning that you must **make sure** your commits will not break it before pushing them to it. At least `training_mappo_cavs.py` can be ran without any issues in your own branch.
- Before you push commits to `main` branch, get a permission from your advisor.
- Write code comments! Write code comments!! Write code comments!!!

## References
We would be grateful if you would refer to the paper below if you find this repository helpful.


<summary>
J. Xu, P. Hu, B. Alrifaee, "A Sample Efficient and Generalizable Multi-Agent Reinforcement Learning Framework for Motion Planning," ResearchGate, Preprint, 2024, doi: 10.13140/RG.2.2.28845.70886
<br>

<!-- icons from https://simpleicons.org/ -->
<a href="http://dx.doi.org/10.13140/RG.2.2.28845.70886" target="_blank"><img src="https://img.shields.io/badge/Preprint-Paper-00629B"></a>
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
    title   = {A Sample Efficient and Generalizable Multi-Agent Reinforcement Learning Framework for Motion Planning},
    year    = {2024},
    doi     = {10.13140/RG.2.2.28845.70886}
}
```

### Reproduce Experiment Results
Checkpoints for trained models to reproduce the experiment results in the paper are provided in the folder `/checkpoints/`. Run evaluation script `utilities/Evaluation.py`, which will automatically load the checkpoints and run the simulations.

You can also run `/testing_mappo_cavs.py` to intuitively evaluate the trained models. Adjust the parameter `path` therein to tell which folder the target model was saved.

</p>

## Logs
- [2024-07-10] Our scenario is now available as a MARL benchmark scenario in VMAS (see <a href="https://github.com/proroklab/VectorizedMultiAgentSimulator/releases/tag/1.4.2" target="_blank">here</a>)! 
- [2024-05-07] Version for the submission to ITSC 2024 on May 1, 2024. See branch `ITSC24`.
- [2024-04-10] Traffic-road scenario works. See branch `traffic-road` and save files in directory `outputs_saved/traffic_road`.
- [2024-01-26] Line-path-tracking scenario works. See branch `path-tracking-scenarios` and saved files in directory `outputs_saved/line`.
- [2024-01-26] Sine-path-tracking scenario works. See branch `path-tracking-scenarios` and saved files in directory `outputs_saved/sine`.
- [2024-01-25] Circle-path-tracking scenario works. See branch `circle-path-tracking` and saved files in directory `outputs_saved/circle`.
- [2024-01-22] Sine-path-tracking scenario works. See branch `sine-path-tracking` and saved files in directory `outputs_saved/sine`.
