# A Sample Efficient and Generalizable Multi-Agent Reinforcement Learning Framework for Motion Planning
This repository provides the full code for the paper “A Sample Efficient and Generalizable Multi-Agent Reinforcement Learning Framework for Motion Planning”.
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

## How to Use
- Run `/training_mappo_cavs.py`. During training, all the intermediate models that have higher performance than the saved one will be saved.
- After training, run `/testing_mappo_cavs.py` to test your model. Adjust the parameter `path` therein to tell which folder the target model was saved.

`/scenarios/road_traffic.py` defines the training environment, such as observation function, reward function, etc. Besides, it provides an interactive interface, which also visualizes the environment. Use `arrow keys` to control agents and use the `tab key` to switch between agents.

## Reproduce Experiment Results
Checkpoints for trained models to reproduce the experiment results in the paper are provided in the folder `/checkpoints/`. Run evaluation script `utilities/Evaluation.py`, which will automatically load the checkpoints and run the simulations.

You can also run `/testing_mappo_cavs.py` to intuitively evaluate the trained models. Adjust the parameter `path` therein to tell which folder the target model was saved.

## References
We would be grateful if you would refer to the paper below if you find this repository helpful.


<summary>
J. Xu, P. Hu, B. Alrifaee, "A Sample Efficient and Generalizable Multi-Agent Reinforcement Learning Framework for Motion Planning," ResearchGate, Preprint, 2024, doi: 10.13140/RG.2.2.28845.70886
<br>

<!-- icons from https://simpleicons.org/ -->
[![Paper](https://img.shields.io/badge/Preprint-Paper-00629B)](http://dx.doi.org/10.13140/RG.2.2.28845.70886)
[![Video](https://img.shields.io/badge/-Video-FF0000?logo=YouTube)](https://youtu.be/36gCamoqEcA)
[![Repository](https://img.shields.io/badge/-GitHub-181717?logo=GitHub)](https://github.com/cas-lab-munich/generalizable-marl)
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
    doi     = {10.13140/RG.2.2.28845.70886}}
}
```

</p>