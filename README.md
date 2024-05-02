# Sample-Efficient-and-Generalizable-MARL
This reposity provides the full code for paper “Sample Efficient and Generalizable Multi-Agent Reinforcement
Learning Framework for Motion Planning”.
## Install
Open a terminal, navigate to where you want to clone this repo. Then run the following commands:
```
git clone git@git.rwth-aachen.de:CPM/Project/jianye/software/marl_for_cavs.git
cd marl_for_cavs/
conda create --name your-env-name python=3.9
conda activate your-env-name
pip install -r requirements.txt
conda list
```

## How to Use
- Open and run `training_mappo_cavs.py` (you can find it at the root directory). During training, all the intermediate models that have higher performance than the saved one will be saved.
- `scenarios/road_traffic.py` defines the training environment, such as observation function, reward function, etc. Besides, it provides an interactive interface, which also visualizes the environment. Use `arrow keys` to control agents and `tab key` to switch between agents.

## Reproduce experiment results
Checkpoints for trained models to reproduce the experiment results in the paper are provided in the folder `checkpoints/`. Run evaluation script `utilities/Evaluation.py`, which will automatically load the checkpoints and run the simulations.

## References
We would be grateful if you would refer to the paper below if you find this repository helpful (TODO).
