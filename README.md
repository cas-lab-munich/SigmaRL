# MARL for CAVs

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

If you are a developer of this repo, set your username and email in terminal using
```
git config --global user.name "first_name last_name"
git config --global user.email "you-email-address"
```

## How to Use
- Open and run `training_mappo_cavs.py` (you can find it at the root directory). During training, all the intermediate models that have higher performance than the saved one will be saved.
- `scenarios/road_traffic.py` defines the training environment, such as reward function, observation function, etc. Besides, it provides an interactive interface, which also visualizes the environment. Use `arrow keys` to control agents and `tab key` to switch between agents.

## Important
- Do not work directly on `main` branch. Instead, create your own branch using `git checkout -b ab-dev`, where "a" is the first letter of your first name and "b" is the first letter of your last name. You can also replace "dev" with another keyword (or other keywords) that reflects the purpose of your branch.
- The `main` branch must be a stable branch, meaning that you must **make sure** your commits will not break it before pushing them to it. At least `training_mappo_cavs.py` can be ran without any issues in your own branch.
- Before you push commits to `main` branch, get a permission from your advisor.
- Write code comments! Write code comments!! Write code comments!!!

## Logs
- [2024-04-10] Traffic-road scenario works. See branch `traffic-road` and save files in directory `outputs_saved/traffic_road`.
- [2024-01-26] Line-path-tracking scenario works. See branch `path-tracking-scenarios` and saved files in directory `outputs_saved/line`.
- [2024-01-26] Sine-path-tracking scenario works. See branch `path-tracking-scenarios` and saved files in directory `outputs_saved/sine`.
- [2024-01-25] Circle-path-tracking scenario works. See branch `circle-path-tracking` and saved files in directory `outputs_saved/circle`.
- [2024-01-22] Sine-path-tracking scenario works. See branch `sine-path-tracking` and saved files in directory `outputs_saved/sine`.
