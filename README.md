# MARL_for_CAVs

## Install
Open a terminal, navigate to where you want to clone this repo. Then copy and run the following commands:
```
git clone git@git.rwth-aachen.de:CPM/Project/jianye/software/marl_for_cavs.git
cd marl_for_cavs/
conda create --name your-env-name python=3.9
conda activate your-env-name
pip install -r requirements.txt
conda list
```

If you are a developer of this repo, set your user name and email in terminal using
```
git config --global user.name "first_name last_name"
git config --global user.email "you-email-address"
```

## How to Use
1. Open `mppo_cavs.py` (you can find it at the root directory)
2. Scroll down to the bottom
3. Adjust the parameter `scenario_name`. You can find all available scenarios at the folder `scenarios`
    - Currently I am working on `scenarios/path_tracking.py`, where multiple simply scenarios are presented. 
    - Final goal is to have a well-trained MARL model that masters the scenario in `scenarios/road_traffic.py`.


## Logs
- [2024-01-26] Line-path-tracking scenario works. See branch `path-tracking-scenarios` and saved files in directory `outputs_saved/line`.
- [2024-01-26] Sine-path-tracking scenario works. See branch `path-tracking-scenarios` and saved files in directory `outputs_saved/sine`.
- [2024-01-25] Circle-path-tracking scenario works. See branch `circle-path-tracking` and saved files in directory `outputs_saved/circle`.
- [2024-01-22] Sine-path-tracking scenario works. See branch `sine-path-tracking` and saved files in directory `outputs_saved/sine`.
