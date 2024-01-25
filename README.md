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
1. Open `multiagent_ppo_cavs.py` (you can find it at the root directory)
2. Scroll down to the bottom
3. Adjust the parameter `scenario_name`. You can find all available scenarios at the folder `scenarios`
    - Currently I am working on `scenarios/car_like_robots_path_tracking.py`, where multiple simply scenarios are presented. 
    - Final goal is to have a well-trained MARL model that masters the scenario in `scenarios/car_like_robots_road_traffic.py`.

## Logs
- [2024-01-25] Circle-path-tracking scenario works. See branch `circle-path-tracking` and saved files in directory `outputs_saved/sine`.
- [2024-01-22] Sine-path-tracking scenario works. See branch `sine-path-tracking` and saved files in directory `outputs_saved/sine`.

## Todos
- [ ] Add small random noise in observations of intermediate goals to avoid overfitting?
- [x] Define the world dimensions. Terminate an iteration with a high negative reward if agent moves outside the world.
- [x] Set position normalizer based on the world dimensions
- [ ] **memory is all you need?**
- [ ] Local coordinate system for path-tracking scenarios
- [ ] When using `interX` check collisions between an agent and its lanelet boundaries, include one point from previous time step (e.g., the position of the center of the agent) to cosider the case that the sample time is too high or the velocity of the agent of so high such that the agent can "jump" outside of the boundary
- [ ] Would putting actions in observation increase performance? If yes, how much?
- [ ] Turn off the drag/friction defaulted in vmas (set `world._drag = 0` in the `make_world` function of the custom scenario)?
- [ ] Do not simply use a constant for the goal reward. Adjust it dynamically according to how well goals are reached.
- [ ] Search "TODO" in the codes to find more TODOs
- [ ] Use fixed camera view (see how the variable `cam_range` is calculated in `/VectorizedMultiAgentSimulator/vmas/simulator/environment/environment.py`). Open a request in the issue board of vmas?
    - Current workaround: manually change the `self.viewer.set_bounds` in `/vmas/simulator/environment/environment.py` to
        ```
        self.viewer.set_bounds(
            -self.world.x_semidim / 2, # left boundary
            self.world.x_semidim / 2, # right boundary
            -self.world.y_semidim / 2, # bottom boundary
            self.world.y_semidim / 2, # top boundary
        )
        ```
- [ ] Consider different operation systems in "Install" section.
- [x] Requirement.txt
- [x] Partial observation "we obtain a central critic with full-observability (i.e., it will take all the concatenated agent observations as input")
- [x] Only observe nearing agents (inside a fixed radius or parameterized radius) to reduce observation dimension

