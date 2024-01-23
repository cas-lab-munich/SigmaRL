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

## How to Use
1. Open `multiagent_ppo_cavs.py` (you can find it at the root directory)
2. Scroll down to the bottom
3. Adjust the parameter `scenario_name`. You can find all available scenarios at the folder `scenarios`
    - Currently I am working on `scenarios/car_like_robots_path_tracking.py`, where multiple simply scenarios are presented. 
    - Final goal is to have a well-trained MARL model that masters the scenario in `scenarios/car_like_robots_road_traffic.py`.


## Logs
- [2024-01-22] Sine-path-tracking-scenario works. See branch `sine-path-tracking` and saved files in directory `outputs_saved/sine`.

## Todos
- [ ] **memory is all you need?**
- [ ] When using `interX` check collisions between an agent and its lanelet boundaries, include one point from previous time step (e.g., the position of the center of the agent) to cosider the case that the sample time is too high or the velocity of the agent of so high such that the agent can "jump" outside of the boundary
- [ ] Would putting actions in observation increase performance? If yes, how much?
- [ ] Turn off the drag/friction defaulted in vmas (set `world._drag = 0` in the `make_world` function of the custom scenario)?
- [ ] Do not simply use a constant for the goal reward. Adjust it dynamically according to how well goals are reached.
- [ ] Search "TODO" in the codes to find more TODOs
- [ ] Use fixed camera view (see how the variable `cam_range` is calculated in `/VectorizedMultiAgentSimulator/vmas/simulator/environment/environment.py`). Open a request in the issue board of vmas?
    - Current workaround: manually change the `self.viewer.set_bounds` in `/vmas/simulator/environment/environment.py` to
        ```
        self.viewer.set_bounds(
            torch.tensor(-1.5, device=self.world.device, dtype=torch.float32), # left boundary
            torch.tensor(3, device=self.world.device, dtype=torch.float32), # right boundary
            torch.tensor(-2, device=self.world.device, dtype=torch.float32), # bottom boundary
            torch.tensor(2, device=self.world.device, dtype=torch.float32), # top boundary
        )
        ```
- [ ] Consider different operation systems in "Install" section.
- [x] Requirement.txt
- [x] Partial observation "we obtain a central critic with full-observability (i.e., it will take all the concatenated agent observations as input")
- [x] Only observe nearing agents (inside a fixed radius or parameterized radius) to reduce observation dimension

## Gained Knowledge
- Moving-in-high-speed reward is good-to-have, but it should not be a dominant reward, because it may encourages agents to move in a high speed but not along the right path or in the right direction.
- Diviation-from-reference-path penalty may help in reducing the path-tracking error, but may also lead to unstable taining, especially at the begining when the policy makes random actions, which leads to a very high penalty

---
The contents below are only for me.
---

## Notes
- Run the script `use_vmas_env` to test your scenarios or dynamic models
- Markov decision process (MDP) follows the sequence: $S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, ...$
- Function execution order (with a 3-agent scenario) in vmas:
```
make_world()
reset_world_at()
observation()
observation()
info()
reset_world_at()
observation()
info()
done()
process_action() -> First time that actions are generated
step()
step()
reward()
observation()
info()
done()
process_action()
step()
step()
reward()
```
```
process_action()
process_action()
process_action()
reward()
observation()
reward()
observation()
reward()
observation()
done()
<!-- extra_render() -->
```
- Position is updated in `VectorizedMutiAgentSimulator/vmas/simulator/core.py` (line 2374, or search "entity.state.pos = new_pos"):
```
accel = self.force[:, index] / entity.mass
entity.state.vel += accel * self._sub_dt
new_pos = entity.state.pos + entity.state.vel * self._sub_dt
entity.state.pos = new_pos
```
