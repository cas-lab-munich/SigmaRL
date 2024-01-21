# MARL_for_CAVs

## Todos
- [ ] Do we need to turn off the drag/friction defaulted in vmas (set `world._drag = 0` in the `make_world` function of the custom scenario)?
- [ ] Do not simply use a constant for the goal reward. Adjust it dynamically according to how well goals are reached.
- [ ] Search "TODO" in the codes to find more TODOs
- [ ] Use fixed camera view (see how the variable `cam_range` is calculated in `/VectorizedMultiAgentSimulator/vmas/simulator/environment/environment.py`)
- [x] Partial observation "we obtain a central critic with full-observability (i.e., it will take all the concatenated agent observations as input"
- [x] Only observe nearing agents (inside a fixed radius or parameterized radius) to reduce observation dimension

## Notes
- Run the script `use_vmas_env` to test your scenarios or dynamic models
- Function execution order (with a 3-agent scenario):
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
extra_render()
```
- Position is updated in `VectorizedMutiAgentSimulator/vmas/simulator/core.py` (line 2374, or search "entity.state.pos = new_pos"):
```
accel = self.force[:, index] / entity.mass
entity.state.vel += accel * self._sub_dt
new_pos = entity.state.pos + entity.state.vel * self._sub_dt
entity.state.pos = new_pos
```
