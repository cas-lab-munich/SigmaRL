# MARL_for_CAVs

## Todos
- [ ] Search "TODO" in the codes to find more TODOs
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
