# Latest PettingZoo Wrappers for SMAC and SMACv2

We wrap [SMAC](https://github.com/oxwhirl/smac) and [SMACv2](https://github.com/oxwhirl/smacv2) with parallel apis of PettingZoo.

Note that we include the modifications of SMAC and SMACv2 in [MAPPO](https://github.com/marlbenchmark/on-policy/tree/main) for reproducing the SOTA results. 

**Modifications in Game Mechanism**

- **Suppressing Annoying outputs from PySC2.**
- Fully control the randomness in SMACv2.
- Fix SMACv1 reward hacking, following https://github.com/oxwhirl/smac/pull/76.


**Example**

### SMACv2 rendering

Pass `render_mode="rgb_array"` or `render_mode="human"` to
`smacv2_pettingzoo_v1.parallel_env()`, then call `reset()` before `render()`.
The default `None` does not initialize a renderer. The mode is fixed for the
lifetime of the environment. SMACv1 rendering is not implemented.

```python
from smac_pettingzoo import smacv2_pettingzoo_v1

env = smacv2_pettingzoo_v1.parallel_env(
    "10gen_terran_10_vs_10",
    render_mode="rgb_array",
    smacv2_env_args={"window_size": (1280, 720)},
)
try:
    env.reset(seed=42)
    frame = env.render()  # Independent uint8 RGB array: (height, width, 3).
finally:
    env.close()
```

This is a top-down schematic, not SC2's native 3D graphics. Render after each
step to obtain consecutive frames; rendering never advances the simulation.
The renderer is maintained locally (adapted from SMACv2 under its MIT license);
the `smacv2` Python package is not required.
The full map is fitted inside `window_size` without stretching or cropping,
including margins outside the playable area. The returned dimensions depend
on the map aspect ratio. Large-map rendering does not itself add map-loading
support. `rgb_array` works off-screen; `human` requires a graphical display
and returns `None`. Closing the viewer does not terminate the battle.
Use one human viewer per process and a separate evaluation environment rather
than enabling windows in every training worker.

### Selecting a larger SMACv2 map

Pass `smacv2_env_args={"map_file": "SMAC_Maps/32x32_flat.SC2Map", "episode_limit": 400}`
to select a physical map relative to `StarCraftII/Maps` (absolute paths also
work). Omit `map_file` to retain the scenario's registered map. The scenario
name, e.g. `10gen_terran_50_vs_50`, still determines the race and team sizes;
`map_file` does not change those rules or automatically resize spawn regions.
Set the `start_positions` distribution to match the map's playable area and
use `fixed_teams` when unit composition must remain constant across episodes.
The chosen map must include SMACv2's nine custom RL units and episode-control
triggers, and support two players. Missing custom units are rejected before
spawning. This check does not validate triggers or spawn coordinates.
The ordinary `Melee/Flat64.SC2Map` is **not** compatible: it needs a separate
SMACv2 adaptation, and its playable 64x64 region starts at (12, 10), not (0, 0).

```python
from co_mas.test import sample_action
from smac_pettingzoo import smacv2_pettingzoo_v1, smacv1_pettingzoo_v1
from loguru import logger

env = smacv1_pettingzoo_v1.parallel_env("8m")
obs, info = env.reset(seed=42)
step = 0
while True:
    obs, _, terminated, truncated, info = env.step(
        {agent: sample_action(env, obs, agent, info) for agent in env.agents}
    )
    step += 1
    if len(env.agents) <= 0:
        logger.debug(f"step {step}, terminated: {terminated} truncated: {truncated}")
        break
env.close()

env = smacv2_pettingzoo_v1.parallel_env("10gen_terran_10_vs_10")
obs, info = env.reset(seed=42)
step = 0
while True:
    obs, _, terminated, truncated, info = env.step(
        {agent: sample_action(env, obs, agent, info) for agent in env.agents}
    )
    step += 1
    if len(env.agents) <= 0:
        logger.debug(f"step {step}, terminated: {terminated} truncated: {truncated}")
        break
env.close()
```
