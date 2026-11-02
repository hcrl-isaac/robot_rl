# CLAUDE.md — robot_rl

The workspace's RL package (a fork of rsl_rl): PPO, SAC and FB-CPR algorithms, their runners, models and storage.
It is one of the sibling repos the manager composes under `resources/`; the workspace-wide guide (install, running,
compute, conventions, skills) is the manager's `CLAUDE.md` (`../../CLAUDE.md` from this repo). Read that first.

## Layout

- `robot_rl/algorithms/`: the update rules (PPO, SAC, FB-CPR, distillation).
- `robot_rl/runners/`: training loops that drive an env and an algorithm; `hcrl_isaaclab/robot_rl/` holds the
  runner cfgs that bind Isaac Lab envs to them.
- `robot_rl/models/`, `robot_rl/modules/`: networks; `robot_rl/storage/`: rollout and replay buffers.
- `robot_rl/env/`, `robot_rl/extensions/`, `robot_rl/utils/`: the env interface, optional features, logging
  (W&B and TensorBoard writers).

## Rules specific to this repo

- Isaac Lab imports stay lazy (inside the function that needs them), so the package imports and tests without
  Isaac Sim.
- torch is pinned to Isaac Lab 3's lock; never change the pin on its own.
- Multi-GPU is iteration scaling (N ranks, budget/N); learning rates scale with the rank count as each runner
  documents.

## Checks

```bash
pytest tests                       # CPU, no Isaac Sim needed (CI also builds the Sphinx docs)
```
