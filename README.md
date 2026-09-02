# Warehouse Robot Reinforcement Learning

_43008 - Project #57_

## Planned architecture

1. **Foundation** — shared configuration, positions, actions, and tile types.
2. **Simulation** — generates the warehouse, robot, boxes, obstacles, and finish zones.
3. **Gymnasium environment** — exposes `reset()`, `step()`, action/observation spaces, and rewards.
4. **RL agents** — trains and evaluates algorithms such as Q-learning and DQN.

The package boundaries let team members work independently without placing all logic in one file.

## Setup

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
python -m pip install -e ".[dev]"
python -m warehouse_robot
pytest
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Project layout

```text
43008-project/
├── src/warehouse_robot/
│   ├── core/          # shared foundation
│   ├── simulation/    # milestone 2
│   ├── environment/   # milestone 3 (Gymnasium)
│   └── agents/        # milestone 4 (RL algorithms)
└── tests/
```

## Team workflow

- Create a branch per feature, for example `simulation/grid-generation`.
- Keep environment-independent rules in `core`.
- Add tests with every feature.
- Do not commit `.venv`, caches, trained models, or generated run logs.
