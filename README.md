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
python train_q_learning.py
python visualise_policy.py
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

## Train across maps and watch a saved policy

From the project root, activate the existing environment:

```bash
source .venv/bin/activate
python train_q_learning.py
python visualise_policy.py
```

Training is terminal-only: by default it runs 3,000 episodes on newly generated,
reachable maps, evaluates once on each of 100 unseen maps, and saves
`models/navigation_q.json` plus `models/navigation_q.evaluation.json`.
Evaluation seeds are outside the training seed range; per-map results and actual
layout seeds are recorded. A fixed initial seed makes the training map stream
reproducible. Training starts fresh each run; saving/loading supports inference,
not resuming an interrupted training session.

The viewer loads the saved model without retraining. It generates a random map,
waits for the red START button, and then displays actions, rewards and deliveries.
Space pauses/resumes; Escape closes. The final state remains for five seconds.
The GUI shows policy execution, not live training updates.

```bash
python train_q_learning.py --episodes 1000 --eval-maps 30
python visualise_policy.py --seed 234567 --fps 12
python train_q_learning.py --width 10 --height 8 --obstacles 4 --boxes 2 --max-steps 400
```

Use `--model PATH` with both scripts for another model location. The viewer uses
the dimensions and object counts stored in the model. A map seed may be advanced
by up to 99 to find a connected layout; the actual seed is printed. Excessively
dense obstacle settings can fail generation. Connectivity does not guarantee
completion within the configured step budget.

### What the agent learns

`NavigationQLearningAgent` is a **hybrid Q-learning baseline**. Breadth-first
search computes distances around obstacles to the nearest remaining box (or an
unused finish while carrying). The learned Q-table uses carrying status, whether
the robot is at a target, and the distance change of each directional move.
These features transfer across layouts; the planner supplies routing guidance.
This is not an end-to-end policy learning to understand raw maps. The original
absolute-state `QLearningAgent` remains available as a fixed-map baseline.

Training uses potential-based distance shaping; evaluation and GUI rewards show
the environment's unshaped rewards. Normal movement costs -1, invalid actions -5,
pickup pays 20, delivery pays 50, and completion adds 100. Merely entering a box
or finish cell no longer produces a repeatable positive reward.

The robot carries one box and can use any unused finish. Nearest-target routing
and the learned feature abstraction **do not guarantee the globally shortest
complete delivery route**. Measure unseen-map success and steps in the saved
report; an exact delivery-order planner is needed to certify optimality. Training
longer alone does not provide that guarantee.

## Evaluation dashboard for demos

```bash
python show_evaluation.py
```

This opens a standalone HTML report in your browser without retraining. Click
**Full screen** for recording a demo. It shows the training episode count,
unseen-map success, actions on successful maps, average reward, invalid actions,
a map-by-map action chart, and expandable seeds/results. The HTML is self-contained
and works offline. Future training runs automatically regenerate it and record
per-episode history for a training-success curve (100-episode blocks). Older
saved results still work, but cannot show a learning curve that was not recorded.

Use `--results PATH` for another evaluation JSON, `--output PATH` to choose where
the HTML goes, or `--no-open` to generate it without launching a browser.
