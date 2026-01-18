# DreamerV3 + Finesse LIGO RL System

A model-based reinforcement learning system using DreamerV3 with Finesse physics simulation for LIGO OMC alignment control.

## Overview

This project differs from DeepMind's Deep Loop Shaping by:
- Using **physics-based simulation** (Finesse) instead of system identification
- Using **model-based RL** (DreamerV3) instead of model-free MPO
- Targeting **OMC alignment + squeezing** instead of θ_CHP

## Installation

```bash
conda activate finesse
pip install finesse finesse-ligo gymnasium numpy scipy matplotlib
pip install jax jaxlib flax optax tensorflow-probability
```

## Project Structure

```
src/
├── envs/           # Gymnasium environments
├── dreamer/        # DreamerV3 training scripts
├── models/         # Finesse KatScript models
├── rewards/        # Frequency-domain rewards
└── utils/          # Noise injection, domain randomization
```

## Quick Start

```python
from src.envs import FinesseOMCEnv

env = FinesseOMCEnv()
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)
```

## Training

```bash
python src/dreamer/train.py --steps 1000000 --logdir ./runs
```

## License

MIT
