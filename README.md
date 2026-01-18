# LIGO RL: Physics-Based Reinforcement Learning for Gravitational Wave Detectors

A research project exploring model-based reinforcement learning with Finesse physics simulation for LIGO alignment control.

## Overview

This project explores a complementary approach to the Deep Loop Shaping paper (arxiv:2509.14016) by Google DeepMind, Caltech, and GSSI. We aim to investigate whether physics-based simulation can enhance RL-based control for gravitational wave detectors.

## Key Differences from Deep Loop Shaping

| Aspect | Deep Loop Shaping | Our Approach |
|--------|-------------------|--------------|
| **Simulation** | Linear identified model | Finesse physics simulation |
| **Target** | θ_CHP (common-hard-pitch) | Output Mode Cleaner (OMC) |
| **Algorithm** | MPO (model-free) | SAC / DreamerV3 (model-based) |
| **Architecture** | MLP + Dilated Conv | Same + Transformer option |

### What We Aim to Explore

1. **Physics Simulation**: Whether training on Finesse captures dynamics that linear models miss
2. **Model-Based RL**: If DreamerV3's world model improves sample efficiency
3. **Different Target**: OMC alignment + squeezing optimization (unexplored in RL)
4. **Hybrid Rewards**: Combining frequency-domain (from paper) with physics-based rewards

---

## Installation

```bash
conda create -n finesse python=3.11 -y
conda activate finesse
pip install finesse finesse-ligo gymnasium numpy scipy matplotlib
pip install jax jaxlib flax optax tensorflow-probability
pip install stable-baselines3  # For SAC baseline
```

## Project Structure

```
├── envs/           # Gymnasium environments (FinesseOMCEnv)
├── dreamer/        # Training scripts and network architectures
├── rewards/        # Frequency-domain filters
└── utils/          # Noise injection, domain randomization
```

## Quick Start

```python
from envs import FinesseOMCEnv

env = FinesseOMCEnv()
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)
```

## Training

```bash
# Simple test (random policy)
python dreamer/train.py --algorithm simple --steps 10000

# SAC training
python dreamer/train.py --algorithm sac --steps 100000 --logdir ./runs/exp1
```

---

## Research Questions

1. **Does physics simulation help?** Can Finesse capture effects that linear models miss?
2. **Does model-based RL help?** Is DreamerV3 more sample-efficient than MPO?
3. **Can we optimize for sensitivity?** Can RL minimize quantum noise directly?
4. **Architecture comparison**: Dilated conv vs. Transformer for control?

---

## Acknowledgments

The Deep Loop Shaping paper represents groundbreaking work. This project aims to:
- Learn from and adopt their proven techniques (frequency-domain rewards, dilated conv)
- Explore complementary directions (different target, physics simulation)
- Contribute to understanding RL for gravitational wave detectors

## License

MIT
