# Proposed Approach: Differences from Deep Loop Shaping

## Overview

This document outlines our proposed research direction and how it differs from the Deep Loop Shaping paper (arxiv:2509.14016). We aim to explore complementary approaches, not compete with their excellent work.

---

## Key Differences in Approach

### 1. Simulation Environment

| Aspect | Deep Loop Shaping | Our Proposed Approach |
|--------|-------------------|----------------------|
| **Simulation** | Linear identified state-space model | Finesse physics simulation |
| **Plant Model** | System identification from measurements | First-principles optical modeling |

**What we aim to explore:**
- Whether physics-based simulation (Finesse) can capture dynamics that linear models miss
- How nonlinear effects like radiation pressure and thermal lensing affect policy learning
- Whether training on physics simulation improves real-world transfer (hypothesis to test)

---

### 2. Target Control Loop

| Aspect | Deep Loop Shaping | Our Proposed Approach |
|--------|-------------------|----------------------|
| **Target** | θ_CHP (common-hard-pitch) | Output Mode Cleaner (OMC) |
| **Scope** | Single SISO loop | Alignment + squeezing optimization |

**What we aim to explore:**
- OMC alignment control using RL (different control challenge)
- Integration with squeezed light injection (unexplored in RL literature)
- Whether the same techniques transfer to different subsystems

---

### 3. Algorithm Choice

| Aspect | Deep Loop Shaping | Our Proposed Approach |
|--------|-------------------|----------------------|
| **Algorithm** | MPO (model-free) | SAC / DreamerV3 (model-based option) |
| **Learning** | Off-policy actor-critic | World model + imagination |

**What we aim to explore:**
- Whether model-based RL (DreamerV3) improves sample efficiency
- If learning a world model helps with generalization
- Trade-offs between model-free and model-based approaches for control

---

### 4. Policy Architecture

| Aspect | Deep Loop Shaping | Our Proposed Approach |
|--------|-------------------|----------------------|
| **Policy** | MLP + Dilated Conv | Same + Transformer option |
| **Motivation** | Proven to work | Explore alternatives |

**What we aim to explore:**
- We implement their dilated conv architecture (proven effective)
- We also explore Transformer-based policies as an alternative
- Goal: understand which architecture works best for different scenarios

---

### 5. Reward Design

| Aspect | Deep Loop Shaping | Our Proposed Approach |
|--------|-------------------|----------------------|
| **Frequency-domain** | IIR filters + sigmoid scoring | Same (we adopt their technique) |
| **Physics-based** | Not used | Proposed addition |

**What we aim to explore:**
- Their frequency-domain reward design is elegant; we adopt it directly
- We propose adding physics-based rewards from Finesse outputs (e.g., transmitted power, quantum noise)
- Hypothesis: hybrid rewards may help optimize for sensitivity, not just stability

---

## Research Questions

1. **Does physics simulation help?**
   - Can training on Finesse improve transfer to real hardware?
   - Does it capture effects that linear models miss?

2. **Does model-based RL help?**
   - Is DreamerV3 more sample-efficient than MPO for this domain?
   - Does the learned world model generalize?

3. **Can we optimize for sensitivity?**
   - Can RL directly minimize quantum noise (not just control noise)?
   - Is squeezing optimization feasible with RL?

4. **Architecture comparison:**
   - Dilated conv (paper) vs. Transformer: which works better?
   - What history length is needed?

---

## Acknowledgments

The Deep Loop Shaping paper by Google DeepMind, Caltech, and GSSI represents groundbreaking work. Our project aims to:
- Learn from and adopt their proven techniques
- Explore complementary directions (different target, different simulation)
- Contribute to the broader understanding of RL for gravitational wave detectors

This is an educational/research project building on their foundation.
