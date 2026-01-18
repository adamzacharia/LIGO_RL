"""
Reward Functions for LIGO OMC Alignment RL.

Implements hybrid rewards combining:
1. Physics-based rewards (from Finesse outputs)
2. Frequency-domain rewards (filter-based, like Deep Loop Shaping)
3. Stability rewards (lock duration)
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy import signal


def sigmoid_score(value: float, good: float, bad: float) -> float:
    """
    Sigmoid-based scoring function (from Deep Loop Shaping paper).
    
    Maps a value to [0, 1] where:
    - value <= good -> score ~= 0.95
    - value >= bad -> score ~= 0.05
    
    Args:
        value: Input value to score
        good: Value at which score should be ~0.95
        bad: Value at which score should be ~0.05
        
    Returns:
        Score in [0, 1]
    """
    # Anchor points for sigmoid
    ln_19 = np.log(19)  # ~2.944
    
    if good == bad:
        return 0.5
    
    # Linear mapping to sigmoid input
    scale = (2 * ln_19) / (bad - good)
    offset = -ln_19 - scale * good
    
    sigmoid_input = scale * value + offset
    score = 1.0 / (1.0 + np.exp(sigmoid_input))
    
    return float(np.clip(score, 0, 1))


def power_reward(power: float, target_power: float = 1.0) -> float:
    """
    Reward based on transmitted optical power.
    
    Higher power = better mode-matching and alignment.
    
    Args:
        power: Current transmitted power (normalized)
        target_power: Target power level
        
    Returns:
        Reward in [0, 1]
    """
    return sigmoid_score(
        value=power,
        good=0.95 * target_power,
        bad=0.5 * target_power
    )


def alignment_error_reward(
    error_rms: float,
    good_threshold: float = 10.0,  # nanoradians
    bad_threshold: float = 100.0
) -> float:
    """
    Reward based on alignment error RMS.
    
    Lower error = higher reward.
    
    Args:
        error_rms: RMS alignment error in nanoradians
        good_threshold: Error below this is excellent
        bad_threshold: Error above this is poor
        
    Returns:
        Reward in [0, 1]
    """
    # Invert because lower error is better
    return sigmoid_score(
        value=error_rms,
        good=good_threshold,
        bad=bad_threshold
    )


def control_effort_penalty(
    action: np.ndarray,
    prev_action: np.ndarray,
    max_rate: float = 0.5
) -> float:
    """
    Penalty for rapid control actions.
    
    Encourages smooth control to reduce injected noise.
    
    Args:
        action: Current action
        prev_action: Previous action
        max_rate: Maximum expected rate of change
        
    Returns:
        Penalty in [0, 1] where 0 = no penalty
    """
    rate = np.linalg.norm(action - prev_action)
    
    # Penalty increases with rate
    penalty = sigmoid_score(
        value=rate,
        good=0.1 * max_rate,
        bad=max_rate
    )
    
    return 1.0 - penalty  # Invert so 0 = good (no penalty)


def frequency_band_penalty(
    action_history: List[np.ndarray],
    sample_rate: float = 256.0,
    band_low: float = 8.0,
    band_high: float = 30.0
) -> float:
    """
    Penalty for control energy in the observation band (8-30 Hz).
    
    This is the key insight from Deep Loop Shaping: minimize control
    action in the frequency band where GW signals are expected.
    
    Args:
        action_history: List of recent actions
        sample_rate: Control rate in Hz
        band_low: Lower frequency of penalty band
        band_high: Upper frequency of penalty band
        
    Returns:
        Penalty in [0, 1] where 0 = no penalty
    """
    if len(action_history) < 64:
        return 0.0  # Not enough history
    
    # Stack recent actions
    actions = np.array(action_history[-256:])  # ~1 second at 256 Hz
    
    # Compute power spectral density
    freqs, psd = signal.welch(
        actions.mean(axis=1),  # Average over DOFs
        fs=sample_rate,
        nperseg=min(64, len(actions))
    )
    
    # Find power in observation band
    band_mask = (freqs >= band_low) & (freqs <= band_high)
    band_power = np.mean(psd[band_mask]) if np.any(band_mask) else 0.0
    
    # Convert to penalty
    penalty = sigmoid_score(
        value=band_power,
        good=0.001,  # Very low power is good
        bad=0.1      # High power is bad
    )
    
    return 1.0 - penalty


def stability_reward(
    steps_locked: int,
    target_steps: int = 1024,
    min_steps: int = 100
) -> float:
    """
    Reward for maintaining lock.
    
    Longer lock duration = higher reward.
    
    Args:
        steps_locked: Number of steps in lock
        target_steps: Target episode length
        min_steps: Minimum steps for any reward
        
    Returns:
        Reward in [0, 1]
    """
    if steps_locked < min_steps:
        return 0.0
    
    return np.clip(steps_locked / target_steps, 0, 1)


def compute_hybrid_reward(
    power: float,
    alignment_rms: float,
    action: np.ndarray,
    prev_action: np.ndarray,
    action_history: List[np.ndarray],
    steps_locked: int,
    weights: Optional[dict] = None
) -> Tuple[float, dict]:
    """
    Compute the full hybrid reward.
    
    Combines all reward components with configurable weights.
    Uses multiplicative combination (like Deep Loop Shaping)
    for AND-like behavior.
    
    Args:
        power: Transmitted optical power
        alignment_rms: RMS alignment error
        action: Current action
        prev_action: Previous action
        action_history: History of actions
        steps_locked: Steps in lock
        weights: Optional weight dictionary
        
    Returns:
        (total_reward, component_dict)
    """
    if weights is None:
        weights = {
            "power": 1.0,
            "alignment": 1.0,
            "control": 0.5,
            "frequency": 0.5,
            "stability": 0.2
        }
    
    # Compute components
    r_power = power_reward(power)
    r_align = alignment_error_reward(alignment_rms)
    p_control = control_effort_penalty(action, prev_action)
    p_freq = frequency_band_penalty(action_history)
    r_stable = stability_reward(steps_locked)
    
    # Multiplicative combination for main objectives
    main_reward = r_power * r_align
    
    # Additive penalties and bonuses
    total = (
        weights["power"] * main_reward
        - weights["control"] * p_control
        - weights["frequency"] * p_freq
        + weights["stability"] * r_stable
    )
    
    components = {
        "power": r_power,
        "alignment": r_align,
        "control_penalty": p_control,
        "frequency_penalty": p_freq,
        "stability": r_stable,
        "total": total
    }
    
    return float(total), components


if __name__ == "__main__":
    # Test reward functions
    print("Testing reward functions...")
    
    # Test sigmoid score
    print(f"sigmoid(0.9, good=0.95, bad=0.5) = {sigmoid_score(0.9, 0.95, 0.5):.4f}")
    print(f"sigmoid(0.5, good=0.95, bad=0.5) = {sigmoid_score(0.5, 0.95, 0.5):.4f}")
    
    # Test power reward
    print(f"power_reward(0.95) = {power_reward(0.95):.4f}")
    print(f"power_reward(0.5) = {power_reward(0.5):.4f}")
    
    # Test hybrid reward
    action = np.array([0.1, -0.1, 0.05, -0.05])
    prev_action = np.array([0.0, 0.0, 0.0, 0.0])
    action_history = [action] * 100
    
    total, components = compute_hybrid_reward(
        power=0.9,
        alignment_rms=20.0,
        action=action,
        prev_action=prev_action,
        action_history=action_history,
        steps_locked=500
    )
    
    print(f"\nHybrid reward components:")
    for k, v in components.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nReward function tests completed!")
