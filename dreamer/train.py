"""
DreamerV3 Training Script for LIGO OMC Control.

This script provides a simplified training loop that can work with
either the full DreamerV3 implementation or a simpler SAC baseline.

For full DreamerV3, install: pip install dreamerv3
For SAC baseline: pip install stable-baselines3

Usage:
    conda activate finesse
    python dreamer/train.py --steps 100000 --logdir ./runs/exp1
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs import FinesseOMCEnv


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_with_sac(env, config: dict, args):
    """
    Train using Stable-Baselines3 SAC as baseline.
    
    SAC is a good alternative to DreamerV3 for continuous control.
    """
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import (
            EvalCallback, 
            CheckpointCallback
        )
    except ImportError:
        print("stable-baselines3 not installed. Install with:")
        print("  pip install stable-baselines3")
        return None
    
    # Create log directory
    log_dir = Path(args.logdir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=config['train'].get('actor_lr', 3e-4),
        buffer_size=config['replay'].get('size', 1000000),
        learning_starts=config['replay'].get('min_size', 1000),
        batch_size=config['train'].get('batch_size', 256),
        gamma=config['train'].get('discount', 0.99),
        tau=0.005,
        verbose=1,
        tensorboard_log=str(log_dir / "tensorboard"),
        device="auto"
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config['log'].get('save_every', 10000),
        save_path=str(log_dir / "checkpoints"),
        name_prefix="sac_omc"
    )
    
    eval_callback = EvalCallback(
        env,
        best_model_save_path=str(log_dir / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=config['log'].get('log_every', 1000),
        deterministic=True,
        render=False
    )
    
    # Train
    print(f"\n{'='*60}")
    print(f"Training SAC on FinesseOMC-v0")
    print(f"Steps: {args.steps}")
    print(f"Log directory: {log_dir}")
    print(f"{'='*60}\n")
    
    model.learn(
        total_timesteps=args.steps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # Save final model
    final_path = log_dir / "final_model"
    model.save(str(final_path))
    print(f"\nFinal model saved to {final_path}")
    
    return model


def train_with_dreamer(env, config: dict, args):
    """
    Train using DreamerV3.
    
    Requires: pip install dreamerv3
    """
    try:
        import dreamerv3
        from dreamerv3 import embodied
    except ImportError:
        print("DreamerV3 not installed. Falling back to SAC baseline.")
        print("To install DreamerV3: pip install dreamerv3")
        return train_with_sac(env, config, args)
    
    # DreamerV3 training configuration
    log_dir = Path(args.logdir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Dreamer config
    dreamer_config = embodied.Config({
        'logdir': str(log_dir),
        'seed': args.seed,
        'steps': args.steps,
        **config  # Merge our config
    })
    
    print(f"\n{'='*60}")
    print(f"Training DreamerV3 on FinesseOMC-v0")
    print(f"Steps: {args.steps}")
    print(f"Log directory: {log_dir}")
    print(f"{'='*60}\n")
    
    # Run DreamerV3 training
    # Note: This is a simplified interface, actual usage may vary
    # based on dreamerv3 version
    try:
        embodied.run.train(dreamer_config, env)
    except Exception as e:
        print(f"DreamerV3 training error: {e}")
        print("Falling back to SAC baseline...")
        return train_with_sac(env, config, args)
    
    return None


def simple_training_loop(env, args):
    """
    Simple training loop for testing without external RL libraries.
    
    This is just for verifying the environment works.
    """
    print(f"\n{'='*60}")
    print(f"Running simple random policy test")
    print(f"Episodes: {args.steps // 1024}")
    print(f"{'='*60}\n")
    
    total_rewards = []
    
    for episode in range(args.steps // 1024):
        obs, info = env.reset()
        episode_reward = 0
        
        for step in range(1024):
            # Random action
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        mean_reward = np.mean(total_rewards[-10:])
        
        print(f"Episode {episode+1}: reward={episode_reward:.2f}, "
              f"mean(10)={mean_reward:.2f}, "
              f"alignment_rms={info['alignment_rms']:.1f}")
    
    print(f"\nSimple test completed!")
    print(f"Mean reward: {np.mean(total_rewards):.2f}")
    print(f"Std reward: {np.std(total_rewards):.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Train RL agent for LIGO OMC control"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=100000,
        help="Total training steps"
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=f"./runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Log directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to config file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["dreamer", "sac", "simple"],
        default="sac",
        help="Algorithm to use"
    )
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    
    # Load config
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"Warning: Config not found at {args.config}, using defaults")
        config = {
            'train': {'batch_size': 256, 'discount': 0.99},
            'replay': {'size': 100000, 'min_size': 1000},
            'log': {'save_every': 10000, 'log_every': 1000}
        }
    
    # Create environment
    print("Creating FinesseOMC environment...")
    env = FinesseOMCEnv(
        episode_length=1024,
        noise_scale=1.0,
        domain_randomization=True
    )
    
    # Train based on algorithm choice
    if args.algorithm == "dreamer":
        train_with_dreamer(env, config, args)
    elif args.algorithm == "sac":
        train_with_sac(env, config, args)
    else:
        simple_training_loop(env, args)
    
    env.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
