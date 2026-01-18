"""
Finesse-based Gymnasium Environment for LIGO OMC Alignment Control.

This environment wraps a Finesse 3 simulation of a simplified Advanced LIGO
Output Mode Cleaner (OMC) for reinforcement learning training.

Key differences from Deep Loop Shaping:
- Uses physics-based simulation (Finesse) instead of system identification
- Targets OMC alignment instead of θ_CHP
- Includes quantum noise modeling
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any

try:
    import finesse
    from finesse.ligo import make_aligo
    FINESSE_AVAILABLE = True
except ImportError:
    FINESSE_AVAILABLE = False
    print("Warning: Finesse not available. Using mock environment.")


class FinesseOMCEnv(gym.Env):
    """
    Gymnasium environment for LIGO OMC alignment control using Finesse.
    
    Observation Space:
        - Power readings from photodetectors (4 values)
        - Wavefront sensor error signals (4 values)
        - Previous action (4 values)
        Total: 12-dimensional continuous
    
    Action Space:
        - OMC mirror pitch/yaw adjustments (4 DOF)
        Continuous: [-1, 1] normalized, scaled to physical units
    
    Reward:
        Hybrid reward combining:
        - Physics-based: Transmitted power / mode-matching
        - Frequency-domain: Band-limited control effort penalty
        - Stability: Lock duration bonus
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        step_frequency: float = 256.0,  # Hz, matching Deep Loop Shaping
        episode_length: int = 1024,  # steps (~4 seconds)
        noise_scale: float = 1.0,
        domain_randomization: bool = True,
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.step_frequency = step_frequency
        self.dt = 1.0 / step_frequency
        self.episode_length = episode_length
        self.noise_scale = noise_scale
        self.domain_randomization = domain_randomization
        
        # Action space: 4 DOF for OMC alignment (pitch/yaw for 2 mirrors)
        # Normalized to [-1, 1], will be scaled to physical units (nanoradians)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
        
        # Observation space: sensor readings + previous action
        # [power_dc, power_ac1, power_ac2, power_ac3, 
        #  wfs_pitch1, wfs_yaw1, wfs_pitch2, wfs_yaw2,
        #  prev_action (4)]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32
        )
        
        # Physical scaling factors
        self.action_scale = 100.0  # nanoradians per normalized unit
        self.max_misalignment = 500.0  # nanoradians, beyond this = lock loss
        
        # Initialize state
        self.current_step = 0
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.alignment_state = np.zeros(4, dtype=np.float32)  # Current mirror positions
        
        # Finesse model (initialized on first reset)
        self.model = None
        self._init_finesse_model()
        
        # History for frequency-domain rewards
        self.action_history = []
        self.reward_history = []
        
    def _init_finesse_model(self):
        """Initialize the Finesse model for OMC simulation."""
        if not FINESSE_AVAILABLE:
            # Mock mode for testing without Finesse
            self.model = None
            return
            
        try:
            # Create a simplified OMC model
            # In production, this would use a full aLIGO model
            self.model = finesse.Model()
            
            # Simple Fabry-Perot cavity representing OMC
            self.model.parse("""
                # Laser source
                l laser 1 0 n0
                
                # Input optics  
                s s_in 1 n0 n1
                
                # OMC input mirror (curved)
                m OMC_IM 0.99 0.01 0 n1 n2
                attr OMC_IM Rc 2.5
                
                # OMC cavity length
                s OMC_cav 0.3 n2 n3
                
                # OMC output mirror (curved)
                m OMC_OM 0.99 0.01 0 n3 n4
                attr OMC_OM Rc 2.5
                
                # Output path
                s s_out 1 n4 n5
                
                # Photodetectors
                pd power_dc n5
                
                # Quantum noise detector
                qnoised qnoise_out 1 0 n5
            """)
            
            # Build the model
            self.model.modes(maxtem=2)  # Include higher-order modes
            
        except Exception as e:
            print(f"Warning: Could not initialize Finesse model: {e}")
            self.model = None
    
    def _apply_domain_randomization(self):
        """Apply random variations to plant parameters for robustness."""
        if not self.domain_randomization or self.model is None:
            return
            
        # Randomize mirror reflectivities slightly
        r_variation = np.random.uniform(0.995, 1.005)
        
        # Randomize alignment offsets (simulating thermal drift)
        self.alignment_offset = np.random.uniform(-50, 50, size=4).astype(np.float32)
        
        # Randomize noise level
        self.current_noise_scale = self.noise_scale * np.random.uniform(0.5, 2.0)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation from the Finesse model or mock."""
        if self.model is None:
            # Mock observation for testing
            # Simulate power that decreases with misalignment
            misalignment_rms = np.sqrt(np.mean(self.alignment_state**2))
            power_factor = np.exp(-misalignment_rms / 200.0)
            
            power_dc = 1.0 * power_factor + np.random.normal(0, 0.01)
            power_ac = np.random.normal(0, 0.1, size=3)
            
            # WFS signals proportional to misalignment
            wfs_signals = self.alignment_state * 0.01 + np.random.normal(0, 0.001, size=4)
            
            obs = np.concatenate([
                [power_dc],
                power_ac,
                wfs_signals,
                self.prev_action
            ]).astype(np.float32)
            
        else:
            # Run Finesse simulation
            try:
                # Apply current alignment state to model
                if hasattr(self.model, 'OMC_IM'):
                    self.model.OMC_IM.phi = self.alignment_state[0] * 1e-9  # nrad to rad
                    self.model.OMC_IM.theta = self.alignment_state[1] * 1e-9
                if hasattr(self.model, 'OMC_OM'):
                    self.model.OMC_OM.phi = self.alignment_state[2] * 1e-9
                    self.model.OMC_OM.theta = self.alignment_state[3] * 1e-9
                
                # Run simulation
                sol = self.model.run()
                
                # Extract outputs
                power_dc = float(sol['power_dc'])
                power_ac = np.zeros(3)  # Would come from demodulated signals
                wfs_signals = self.alignment_state * 0.01  # Simplified WFS model
                
                obs = np.concatenate([
                    [power_dc],
                    power_ac,
                    wfs_signals,
                    self.prev_action
                ]).astype(np.float32)
                
            except Exception as e:
                print(f"Finesse simulation error: {e}")
                obs = np.zeros(12, dtype=np.float32)
        
        return obs
    
    def _compute_reward(self, action: np.ndarray) -> float:
        """
        Compute hybrid reward combining physics and frequency-domain components.
        
        Components:
        1. Power reward: Higher transmitted power = better alignment
        2. Control effort penalty: Penalize large actions in observation band
        3. Stability reward: Bonus for maintaining lock
        """
        # Get current power (from observation)
        obs = self._get_observation()
        power_dc = obs[0]
        
        # 1. Power/alignment reward (physics-based)
        power_reward = np.clip(power_dc, 0, 1.0)
        
        # 2. Control effort penalty (frequency-domain approximation)
        # Penalize large rapid changes in action
        action_change = np.linalg.norm(action - self.prev_action)
        control_penalty = 0.1 * action_change
        
        # 3. Stability reward (lock not lost)
        misalignment_rms = np.sqrt(np.mean(self.alignment_state**2))
        if misalignment_rms < self.max_misalignment:
            stability_reward = 0.1
        else:
            stability_reward = -1.0  # Large penalty for lock loss
        
        # Combine rewards (multiplicative like Deep Loop Shaping)
        total_reward = power_reward - control_penalty + stability_reward
        
        return float(total_reward)
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate (lock lost)."""
        misalignment_rms = np.sqrt(np.mean(self.alignment_state**2))
        return misalignment_rms > self.max_misalignment
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset step counter
        self.current_step = 0
        
        # Reset alignment state with small random offset
        self.alignment_state = np.random.uniform(-50, 50, size=4).astype(np.float32)
        self.prev_action = np.zeros(4, dtype=np.float32)
        
        # Clear history
        self.action_history = []
        self.reward_history = []
        
        # Apply domain randomization
        self._apply_domain_randomization()
        
        # Get initial observation
        obs = self._get_observation()
        info = {
            "alignment_rms": float(np.sqrt(np.mean(self.alignment_state**2))),
            "step": 0
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Normalized action [-1, 1] for 4 DOF
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        
        # Scale action to physical units and apply
        physical_action = action * self.action_scale  # nanoradians
        self.alignment_state += physical_action
        
        # Add seismic noise disturbance
        seismic_noise = np.random.normal(0, 5 * self.noise_scale, size=4).astype(np.float32)
        self.alignment_state += seismic_noise
        
        # Compute reward
        reward = self._compute_reward(action)
        
        # Update history
        self.action_history.append(action.copy())
        self.reward_history.append(reward)
        self.prev_action = action.copy()
        
        # Check termination conditions
        terminated = self._check_termination()
        self.current_step += 1
        truncated = self.current_step >= self.episode_length
        
        # Get observation
        obs = self._get_observation()
        
        # Info dict
        info = {
            "alignment_rms": float(np.sqrt(np.mean(self.alignment_state**2))),
            "step": self.current_step,
            "power": float(obs[0]),
            "episode_reward": float(np.sum(self.reward_history))
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment (visualization)."""
        if self.render_mode == "human":
            print(f"Step {self.current_step}: Alignment RMS = {np.sqrt(np.mean(self.alignment_state**2)):.2f} nrad")
        return None
    
    def close(self):
        """Clean up resources."""
        pass


# Register the environment with Gymnasium
gym.register(
    id="FinesseOMC-v0",
    entry_point="src.envs.finesse_ligo_env:FinesseOMCEnv",
)


if __name__ == "__main__":
    # Quick test
    env = FinesseOMCEnv()
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial alignment RMS: {info['alignment_rms']:.2f} nrad")
    
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.4f}, power={info['power']:.4f}, rms={info['alignment_rms']:.2f}")
        
        if terminated or truncated:
            break
    
    env.close()
    print("Environment test completed!")
