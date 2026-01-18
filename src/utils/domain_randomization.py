"""
Domain Randomization for Robust Policy Training.

Randomizes plant parameters during training to ensure the learned
policy generalizes to real hardware with parameter uncertainty.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class RandomizationConfig:
    """Configuration for domain randomization."""
    
    # Mirror reflectivity variations (fraction)
    reflectivity_range: tuple = (0.995, 1.005)
    
    # Alignment offset range (nanoradians)
    alignment_offset_range: tuple = (-50.0, 50.0)
    
    # Noise scale multiplier range
    noise_scale_range: tuple = (0.5, 2.0)
    
    # Seismic amplitude variation
    seismic_range: tuple = (0.5, 3.0)
    
    # Cavity length variation (micrometers)
    length_variation: tuple = (-0.1, 0.1)
    
    # Time delay variation (samples at 256 Hz)
    delay_range: tuple = (0, 3)


class DomainRandomizer:
    """
    Applies domain randomization to simulation parameters.
    
    Following Deep Loop Shaping, we randomize:
    - Angular instability pole frequency
    - Seismic noise variations
    - Overall noise strength
    """
    
    def __init__(
        self,
        config: Optional[RandomizationConfig] = None,
        seed: Optional[int] = None
    ):
        self.config = config or RandomizationConfig()
        self.rng = np.random.default_rng(seed)
        
        # Current randomized values
        self.current_params: Dict[str, Any] = {}
        
    def randomize(self) -> Dict[str, Any]:
        """
        Generate a new set of randomized parameters.
        
        Returns:
            Dictionary of randomized parameters
        """
        cfg = self.config
        
        self.current_params = {
            # Mirror parameters
            'reflectivity_scale': self.rng.uniform(*cfg.reflectivity_range),
            
            # Alignment offsets (4 DOF)
            'alignment_offset': self.rng.uniform(
                cfg.alignment_offset_range[0],
                cfg.alignment_offset_range[1],
                size=4
            ),
            
            # Noise parameters
            'noise_scale': self.rng.uniform(*cfg.noise_scale_range),
            'seismic_scale': self.rng.uniform(*cfg.seismic_range),
            
            # Physical variations
            'length_offset': self.rng.uniform(*cfg.length_variation),
            
            # Control system
            'delay_samples': self.rng.integers(*cfg.delay_range, endpoint=True)
        }
        
        return self.current_params
    
    def get_params(self) -> Dict[str, Any]:
        """Get current randomized parameters."""
        if not self.current_params:
            return self.randomize()
        return self.current_params
    
    def apply_to_state(
        self,
        alignment_state: np.ndarray
    ) -> np.ndarray:
        """
        Apply randomization to alignment state.
        
        Args:
            alignment_state: Current alignment state [4]
            
        Returns:
            Modified alignment state with offsets
        """
        params = self.get_params()
        offset = np.array(params['alignment_offset'], dtype=np.float32)
        return alignment_state + offset
    
    def get_noise_scale(self) -> float:
        """Get current noise scaling factor."""
        params = self.get_params()
        return float(params['noise_scale'] * params['seismic_scale'])
    
    def get_delay(self) -> int:
        """Get current control delay in samples."""
        params = self.get_params()
        return int(params['delay_samples'])
    
    def __repr__(self) -> str:
        if not self.current_params:
            return "DomainRandomizer(not randomized)"
        
        return (
            f"DomainRandomizer(\n"
            f"  reflectivity_scale={self.current_params['reflectivity_scale']:.4f},\n"
            f"  alignment_offset={self.current_params['alignment_offset']},\n"
            f"  noise_scale={self.current_params['noise_scale']:.2f},\n"
            f"  seismic_scale={self.current_params['seismic_scale']:.2f},\n"
            f"  delay_samples={self.current_params['delay_samples']}\n"
            f")"
        )


class ProgressiveRandomizer(DomainRandomizer):
    """
    Domain randomizer with curriculum learning.
    
    Starts with small variations and increases over training.
    """
    
    def __init__(
        self,
        config: Optional[RandomizationConfig] = None,
        seed: Optional[int] = None,
        start_scale: float = 0.1,
        end_scale: float = 1.0,
        warmup_steps: int = 100000
    ):
        super().__init__(config, seed)
        
        self.start_scale = start_scale
        self.end_scale = end_scale
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
    def set_step(self, step: int):
        """Update current training step."""
        self.current_step = step
        
    def get_scale(self) -> float:
        """Get current randomization scale based on training progress."""
        if self.current_step >= self.warmup_steps:
            return self.end_scale
        
        progress = self.current_step / self.warmup_steps
        return self.start_scale + (self.end_scale - self.start_scale) * progress
    
    def randomize(self) -> Dict[str, Any]:
        """Generate scaled randomized parameters."""
        # Get full randomization
        params = super().randomize()
        
        # Scale by curriculum progress
        scale = self.get_scale()
        
        # Apply scaling to variations (not base values)
        params['alignment_offset'] = params['alignment_offset'] * scale
        params['noise_scale'] = 1.0 + (params['noise_scale'] - 1.0) * scale
        params['seismic_scale'] = 1.0 + (params['seismic_scale'] - 1.0) * scale
        params['length_offset'] = params['length_offset'] * scale
        
        self.current_params = params
        return params


if __name__ == "__main__":
    print("Testing domain randomization...")
    
    # Test basic randomizer
    dr = DomainRandomizer(seed=42)
    
    print("\nBasic randomization (5 samples):")
    for i in range(5):
        params = dr.randomize()
        print(f"  Sample {i+1}: noise_scale={params['noise_scale']:.2f}, "
              f"seismic={params['seismic_scale']:.2f}")
    
    # Test progressive randomizer
    pr = ProgressiveRandomizer(
        seed=42,
        start_scale=0.1,
        end_scale=1.0,
        warmup_steps=10000
    )
    
    print("\nProgressive randomization:")
    for step in [0, 2500, 5000, 7500, 10000]:
        pr.set_step(step)
        params = pr.randomize()
        print(f"  Step {step}: scale={pr.get_scale():.2f}, "
              f"alignment_offset_max={np.max(np.abs(params['alignment_offset'])):.1f}")
    
    print("\nDomain randomization tests completed!")
