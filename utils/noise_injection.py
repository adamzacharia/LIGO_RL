"""
Noise Injection Models for LIGO Simulation.

Implements realistic noise sources:
- Seismic noise (microseismic peak ~0.1-1 Hz)
- Thermal noise (suspension thermal, mirror thermal)
- Quantum noise (shot noise, radiation pressure)
- Technical noise (laser RIN, electronics)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from scipy import signal


class NoiseModel(ABC):
    """Abstract base class for noise models."""
    
    @abstractmethod
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Generate noise samples."""
        pass
    
    @abstractmethod
    def get_psd(self, frequencies: np.ndarray) -> np.ndarray:
        """Get power spectral density."""
        pass


class SeismicNoise(NoiseModel):
    """
    Seismic noise model for LIGO.
    
    Includes:
    - Microseismic peak (~0.1-0.3 Hz)
    - Ground motion spectrum
    - Pendulum filtering effect
    """
    
    def __init__(
        self,
        sample_rate: float = 256.0,
        amplitude: float = 1.0,  # Scale factor
        microseismic_freq: float = 0.15,  # Hz
        microseismic_q: float = 5.0
    ):
        self.sample_rate = sample_rate
        self.amplitude = amplitude
        self.f_micro = microseismic_freq
        self.q_micro = microseismic_q
        
        # Create colored noise filter
        self._setup_filter()
        
    def _setup_filter(self):
        """Setup filter to color white noise to seismic spectrum."""
        # Low-pass to get the ~f^-2 roll-off at high frequencies
        nyquist = self.sample_rate / 2
        
        # Microseismic peak as bandpass
        low = max(0.05 / nyquist, 0.01)
        high = min(0.5 / nyquist, 0.99)
        
        if low < high:
            self.b, self.a = signal.butter(2, [low, high], btype='band')
            self.zi = signal.lfilter_zi(self.b, self.a)
        else:
            # Fallback to simple lowpass
            self.b, self.a = signal.butter(2, 0.1, btype='low')
            self.zi = signal.lfilter_zi(self.b, self.a)
        
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Generate seismic noise samples."""
        # White noise
        white = np.random.randn(n_samples)
        
        # Color it with seismic spectrum
        if n_samples > 1:
            colored, self.zi = signal.lfilter(self.b, self.a, white, zi=self.zi)
        else:
            colored = white
            
        return self.amplitude * colored
    
    def get_psd(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Get seismic PSD.
        
        Approximate seismic spectrum: f^-2 with microseismic bump.
        """
        psd = np.zeros_like(frequencies)
        
        for i, f in enumerate(frequencies):
            if f < 0.01:
                psd[i] = 1e-10  # Very low freq
            else:
                # f^-2 spectrum
                base = 1e-10 / (f ** 2)
                
                # Microseismic peak
                micro_peak = np.exp(-((f - self.f_micro) / 0.05) ** 2)
                
                psd[i] = base * (1 + 10 * micro_peak)
        
        return psd * self.amplitude ** 2


class ThermalNoise(NoiseModel):
    """
    Thermal noise model.
    
    Includes:
    - Suspension thermal noise
    - Mirror coating thermal noise
    """
    
    def __init__(
        self,
        sample_rate: float = 256.0,
        amplitude: float = 0.1,
        characteristic_freq: float = 10.0  # Hz
    ):
        self.sample_rate = sample_rate
        self.amplitude = amplitude
        self.f_char = characteristic_freq
        
        # Simple 1/f noise filter
        self._setup_filter()
        
    def _setup_filter(self):
        """Setup filter for thermal noise spectrum."""
        nyquist = self.sample_rate / 2
        cutoff = min(self.f_char / nyquist, 0.9)
        
        self.b, self.a = signal.butter(1, cutoff, btype='low')
        self.zi = signal.lfilter_zi(self.b, self.a)
        
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Generate thermal noise samples."""
        white = np.random.randn(n_samples)
        
        if n_samples > 1:
            colored, self.zi = signal.lfilter(self.b, self.a, white, zi=self.zi)
        else:
            colored = white
            
        return self.amplitude * colored
    
    def get_psd(self, frequencies: np.ndarray) -> np.ndarray:
        """Get thermal noise PSD (1/f characteristic)."""
        psd = np.zeros_like(frequencies)
        
        for i, f in enumerate(frequencies):
            if f < 0.01:
                psd[i] = 1e-8
            else:
                psd[i] = 1e-12 / f  # 1/f noise
        
        return psd * self.amplitude ** 2


class QuantumNoise(NoiseModel):
    """
    Quantum noise model.
    
    Combines:
    - Shot noise (white, high frequency)
    - Radiation pressure noise (low frequency)
    
    At the standard quantum limit, these are balanced.
    """
    
    def __init__(
        self,
        sample_rate: float = 256.0,
        amplitude: float = 0.01,
        crossover_freq: float = 50.0  # Hz where shot = rad pressure
    ):
        self.sample_rate = sample_rate
        self.amplitude = amplitude
        self.f_cross = crossover_freq
        
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Generate quantum noise samples."""
        # Quantum noise is approximately white in the control band
        return self.amplitude * np.random.randn(n_samples)
    
    def get_psd(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Get quantum noise PSD.
        
        SQL limited: shot ~ constant, rad pressure ~ 1/f^2
        """
        psd = np.zeros_like(frequencies)
        
        for i, f in enumerate(frequencies):
            if f < 0.01:
                f = 0.01
            
            # Shot noise (constant)
            shot = 1e-20
            
            # Radiation pressure (1/f^2 at low freq)
            rad_pressure = 1e-20 * (self.f_cross / f) ** 2
            
            # Combine (add in quadrature)
            psd[i] = shot + rad_pressure
        
        return psd * self.amplitude ** 2


class CombinedNoiseModel(NoiseModel):
    """Combines multiple noise sources."""
    
    def __init__(self, sample_rate: float = 256.0, scale: float = 1.0):
        self.sample_rate = sample_rate
        self.scale = scale
        
        self.seismic = SeismicNoise(sample_rate, amplitude=5.0 * scale)
        self.thermal = ThermalNoise(sample_rate, amplitude=1.0 * scale)
        self.quantum = QuantumNoise(sample_rate, amplitude=0.1 * scale)
        
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Generate combined noise samples."""
        return (
            self.seismic.sample(n_samples) +
            self.thermal.sample(n_samples) +
            self.quantum.sample(n_samples)
        )
    
    def get_psd(self, frequencies: np.ndarray) -> np.ndarray:
        """Get combined PSD."""
        return (
            self.seismic.get_psd(frequencies) +
            self.thermal.get_psd(frequencies) +
            self.quantum.get_psd(frequencies)
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Testing noise models...")
    
    # Create noise sources
    seismic = SeismicNoise(amplitude=2.0)
    thermal = ThermalNoise(amplitude=0.5)
    quantum = QuantumNoise(amplitude=0.1)
    combined = CombinedNoiseModel(scale=1.0)
    
    # Generate samples
    n = 1024
    seismic_samples = seismic.sample(n)
    thermal_samples = thermal.sample(n)
    quantum_samples = quantum.sample(n)
    combined_samples = combined.sample(n)
    
    # Print stats
    print(f"Seismic noise RMS: {np.std(seismic_samples):.4f}")
    print(f"Thermal noise RMS: {np.std(thermal_samples):.4f}")
    print(f"Quantum noise RMS: {np.std(quantum_samples):.4f}")
    print(f"Combined noise RMS: {np.std(combined_samples):.4f}")
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    t = np.arange(n) / 256.0
    axes[0].plot(t, seismic_samples, 'b-', alpha=0.7, label='Seismic')
    axes[0].plot(t, thermal_samples, 'r-', alpha=0.7, label='Thermal')
    axes[0].plot(t, quantum_samples, 'g-', alpha=0.7, label='Quantum')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Noise Time Series')
    axes[0].legend()
    axes[0].grid(True)
    
    # PSD
    freqs = np.logspace(-1, 2, 100)  # 0.1 to 100 Hz
    axes[1].loglog(freqs, np.sqrt(seismic.get_psd(freqs)), 'b-', label='Seismic')
    axes[1].loglog(freqs, np.sqrt(thermal.get_psd(freqs)), 'r-', label='Thermal')
    axes[1].loglog(freqs, np.sqrt(quantum.get_psd(freqs)), 'g-', label='Quantum')
    axes[1].loglog(freqs, np.sqrt(combined.get_psd(freqs)), 'k--', label='Combined')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('ASD (1/√Hz)')
    axes[1].set_title('Noise Power Spectral Density')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('noise_test.png', dpi=150)
    print("\nNoise test plot saved to noise_test.png")
