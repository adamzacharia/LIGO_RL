"""
Frequency Domain Filters for RL Rewards.

Implements IIR filters for extracting frequency-domain properties
of control signals, following the Deep Loop Shaping approach.

Key bands:
- Low frequency (<3 Hz): Control authority band
- Observation band (8-30 Hz): Where we want minimal noise
- High frequency (>40 Hz): Avoid artifacts
"""

import numpy as np
from scipy import signal
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class FilterState:
    """State for stateful IIR filter."""
    zi: np.ndarray  # Filter initial conditions
    
    
def create_bandpass_filter(
    low_freq: float,
    high_freq: float,
    sample_rate: float = 256.0,
    order: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a Butterworth bandpass filter.
    
    Args:
        low_freq: Lower cutoff frequency (Hz)
        high_freq: Upper cutoff frequency (Hz)
        sample_rate: Sample rate (Hz)
        order: Filter order
        
    Returns:
        (b, a) filter coefficients
    """
    nyquist = sample_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Clip to valid range
    low = np.clip(low, 0.01, 0.99)
    high = np.clip(high, low + 0.01, 0.99)
    
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def create_lowpass_filter(
    cutoff_freq: float,
    sample_rate: float = 256.0,
    order: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a Butterworth lowpass filter.
    
    Args:
        cutoff_freq: Cutoff frequency (Hz)
        sample_rate: Sample rate (Hz)
        order: Filter order
        
    Returns:
        (b, a) filter coefficients
    """
    nyquist = sample_rate / 2
    cutoff = np.clip(cutoff_freq / nyquist, 0.01, 0.99)
    
    b, a = signal.butter(order, cutoff, btype='low')
    return b, a


def create_highpass_filter(
    cutoff_freq: float,
    sample_rate: float = 256.0,
    order: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a Butterworth highpass filter.
    
    Args:
        cutoff_freq: Cutoff frequency (Hz)
        sample_rate: Sample rate (Hz)
        order: Filter order
        
    Returns:
        (b, a) filter coefficients
    """
    nyquist = sample_rate / 2
    cutoff = np.clip(cutoff_freq / nyquist, 0.01, 0.99)
    
    b, a = signal.butter(order, cutoff, btype='high')
    return b, a


def apply_filter(
    signal_in: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
    zi: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply IIR filter with state tracking for real-time use.
    
    Args:
        signal_in: Input signal (1D or 2D with samples on axis 0)
        b: Numerator coefficients
        a: Denominator coefficients
        zi: Initial filter state (optional)
        
    Returns:
        (filtered_signal, final_state)
    """
    if zi is None:
        zi = signal.lfilter_zi(b, a) * signal_in[0] if len(signal_in) > 0 else signal.lfilter_zi(b, a)
    
    filtered, zf = signal.lfilter(b, a, signal_in, zi=zi)
    return filtered, zf


class FilterBank:
    """
    Collection of filters for frequency-domain rewards.
    
    Implements the filter bank from Deep Loop Shaping:
    - Low-pass for alignment tracking
    - Band-pass for observation band (8-30 Hz)
    - High-pass for artifact avoidance (>40 Hz)
    """
    
    def __init__(
        self,
        sample_rate: float = 256.0,
        observation_band: Tuple[float, float] = (8.0, 30.0),
        high_freq_cutoff: float = 40.0,
        low_freq_cutoff: float = 3.0
    ):
        """
        Initialize filter bank.
        
        Args:
            sample_rate: Control loop sample rate (Hz)
            observation_band: (low, high) for observation band filter
            high_freq_cutoff: Cutoff for high-frequency artifact filter
            low_freq_cutoff: Cutoff for low-frequency alignment filter
        """
        self.sample_rate = sample_rate
        
        # Create filters
        self.lp_b, self.lp_a = create_lowpass_filter(
            low_freq_cutoff, sample_rate
        )
        
        self.bp_b, self.bp_a = create_bandpass_filter(
            observation_band[0], observation_band[1], sample_rate
        )
        
        self.hp_b, self.hp_a = create_highpass_filter(
            high_freq_cutoff, sample_rate
        )
        
        # Initialize filter states
        self.reset_states()
        
    def reset_states(self):
        """Reset all filter states."""
        self.lp_zi = signal.lfilter_zi(self.lp_b, self.lp_a)
        self.bp_zi = signal.lfilter_zi(self.bp_b, self.bp_a)
        self.hp_zi = signal.lfilter_zi(self.hp_b, self.hp_a)
        
    def process_sample(self, x: float) -> dict:
        """
        Process a single sample through all filters.
        
        Args:
            x: Input sample (scalar)
            
        Returns:
            Dict with filter outputs
        """
        # Process through each filter
        lp_out, self.lp_zi = signal.lfilter(
            self.lp_b, self.lp_a, [x], zi=self.lp_zi
        )
        
        bp_out, self.bp_zi = signal.lfilter(
            self.bp_b, self.bp_a, [x], zi=self.bp_zi
        )
        
        hp_out, self.hp_zi = signal.lfilter(
            self.hp_b, self.hp_a, [x], zi=self.hp_zi
        )
        
        return {
            'lowpass': float(lp_out[0]),
            'bandpass': float(bp_out[0]),
            'highpass': float(hp_out[0]),
            'raw': float(x)
        }
    
    def process_batch(self, x: np.ndarray) -> dict:
        """
        Process a batch of samples through all filters.
        
        Args:
            x: Input signal array
            
        Returns:
            Dict with filter output arrays
        """
        lp_out, _ = apply_filter(x, self.lp_b, self.lp_a)
        bp_out, _ = apply_filter(x, self.bp_b, self.bp_a)
        hp_out, _ = apply_filter(x, self.hp_b, self.hp_a)
        
        return {
            'lowpass': lp_out,
            'bandpass': bp_out,
            'highpass': hp_out,
            'raw': x
        }
    
    def compute_band_energies(self, x: np.ndarray) -> dict:
        """
        Compute RMS energy in each frequency band.
        
        Args:
            x: Input signal array
            
        Returns:
            Dict with band energies
        """
        outputs = self.process_batch(x)
        
        return {
            'lowpass_rms': float(np.sqrt(np.mean(outputs['lowpass']**2))),
            'bandpass_rms': float(np.sqrt(np.mean(outputs['bandpass']**2))),
            'highpass_rms': float(np.sqrt(np.mean(outputs['highpass']**2))),
            'total_rms': float(np.sqrt(np.mean(x**2)))
        }
    
    def get_frequency_response(
        self,
        filter_type: str = 'bandpass',
        num_points: int = 512
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get frequency response of a filter.
        
        Args:
            filter_type: 'lowpass', 'bandpass', or 'highpass'
            num_points: Number of frequency points
            
        Returns:
            (frequencies, magnitude_db)
        """
        if filter_type == 'lowpass':
            b, a = self.lp_b, self.lp_a
        elif filter_type == 'bandpass':
            b, a = self.bp_b, self.bp_a
        elif filter_type == 'highpass':
            b, a = self.hp_b, self.hp_a
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        w, h = signal.freqz(b, a, worN=num_points, fs=self.sample_rate)
        magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)
        
        return w, magnitude_db


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Test filter bank
    print("Testing frequency filters...")
    
    fb = FilterBank(sample_rate=256.0)
    
    # Create test signal with multiple frequency components
    t = np.linspace(0, 2, 512)  # 2 seconds at 256 Hz
    signal_test = (
        1.0 * np.sin(2 * np.pi * 1 * t) +    # 1 Hz (low freq)
        0.5 * np.sin(2 * np.pi * 15 * t) +   # 15 Hz (observation band)
        0.3 * np.sin(2 * np.pi * 60 * t)     # 60 Hz (high freq)
    )
    
    # Process and compute energies
    energies = fb.compute_band_energies(signal_test)
    print("\nBand energies:")
    for band, energy in energies.items():
        print(f"  {band}: {energy:.4f}")
    
    # Plot frequency responses
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Filter responses
    for ftype, color in [('lowpass', 'blue'), ('bandpass', 'green'), ('highpass', 'red')]:
        freq, mag = fb.get_frequency_response(ftype)
        axes[0].plot(freq, mag, color=color, label=ftype)
    
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_title('Filter Frequency Responses')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_xlim(0, 128)
    axes[0].set_ylim(-60, 5)
    
    # Test signal and filtered outputs
    outputs = fb.process_batch(signal_test)
    axes[1].plot(t, signal_test, 'k-', alpha=0.3, label='raw')
    axes[1].plot(t, outputs['bandpass'], 'g-', label='bandpass (8-30 Hz)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Bandpass Filtered Signal')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('filter_test.png', dpi=150)
    print("\nFilter test plot saved to filter_test.png")
    
    print("\nFrequency filter tests completed!")
