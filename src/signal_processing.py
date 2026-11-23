"""
Feature extraction module for computing the Stockwell Transform (S-Transform).
"""

import numpy as np
import warnings

def perform_s_transform(
    signal: np.ndarray, min_freq: int, max_freq: int, fs: int
) -> np.ndarray:
    """
    Computes the S-Transform using a vectorized NumPy implementation.

    Args:
        signal: 1D input signal array.
        min_freq: The minimum frequency for analysis (Hz).
        max_freq: The maximum frequency for analysis (Hz).
        fs: The sampling frequency of the signal (Hz).

    Returns:
        A complex-valued S-Transform matrix of shape [frequency_bins, time_samples].
    """
    n_samples = len(signal)
    try:
        # Pre-compute the Fast Fourier Transform (FFT) of the signal.
        signal_fft = np.fft.fft(signal)
        freqs_fft = np.fft.fftfreq(n_samples, 1 / fs)

        # Define the frequency range for the S-Transform output.
        freq_range = np.arange(min_freq, max_freq + 1)
        st_matrix = np.zeros((len(freq_range), n_samples), dtype=complex)

        # Handle the DC component (0 Hz) as a special case.
        if min_freq == 0:
            st_matrix[0, :] = np.mean(signal)
            freq_start_idx = 1
        else:
            freq_start_idx = 0

        # Vectorized calculation for all other frequencies.
        for i, f in enumerate(freq_range[freq_start_idx:], start=freq_start_idx):
            # The Gaussian window's standard deviation is frequency-dependent.
            sigma = 1.0 / f
            
            # Create the Gaussian window in the frequency domain.
            gauss_window_fft = np.exp(-2 * (np.pi**2) * (sigma**2) * ((freqs_fft - f)**2))
            
            # Apply the window and perform inverse FFT to get the S-Transform for this frequency.
            st_matrix[i, :] = np.fft.ifft(signal_fft * gauss_window_fft)
        
        if np.any(np.isnan(st_matrix)):
            warnings.warn("NaN values found in S-Transform result; replacing with 0.")
            st_matrix = np.nan_to_num(st_matrix)

        return st_matrix

    except Exception as e:
        warnings.warn(f"S-Transform computation failed: {e}. Returning a zero matrix.")
        return np.zeros((max_freq - min_freq + 1, n_samples), dtype=complex)