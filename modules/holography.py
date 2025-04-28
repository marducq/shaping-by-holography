"""
Provides functions for creating holograms and reconstructing fields.
"""

import numpy as np


def create_hologram(signal_field, reference_field):
    """
    Create a hologram by interfering signal and reference fields.

    Parameters:
    -----------
    signal_field : ndarray (complex)
        Complex signal field
    reference_field : ndarray (complex)
        Complex reference field

    Returns:
    --------
    hologram : ndarray (real)
        Intensity pattern resulting from interference
    """
    # Calculate interference pattern (intensity)
    hologram = np.abs(signal_field + reference_field) ** 2

    return hologram


def compute_fourier_transform(hologram, padding_factor=1):
    """
    Compute the Fourier transform of a hologram with optional zero-padding.
    Parameters:
    -----------
    hologram : ndarray (real)
        Hologram intensity pattern
    padding_factor : int
        Zero-padding factor (1 = no padding)

    Returns:
    --------
    ft : ndarray (complex)
        Fourier transform of the hologram
    """
    ny, nx = hologram.shape

    # Compute FFT with padding
    ft = np.fft.fftshift(np.fft.fft2(hologram, s=[ny * padding_factor, nx * padding_factor]))

    return ft


def create_filter_mask(fxv, fyv, center_freq, width, shape="rect"):
    """
    Create a filter mask in the Fourier domain.

    Parameters:
    -----------
    fxv : ndarray
        2D array of x frequencies
    fyv : ndarray
        2D array of y frequencies
    center_freq : tuple
        (fx, fy) center frequency of the filter
    width : float or tuple
        Width of the filter (scalar or (width_x, width_y))
    shape : str
        Shape of the filter: "rect", "gaussian"
    Returns:
    --------
    mask : ndarray
        Binary or continuous filter mask
    """
    # Convert scalar width to tuple if needed
    if isinstance(width, (int, float)):
        width = (width, width)

    # Calculate distance from center frequency
    dist_x = np.abs(fxv - center_freq[0])
    dist_y = np.abs(fyv - center_freq[1])

    if shape == "rect":
        # Rectangular filter (binary)
        mask = (dist_x < width[0] / 2) & (dist_y < width[1] / 2)

    elif shape == "circular":
        # Circular filter (binary)
        # Calculate Euclidean distance from center frequency
        distance = np.sqrt((fxv - center_freq[0]) ** 2 + (fyv - center_freq[1]) ** 2)
        # Use the first width component as the radius
        radius = width[0] / 2
        # Create binary mask where distance < radius
        mask = distance < radius

    return mask


def apply_filter(ft, mask):
    """
    Apply a filter mask to a Fourier transform.

    Parameters:
    -----------
    ft : ndarray (complex)
        Fourier transform of the hologram
    mask : ndarray
        Filter mask

    Returns:
    --------
    filtered_ft : ndarray (complex)
        Filtered Fourier transform
    """
    return ft * mask


def shift_filtered_spectrum(filtered_ft, mask, shift_vector):
    """
    Shift the filtered spectrum to center it at zero frequency.
    See Popoff GIT for details

    Parameters:
    -----------
    filtered_ft : ndarray (complex)
        Filtered Fourier transform
    mask : ndarray
        Filter mask used for filtering
    shift_vector : tuple
        (shift_x, shift_y) number of pixels to shift in each direction

    Returns:
    --------
    shifted_ft : ndarray (complex)
        Shifted Fourier transform
    """
    # Create an empty array for the shifted spectrum
    shifted_ft = np.zeros_like(filtered_ft)

    # Get indices where mask is non-zero
    I = np.nonzero(mask)

    # Calculate shifted indices
    shift_x, shift_y = shift_vector
    shifted_idx_y = I[0]
    shifted_idx_x = (I[1] + np.round(shift_x)).astype(int)

    # Check bounds
    valid = (shifted_idx_x >= 0) & (shifted_idx_x < filtered_ft.shape[1])

    # Copy values with shift
    shifted_ft[shifted_idx_y[valid], shifted_idx_x[valid]] = filtered_ft[I[0][valid], I[1][valid]]

    return shifted_ft


def reconstruct_field(ft, original_shape):
    """
    Reconstruct the field by inverse Fourier transform.

    Parameters:
    -----------
    ft : ndarray (complex)
        Fourier transform to inverse transform
    original_shape : tuple
        (ny, nx) shape of the original field

    Returns:
    --------
    field : ndarray (complex)
        Reconstructed complex field
    """
    # Perform inverse FFT
    temp = np.fft.ifft2(np.fft.ifftshift(ft))

    # Extract original size
    ny, nx = original_shape
    field = temp[:ny, :nx]

    return field


def evaluate_reconstruction(original_field, reconstructed_field):
    """
    Evaluate the quality of field reconstruction.

    Parameters:
    -----------
    original_field : ndarray (complex)
        Original complex field
    reconstructed_field : ndarray (complex)
        Reconstructed complex field

    Returns:
    --------
    metrics : dict
        Dictionary of quality metrics
    """
    # Calculate phase error
    phase_diff = np.angle(original_field / reconstructed_field)

    # Calculate metrics
    mean_phase_error = np.mean(np.abs(phase_diff))
    rms_phase_error = np.sqrt(np.mean(phase_diff ** 2))

    # Calculate amplitude error
    amp_orig = np.abs(original_field)
    amp_recon = np.abs(reconstructed_field)

    # Normalize amplitudes for comparison
    if np.max(amp_orig) > 0:
        amp_orig = amp_orig / np.max(amp_orig)
    if np.max(amp_recon) > 0:
        amp_recon = amp_recon / np.max(amp_recon)

    amp_diff = amp_orig - amp_recon
    rms_amp_error = np.sqrt(np.mean(amp_diff ** 2))

    # Return metrics as dictionary
    metrics = {
        'mean_phase_error': mean_phase_error,
        'rms_phase_error': rms_phase_error,
        'rms_amplitude_error': rms_amp_error
    }

    return metrics