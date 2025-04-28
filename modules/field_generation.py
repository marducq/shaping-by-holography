"""
Provides functions to create signal and reference fields.
"""

import numpy as np
import scipy.ndimage


def generate_speckle_field(nx, ny, grain_size, normalize=True):
    """
    Generate a random speckle field with spatial correlation.

    Parameters:
    -----------
    nx : int
        Number of points in x direction
    ny : int
        Number of points in y direction
    grain_size : float
    normalize : bool
    Returns:
    --------
    field : ndarray (complex)
        Complex speckle field
    """
    # Create random complex field
    temp = np.random.rand(ny, nx) - 0.5 + (np.random.rand(ny, nx) - 0.5) * 1j

    # Apply Gaussian filtering 
    field = scipy.ndimage.gaussian_filter(temp.real, grain_size / 2) + \
            1j * scipy.ndimage.gaussian_filter(temp.imag, grain_size / 2)

    # Normalize if requested
    if normalize:
        field = field / np.abs(field)

    return field


def generate_gaussian_field(xv, yv, center=(0.5, 0.5), sigma=(0.1, 0.1), k=(0, 0), phase_offset=0):
    """
    Generate a Gaussian beam field with optional tilt.

    Parameters:
    -----------
    xv : ndarray
        2D array of x coordinates
    yv : ndarray
        2D array of y coordinates
    center : tuple
        (x, y) center of the Gaussian beam
    sigma : tuple
        (σx, σy) standard deviations of the Gaussian beam
    k : tuple
        (kx, ky) wave vector components for tilt
    phase_offset : float
        Constant phase offset in radians

    Returns:
    --------
    field : ndarray (complex)
        Complex Gaussian field
    """
    # Calculate distance from center
    r_squared = (xv - center[0]) ** 2 / (2 * sigma[0] ** 2) + \
                (yv - center[1]) ** 2 / (2 * sigma[1] ** 2)

    # Calculate amplitude (Gaussian profile)
    amplitude = np.exp(-r_squared)

    # Calculate phase (linear tilt + offset)
    phase = k[0] * xv + k[1] * yv + phase_offset

    # Create complex field
    field = amplitude * np.exp(1j * phase)

    return field


def generate_reference_field(xv, yv, kx=0, ky=0, ref_type="plane"):
    """
    Generate a reference field for holography.

    Parameters:
    -----------
    xv : ndarray
        2D array of x coordinates
    yv : ndarray
        2D array of y coordinates
    kx : float
        Spatial frequency in x direction
    ky : float
        Spatial frequency in y direction
    ref_type : str
        Type of reference field: "plane", "spherical", or "gaussian"

    Returns:
    --------
    field : ndarray (complex)
        Complex reference field
    """
    if ref_type == "plane":
        # Plane wave with linear phase
        field = np.exp(1j * (kx * xv + ky * yv))

    elif ref_type == "spherical":
        # Spherical wave with quadratic phase
        center_x = np.mean(xv[0, :])
        center_y = np.mean(yv[:, 0])
        radius = np.sqrt((xv - center_x) ** 2 + (yv - center_y) ** 2)
        field = np.exp(1j * kx * radius)

    elif ref_type == "gaussian":
        # Gaussian beam
        center = (np.mean(xv[0, :]), np.mean(yv[:, 0]))
        sigma = (0.3, 0.3)  # Width
        field = generate_gaussian_field(xv, yv, center, sigma, (kx, ky))

    else:
        raise ValueError(f"Unknown reference type: {ref_type}")

    return field



