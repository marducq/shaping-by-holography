"""
Grid generation module for off-axis holography simulations.
"""

import numpy as np


def create_spatial_grid(nx, ny, x_range=(0, 1), y_range=(0, 1)):
    """
    Create a spatial grid for the simulation.

    Parameters:
    -----------
    nx : int
        Number of points in x direction
    ny : int
        Number of points in y direction
    x_range : tuple
        (min_x, max_x) range of x coordinates
    y_range : tuple
        (min_y, max_y) range of y coordinates

    Returns:
    --------
    X : ndarray
        1D array of x coordinates
    Y : ndarray
        1D array of y coordinates
    xv : ndarray
        2D array of x coordinates (meshgrid)
    yv : ndarray
        2D array of y coordinates (meshgrid)
    """
    X = np.linspace(x_range[0], x_range[1], num=nx)
    Y = np.linspace(y_range[0], y_range[1], num=ny)
    xv, yv = np.meshgrid(X, Y)

    return X, Y, xv, yv


def create_frequency_grid(nx, ny, padding_factor=1):
    """
    Create a frequency grid for the Fourier domain.

    Parameters:
    -----------
    nx : int
        Number of points in x direction in spatial domain
    ny : int
        Number of points in y direction in spatial domain
    padding_factor : int
        Zero-padding factor for FFT

    Returns:
    --------
    fx : ndarray
        1D array of x frequencies
    fy : ndarray
        1D array of y frequencies
    fxv : ndarray
        2D array of x frequencies (meshgrid)
    fyv : ndarray
        2D array of y frequencies (meshgrid)
    """
    # Calculate frequency ranges based on spatial grid dimensions
    fx = np.arange(-0.5 * nx, 0.5 * nx, 1 / padding_factor)
    fy = np.arange(-0.5 * ny, 0.5 * ny, 1 / padding_factor)
    fxv, fyv = np.meshgrid(fx, fy)

    return fx, fy, fxv, fyv


