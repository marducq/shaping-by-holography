"""
Visualization module for off-axis holography simulations.
Provides functions for displaying holograms, fields, and reconstruction results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
import os


def plot_complex_field(field, title=None, phase_colormap='hsv', amplitude_colormap='viridis',
                       fig=None, figsize=(12, 5),
                       save_path=None):
    """
    Plot the amplitude and/or phase of a complex field.

    Parameters:
    -----------
    field : ndarray (complex)
        Complex field to plot
    title : str
        Overall title for the plot
    phase_colormap : str
        Colormap to use for phase plot
    amplitude_colormap : str
        Colormap to use for amplitude plot
    show_amplitude : bool
        Whether to display amplitude
    show_phase : bool
        Whether to display phase
    fig : matplotlib.figure.Figure
        Existing figure to plot on (optional)
    figsize : tuple
        Figure size if creating new figure
    save_path : str
        Path to save the figure (optional)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """


    if fig is None:
        fig = plt.figure(figsize=figsize)

    if title:
        fig.suptitle(title)


    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)


    # Plot amplitude
    amplitude = np.abs(field)
    im1 = ax1.imshow(amplitude, cmap=amplitude_colormap)
    ax1.set_title('Amplitude')
    plt.colorbar(im1, ax=ax1)

    # Plot phase
    phase = np.angle(field)

    im2 = ax2.imshow(phase, cmap=phase_colormap)
    ax2.set_title('Phase')
    plt.colorbar(im2, ax=ax2)


    plt.tight_layout()

    # Save figure 
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_fourier_spectrum(ft, extent=None, title=None,
                          colormap='viridis', figsize=(8, 6),
                          save_path=None):
    """
    Plot the magnitude of a Fourier spectrum.

    Parameters:
    -----------
    ft : ndarray (complex)
        Fourier transform to plot
    extent : tuple
        (xmin, xmax, ymin, ymax) extent of the plot
        Whether to use logarithmic scale
    title : str
        Title for the plot
    colormap : str
        Colormap to use
    fig : matplotlib.figure.Figure
        Existing figure to plot on (optional)
    figsize : tuple
        Figure size if creating new figure
    save_path : str
        Path to save the figure (optional)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(111)

    # Calculate magnitude with small offset to avoid log(0)
    magnitude = np.abs(ft)

    norm = LogNorm(vmin=magnitude.max() * 1e-5, vmax=magnitude.max())


    # Plot spectrum
    im = ax.imshow(magnitude, extent=extent, norm=norm, cmap=colormap)

    if title:
        ax.set_title(title)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_filtered_spectrum(original_ft, filtered_ft, mask=None, extent=None,
                           colormap='viridis', fig=None,
                           figsize=(16, 5), save_path=None):
    """
    Plot the original spectrum, filter mask, and filtered spectrum.

    Parameters:
    -----------
    original_ft : ndarray (complex)
        Original Fourier transform
    filtered_ft : ndarray (complex)
        Filtered Fourier transform
    mask : ndarray
        Filter mask (optional)
    extent : tuple
        (xmin, xmax, ymin, ymax) extent of the plot
        Whether to use logarithmic scale
    colormap : str
        Colormap to use
    fig : matplotlib.figure.Figure
        Existing figure to plot on (optional)
    figsize : tuple
        Figure size if creating new figure
    save_path : str
        Path to save the figure (optional)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    num_plots = 3

    if fig is None:
        fig = plt.figure(figsize=figsize)

    # Plot original spectrum
    ax1 = fig.add_subplot(1, num_plots, 1)
    magnitude = np.abs(original_ft)

    norm = LogNorm(vmin=magnitude.max() * 1e-5, vmax=magnitude.max())


    im1 = ax1.imshow(magnitude, extent=extent, norm=norm, cmap=colormap)
    ax1.set_title('Original Spectrum')
    plt.colorbar(im1, ax=ax1)

    # Plot mask if provided
    if mask is not None:
        ax2 = fig.add_subplot(1, num_plots, 2)
        im2 = ax2.imshow(mask, extent=extent, cmap='gray')
        ax2.set_title('Filter Mask')
        plt.colorbar(im2, ax=ax2)

        idx = 3
    else:
        idx = 2

    # Plot filtered spectrum
    ax3 = fig.add_subplot(1, num_plots, idx)
    filtered_magnitude = np.abs(filtered_ft)

    norm = LogNorm(vmin=filtered_magnitude.max() * 1e-5, vmax=filtered_magnitude.max())

    im3 = ax3.imshow(filtered_magnitude, extent=extent, norm=norm, cmap=colormap)
    ax3.set_title('Filtered Spectrum')
    plt.colorbar(im3, ax=ax3)

    plt.tight_layout()

    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_reconstruction_comparison(original_field, reconstructed_field, title=None,
                                   phase_colormap='hsv', fig=None, figsize=(15, 5),
                                   save_path=None):
    """
    Plot original field, reconstructed field, and their phase difference.

    Parameters:
    -----------
    original_field : ndarray (complex)
        Original complex field
    reconstructed_field : ndarray (complex)
        Reconstructed complex field
    title : str
        Overall title for the plot
    phase_colormap : str
        Colormap to use for phase plot
    fig : matplotlib.figure.Figure
        Existing figure to plot on (optional)
    figsize : tuple
        Figure size if creating new figure
    save_path : str
        Path to save the figure (optional)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    metrics : dict
        Dictionary of quality metrics
    """
    from modules.holography import evaluate_reconstruction

    if fig is None:
        fig = plt.figure(figsize=figsize)

    if title:
        fig.suptitle(title)

    # Plot original phase
    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(np.angle(original_field), cmap=phase_colormap)
    ax1.set_title('Original Phase')
    plt.colorbar(im1, ax=ax1)

    # Plot reconstructed phase
    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(np.angle(reconstructed_field), cmap=phase_colormap)
    ax2.set_title('Reconstructed Phase')
    plt.colorbar(im2, ax=ax2)

    # Plot phase difference
    ax3 = fig.add_subplot(133)
    phase_diff = np.angle(original_field / reconstructed_field) # Divide for minus in angle
    im3 = ax3.imshow(phase_diff, cmap='RdBu', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax3.set_title('Phase Difference')
    plt.colorbar(im3, ax=ax3)

    plt.tight_layout()

    # Compute quality metrics
    try:
        metrics = evaluate_reconstruction(original_field, reconstructed_field)

        # Add metrics as text on the figure
        info_text = f"RMS Phase Error: {metrics['rms_phase_error']:.4f} rad\n"
        info_text += f"RMS Amplitude Error: {metrics['rms_amplitude_error']:.4f}"

        fig.text(0.5, 0.01, info_text, ha='center', bbox=dict(facecolor='white', alpha=0.5))
    except:
        metrics = {}

    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, metrics

import numpy as np

def look_for_spot_Fspace(Ef,
                         central_cut=30,
                         side='right',
                         margin=10):
    """
    Recherche automatique du centre de l'ordre +1 (ou -1) dans le plan
    de Fourier d'un champ 2‑D.
    
    Parameters
    ----------
    Ef : 2‑D ndarray (complex or float)
        Champ dans l'espace fréquence (sortie de FFT2).
    central_cut : int, optional
        Rayon (en px) du disque à mettre à zéro autour de l'ordre 0
        pour éviter qu'il soit détecté (default: 30).
    side : {'right', 'left', 'both'}, optional
        Quelle moitié du plan regarder :
            'right'  -> kx > 0
            'left'   -> kx < 0
            'both'   -> recherche sur tout le plan
        Utile si +1 et −1 ont des intensités proches.
    margin : int, optional
        Pixels à ignorer autour de la ligne frontière entre les moitiés
        (évite de reprendre l'ordre 0 si le masque est trop petit).
    
    Returns
    -------
    xc, yc : int
        Coordonnées colonne, ligne (FFT shiftée) du centre du spot.
    """
    # Intensité (valeurs réelles)
    If = np.abs(Ef)**2
    
    # 1) masque le centre (ordre 0)
    N, M = If.shape
    yc0, xc0 = N//2, M//2
    If[yc0-central_cut:yc0+central_cut,
       xc0-central_cut:xc0+central_cut] = 0.0
    
    # 2) restreint l'aire de recherche si demandé
    if side.lower() == 'right':
        If[:, :xc0+margin] = 0.0   # garde uniquement kx>0
    elif side.lower() == 'left':
        If[:, xc0-margin:] = 0.0   # garde kx<0
    # sinon 'both' -> on ne touche pas
    
    # 3) pixel le plus lumineux
    yc, xc = np.unravel_index(np.argmax(If), If.shape)
    return xc, yc



