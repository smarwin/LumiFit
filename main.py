import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.patches import FancyArrowPatch
from scipy.signal import savgol_filter
import os


def _wavelength_to_rgb(wavelength, gamma=0.8):
    ''' taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
    This converts a given wavelength of light to an 
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    Additionally alpha value set to 0.5 outside range
    '''
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 750:
        A = 1.
    else:
        A = 0.5
    if wavelength < 380:
        wavelength = 380.
    if wavelength > 750:
        wavelength = 750.
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return (R, G, B, A)


def _gaussian_EuFit(x, h1, e01, sig1):
    return h1 * (np.exp(-np.power(x - e01, 2.) / (2 * np.power(sig1, 2.))) + 0.12 * np.exp(-np.power(x - e01 + 1.7 * sig1, 2.) / (2 * np.power(sig1, 2.))))


def _combined_gaussian_EuFit(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        y += _gaussian_EuFit(x, *params[i:i+3])
    return y


def _calculate_curve_details(x, y):
    curve = UnivariateSpline(x, y - np.max(y) / 2, s=0)
    roots = curve.roots()
    r1, r2 = roots[0], roots[1]
    return {
        'nm': {'max': round(x[np.argmax(y)]), 'fwhm': round(r2 - r1), 'area / a.u.': round(np.trapz(y, x)), 'x1': round(r1, 2), 'x2': round(r2, 2)},
        'cm-1': {'max': round(1 / x[np.argmax(y)] * 1e7), 'fwhm': round(1e7 * (1 / r1 - 1 / r2))}
    }


def import_emission_data(path):
    data = np.genfromtxt(path, delimiter=None, skip_header=22, usecols=(
        0, 1), dtype=[float, float], autostrip=True)

    return [data['f0'], data['f1']]


def import_excitation_data(path):
    data = np.genfromtxt(path, delimiter=None, skip_header=2, usecols=(
        0, 1), dtype=[float, float], autostrip=True)

    return [data['f0'], data['f1']]


def process_data(y, window=21, order=3):
    y_smoothed = savgol_filter(y, window, order)

    return (y_smoothed - np.min(y_smoothed)) / (np.max(y_smoothed) - np.min(y_smoothed))


def create_bounds(initial_guess):
    bounds = ([0]*len(initial_guess), [np.inf]*len(initial_guess))

    return bounds


def perform_fit(x, y, initial_guess, bounds=None):
    if not bounds:
        bounds = create_bounds(initial_guess)

    x_cm1 = (1 / x * 1e7)
    fit_parameters, _ = curve_fit(
        _combined_gaussian_EuFit, x_cm1, y, p0=initial_guess, bounds=bounds)

    # Total fit
    total_fit = _combined_gaussian_EuFit(x_cm1, *fit_parameters)
    total_fit_details = _calculate_curve_details(x, total_fit)
    fits = [(x, total_fit, total_fit_details)]
    if len(fit_parameters) > 3:
        for i in range(0, len(fit_parameters), 3):
            fit = _gaussian_EuFit(x_cm1, *fit_parameters[i:i+3])
            fit_detail = _calculate_curve_details(x, fit)
            fits.append((x, fit, fit_detail))

    return fits


def _create_fit_text(fits):
    fit_text = ''
    for i, fit in enumerate(fits):
        no = i if i != 0 else "Total"
        fit_text += f"Fit {no}\n"
        fit_text += f'max: {fit[2]["nm"]["max"]} nm\nfwhm: {fit[2]["nm"]["fwhm"]} nm /\n{fit[2]["cm-1"]["fwhm"]} cm$^-$$^1$\nAREA: {fit[2]["nm"]["area / a.u."]} a.u.\n\n'

    return fit_text


def _print_max(x, y):
    print("Emission Maximum:", x[np.argmax(y)], "nm")
    print("Emission Maximum:", 1/x[np.argmax(y)]*1e7, "cm-1")


def _create_arrow(fits):
    start_point = (fits[0][2]["nm"]["x1"], 0.5)
    end_point = (fits[0][2]["nm"]["x2"], 0.5)
    arrow_color = '#333'
    arrow_width = 0.7  # Width of the arrow
    arrow_head_width = 0.1  # Width of the arrowhead
    arrow = FancyArrowPatch(
        start_point,
        end_point,
        arrowstyle='<->',
        color=arrow_color,
        linewidth=arrow_width,
        mutation_scale=arrow_head_width * 60
    )

    return arrow


def _create_colormap(axs, x_start, x_end):
    # Create a colormap for the spectrum
    clim = (380, 750)
    norm = plt.Normalize(*clim)
    wl = np.arange(clim[0], clim[1]+1, 2)
    colorlist = list(zip(norm(wl), [_wavelength_to_rgb(w) for w in wl]))
    spectralmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "spectrum", colorlist)

    wavelengths = np.linspace(x_start, x_end, 3500)
    spectrum = np.ones_like(wavelengths)

    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(wavelengths, y)

    extent = (np.min(wavelengths), np.max(wavelengths), np.min(y), np.max(y))
    axs[1].imshow(X, clim=clim,  extent=extent,
                  cmap=spectralmap, aspect='auto')


def plot_results(exc, emi, fits, xlims, color, show_text=True):
    cm = 0.393701
    fig, axs = plt.subplots(
        2,  # Number of rows
        1,  # Number of columns
        figsize=(6*cm, 5*cm),  # figsize=(width, height),
        layout="tight",
        sharex=True,  # Share the x-axis between the plots
        gridspec_kw={'height_ratios': [15, 1], 'hspace': 0}  # subplot ratios
    )

    # Plot excitation
    axs[0].plot(exc[0], exc[1], color=color,
                lw=pp["lw"], ls="--", zorder=1)

    # Plot emission
    axs[0].scatter(emi[0], emi[1], color="#333",
                   lw=0.3, marker="+", s=2, zorder=2)

    # Plot fits
    if len(fits) == 0:
        _print_max(emi[0], emi[1])
    else:
        for i, fit in enumerate(fits):
            if i == 0:
                axs[0].plot(fit[0], fit[1], color=color,
                            lw=pp["lw"], ls="-", zorder=3)
            else:
                axs[0].plot(fit[0], fit[1], lw=pp["lw"]-0.25,
                            ls="-", zorder=-1, color="#333")

        # Plot max
        axs[0].vlines(fits[0][2]["nm"]["max"], -0.1, 1,
                      color=color, lw=pp["lw"], ls='--', zorder=0)

        # Plot arrow
        arrow = _create_arrow(fits)
        axs[0].add_patch(arrow)

        # Shade the fwhm
        axs[0].axvspan(fits[0][2]["nm"]["x1"],
                       fits[0][2]["nm"]["x2"],
                       facecolor=color,
                       alpha=0.1
                       )
    # Plot text
    if show_text:
        fit_text = _create_fit_text(fits)
        axs[0].text(0.75, 0.95, fit_text, ha="left", va="top",
                    transform=axs[0].transAxes, fontsize=4)

    # Plot colormap
    _create_colormap(axs, x_start=xlims[0], x_end=xlims[1])

    # Plot labels
    plt.xlabel('Wavelength / nm', fontsize=8)
    axs[0].set_ylabel('Intensity / a.u.', fontsize=8)

    # Plot ticks
    for ax in axs:
        ax.set_yticks([])
        ax.tick_params(axis='both', which='both', labelsize=8)

    # Set y limit
    axs[0].set_ylim(-0.03, 1.2)

    return fig


def save_fig(fig, filename="test", path="graphs", svg=False):
    if not os.path.exists(path):
        os.makedirs(path)

    fig.savefig(f"{path}/{filename}.png", dpi=300)
    if svg:
        fig.savefig(f"{path}/{filename}.svg")


def main(filename, exc_path, emi_path, xlims, color, smooth=11, fit=False, initial_guess=[1, 20000, 1000], bounds=None, path="graphs", show_text=True, svg=False):
    # Import data
    exc = import_excitation_data(exc_path)
    emi = import_emission_data(emi_path)

    # Process data
    exc[1] = process_data(exc[1], smooth)
    emi[1] = process_data(emi[1], smooth)

    # Perform fit
    if fit:
        fits = perform_fit(emi[0], emi[1], initial_guess, bounds)
    else:
        fits = []

    # Plot results
    fig = plot_results(exc, emi, fits, xlims, color, show_text=show_text)
    save_fig(fig, filename=filename, path=path, svg=svg)


pp = {
    "primary": "#0db575",
    "secondary": "#e9b872",
    "tertiary": "#065a82",
    "lw": 0.75,
}

fit = False
fit = True
xlims = (400, 700)
smooth = 5

main(
    "SrSi2PN5_v1",
    "data/PLE_data.dat",
    "data/PL_data.dat",
    xlims=xlims,
    color="#0db531",
    smooth=smooth,
    fit=fit,
    show_text=True,
    initial_guess=[1, 18000, 1000],  # in cm-1
    svg=True
)
