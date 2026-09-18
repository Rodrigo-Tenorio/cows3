#!/usr/bin/env python
"""
Quick example: CW sensitivity estimation with two detectors at different noise levels.

This example demonstrates the use of cows3 to:

1. Set up a two-detector network (LIGO Hanford H1, LIGO Livingston L1)
   with different amplitude spectral densities (ASDs).
2. Compute the unit-depth SNR^2 distribution for an isotropic population
   of CW sources.
3. Estimate the false-dismissal probability for selected sensitivity depths.

The sensitivity estimation follows Sec. II.3 of Mirasola & Tenorio (2024),
arXiv:2405.18934, Phys. Rev. D 110, 124049.

Usage:
    python custom_detector.py
    python custom_detector.py --sqrtSX 1e-23 1.5e-23 --nsamples 1000
    python custom_detector.py --save-plot snr2_histogram.png --no-show-plot
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from cows3.detectorstates import MultiDetectorStates
from cows3.noise import make_multi_noise_weights_from_asd
from cows3.snr import SignalToNoiseRatio


def draw_isotropic_population(nsamples, rng):
    """Draw CW source parameters from an isotropic population.

    Returns dict with keys: Alpha, Delta, cosi, psi, phi0.
    """
    return {
        "Alpha": rng.uniform(0, 2 * np.pi, nsamples),
        "Delta": np.arcsin(rng.uniform(-1, 1, nsamples)),
        "cosi": rng.uniform(-1, 1, nsamples),
        "psi": rng.uniform(-np.pi / 4, np.pi / 4, nsamples),
        "phi0": rng.uniform(0, 2 * np.pi, nsamples),
    }


def compute_unit_depth_rho2_distribution(snr_calc, population, Sn_ref):
    """Compute unit-depth SNR^2 for each source in the population.

    At unit depth (D = 1), h0 = sqrt(Sn_ref). We set amplitudes accordingly and
    compute SNR^2 directly.
    """
    h0_unit = np.sqrt(Sn_ref)
    nsamples = population["Alpha"].size
    rho02 = np.empty(nsamples, dtype=float)

    for ii in range(nsamples):
        cosi = population["cosi"][ii]
        aPlus = 0.5 * h0_unit * (1 + cosi**2)
        aCross = h0_unit * cosi
        rho02[ii] = snr_calc.compute_snr2(
            Alpha=float(population["Alpha"][ii]),
            Delta=float(population["Delta"][ii]),
            psi=float(population["psi"][ii]),
            phi0=float(population["phi0"][ii]),
            aPlus=aPlus,
            aCross=aCross,
        )

    return rho02


def build_parser():
    parser = argparse.ArgumentParser(
        description="CW sensitivity with two detectors at different noise levels."
    )
    parser.add_argument(
        "--sqrtSX",
        type=float,
        nargs=2,
        default=[1e-23, 1.3e-23],
        metavar=("H1", "L1"),
        help="Single-sided ASD for H1 and L1 (strain/sqrt(Hz)). Default: 1e-23 1.3e-23",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=500,
        help="Number of random source draws. Default: 500",
    )
    parser.add_argument(
        "--Tobs",
        type=float,
        default=10,
        help="Observation time in days. Default: 10",
    )
    parser.add_argument(
        "--Tsft",
        type=int,
        default=1800,
        help="SFT duration in seconds. Default: 1800",
    )
    parser.add_argument(
        "--Tcoh",
        type=float,
        default=0.5,
        help="Coherence time in days for the semicoherent search. Default: 0.5",
    )
    parser.add_argument(
        "--depths",
        type=float,
        nargs="+",
        default=[20, 30, 50],
        help="Sensitivity depths (1/sqrt(Hz)). Default: 20 30 50",
    )
    parser.add_argument(
        "--twoF-threshold",
        type=float,
        default=60,
        help="Semicoherent 2F threshold. Default: 60",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=50,
        help="Number of histogram bins. Default: 50",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )
    parser.add_argument(
        "--show-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display the histogram plot (use --no-show-plot to disable).",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        metavar="PATH",
        help="Save the histogram figure to this path.",
    )
    return parser


def validate_args(args):
    if args.nsamples <= 0:
        raise ValueError("--nsamples must be positive.")
    if args.Tsft <= 0:
        raise ValueError("--Tsft must be positive.")
    if args.Tobs <= 0:
        raise ValueError("--Tobs must be positive.")
    if args.Tcoh <= 0:
        raise ValueError("--Tcoh must be positive.")
    if args.hist_bins <= 0:
        raise ValueError("--hist-bins must be positive.")
    if args.twoF_threshold <= 0:
        raise ValueError("--twoF-threshold must be positive.")
    if np.any(np.asarray(args.sqrtSX, dtype=float) <= 0):
        raise ValueError("--sqrtSX must contain positive values.")
    if np.any(np.asarray(args.depths, dtype=float) <= 0):
        raise ValueError("--depths must contain positive values.")


# values for L1 & H1 taken from https://lscsoft.docs.ligo.org/lalsuite/dev/lal/_l_a_l_detectors_8h_source.html 
def get_nominal_H1_infos():
    det_H1 = {"name": "Hanford",
            "latitude_rad" : 0.81079526383,
            "longitude_rad": -2.08405676917,
            "elevation_m"  : 142.554,
            "xarm_azimuth_rad" : 5.65487724844,
            "yarm_azimuth_rad" : 4.08408092164,
            "xarm_alt_rad" : -0.00061950000,
            "yarm_alt_rad" :0.00001250000,
            }
    return det_H1

def get_nominal_L1_infos():
    det_L1 = {"name": "Livingston",
            "latitude_rad" : 0.53342313506,
            "longitude_rad": -1.58430937078,
            "elevation_m"  : -6.574,
            "xarm_azimuth_rad" : 4.40317772346,
            "yarm_azimuth_rad" : 2.83238139666,
            "xarm_alt_rad" : -0.00031210000,
            "yarm_alt_rad" : -0.00061070000,
            }
    return det_L1


def main():
    args = build_parser().parse_args()
    validate_args(args)

    sqrtSX_H1, sqrtSX_L1 = map(float, args.sqrtSX)

    rng = np.random.default_rng(args.seed)

    # --- Setup ---
    Tobs_s = float(args.Tobs) * 86400.0  # days -> seconds
    tstart = 1_238_166_018  # GPS start (O3 epoch)
    timestamps_array = np.arange(tstart, tstart + Tobs_s, args.Tsft, dtype=np.int64)

    num_segments = max(1, int(round(args.Tobs / args.Tcoh)))

    print(f"Detectors: H1 (sqrtSX={sqrtSX_H1:.2e}), L1 (sqrtSX={sqrtSX_L1:.2e})")
    print(f"Tobs = {args.Tobs} d, Tsft = {args.Tsft} s, Tcoh = {args.Tcoh} d")
    print(f"SFTs per detector: {len(timestamps_array)}, Nseg = {num_segments}")
    print(f"Drawing {args.nsamples} isotropic source realizations...")

    # --- Detector states and noise weights ---
    mds = MultiDetectorStates(
        timestamps={"H1": timestamps_array, "L1": timestamps_array},
        T_sft=args.Tsft,
    )
    mdss = mds.Series

    noise_weights = make_multi_noise_weights_from_asd(
        mdss=mdss,
        sqrtSX_list=[sqrtSX_H1, sqrtSX_L1],
    )


    # --- Now do the same for the "custom" implementation of the detectors ---
    mds = MultiDetectorStates(
            timestamps={"X2": timestamps_array, "Y2": timestamps_array},
            T_sft=args.Tsft,
            custom_ifos={"X2": get_nominal_H1_infos(), "Y2": get_nominal_L1_infos()}
        )
    
    mdss_custom_ifo = mds.Series

    noise_weights_custom_ifo = make_multi_noise_weights_from_asd(
        mdss=mdss,
        sqrtSX_list=[sqrtSX_H1, sqrtSX_L1],
    )

    # Reference PSD derived from noise weights (effective harmonic mean).
    Sn_ref = args.Tsft / noise_weights.Sinv_Tsft
    Sn_ref_custom = args.Tsft / noise_weights_custom_ifo.Sinv_Tsft
    print(f"Effective reference PSD: Sn = {Sn_ref:.2e} Hz^-1")
    print(f"Effective reference PSD with custom implementation: Sn = {Sn_ref_custom:.2e} Hz^-1")

    # --- SNR calculator ---
    snr_calc = SignalToNoiseRatio(mdss=mdss, noise_weights=noise_weights)
    snr_calc_custom_ifo = SignalToNoiseRatio(mdss=mdss_custom_ifo, noise_weights=noise_weights)

    # --- Compute unit-depth SNR^2 distribution ---
    population = draw_isotropic_population(args.nsamples, rng)
    rho02 = compute_unit_depth_rho2_distribution(snr_calc, population, Sn_ref)
    rho02_custom_ifo = compute_unit_depth_rho2_distribution(snr_calc_custom_ifo, population, Sn_ref)
    print(f"rho0^2 range: [{rho02.min():.2f}, {rho02.max():.2f}]")
    print(f"rho0^2 median: {np.median(rho02):.2f}")

    print(f"rho0^2 range w custom ifos: [{rho02_custom_ifo.min():.2f}, {rho02_custom_ifo.max():.2f}]")
    print(f"rho0^2 median w custom ifos: {np.median(rho02_custom_ifo):.2f}")

    # --- Matplotlib histogram ---
    fig, ax = plt.subplots()
    ax.hist(rho02, bins=args.hist_bins, density=True, alpha=0.7, edgecolor="black", label="lal-implemented ifos")
    ax.hist(rho02_custom_ifo, bins=args.hist_bins, density=True, alpha=0.4, edgecolor="black", label="w custom ifos")
    ax.set_xlabel(r"Unit-depth SNR$^2$ ($\rho_0^2$)")
    ax.set_ylabel("Probability density")
    ax.set_title(
        f"H1 ($\\sqrt{{S_X}}$={sqrtSX_H1:.1e}) + "
        f"L1 ($\\sqrt{{S_X}}$={sqrtSX_L1:.1e}), "
        f"{args.nsamples} draws"
    )
    ax.legend()
    fig.tight_layout()

    if args.save_plot:
        fig.savefig(args.save_plot, dpi=150)
        print(f"\nFigure saved to {args.save_plot}")

    if args.show_plot:
        plt.show()


if __name__ == "__main__":
    main()
