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
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from cows3.detectorstates import CustomIFO, MultiDetectorStates
from cows3.noise import make_multi_noise_weights_from_asd
from cows3.sensitivity import pfd_Fstatistic
from cows3.snr import SignalToNoiseRatio

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def draw_unit_depth_rho2_from_isotropic_population(snr_calc, rng):
    """Draw one isotropic source and compute one unit-depth SNR^2 sample.

    At unit depth (D = 1), h0 = sqrt(Sn_ref). We draw source parameters from an
    isotropic population and compute SNR^2 directly.
    """
    cosi = rng.uniform(-1, 1)
    h0_unit = np.sqrt(snr_calc.effective_PSD)

    return snr_calc.compute_snr2(
        Alpha=rng.uniform(0, 2 * np.pi),
        Delta=np.arcsin(rng.uniform(-1, 1)),
        psi=rng.uniform(-np.pi / 4, np.pi / 4),
        phi0=rng.uniform(0, 2 * np.pi),
        aPlus=0.5 * h0_unit * (1 + cosi**2),
        aCross=h0_unit * cosi,
    )

# --- Custom detector definition (Palma, 10 km arms) ---
# Site coordinates: 39°38'08.1"N, 2°38'00.5"E.
# Bisector points toward: 39°48'28.3"N, 2°47'38.1"E.
# For a right-angle interferometer, arm azimuths are taken as bisector ± 45°.
CustomIFO(
    name="PB_10k",
    prefix="X0",
    latitude_rad=0.6917714301152558,
    longitude_rad=0.04596276103758956,
    elevation_m=0.0,
    xarm_azimuth_rad=1.406102938164868,
    yarm_azimuth_rad=6.118491918549558,
    xarm_alt_rad=0.0,
    yarm_alt_rad=0.0,
    xarm_midpoint_m=5000.0,
    yarm_midpoint_m=5000.0,
)


# --- Example parameters ---
sqrtSX_H1 = 1e-23
sqrtSX_L1 = 1.3e-23
sqrtSX_X0 = 1e-24

nsamples = 10_000

tstart = 1_238_166_018  # GPS start (O3 epoch)
Tobs_days = 10.0
Tsft = 1800
Tcoh_days = 0.5

depths = [20.0, 30.0, 50.0]
twoF_threshold = 60.0

hist_bins = 50

rng = np.random.default_rng(42)

# --- Setup ---
Tobs_s = Tobs_days * 86400.0  # days -> seconds
timestamps_array = np.arange(tstart, tstart + Tobs_s, Tsft, dtype=np.int64)

num_segments = max(1, int(round(Tobs_days / Tcoh_days)))

logger.info(f"Detectors: H1 (sqrtSX={sqrtSX_H1:.2e}), L1 (sqrtSX={sqrtSX_L1:.2e})")
logger.info(f"Tobs = {Tobs_days} d, Tsft = {Tsft} s, Tcoh = {Tcoh_days} d")
logger.info(f"SFTs per detector: {len(timestamps_array)}, Nseg = {num_segments}")
logger.info(f"Drawing {nsamples} isotropic source realizations...")


# --- Detector states and noise weights ---
mds = MultiDetectorStates(
    timestamps={"H1": timestamps_array, "L1": timestamps_array, "X0": timestamps_array},
    T_sft=Tsft,
)
mdss = mds.Series

noise_weights = make_multi_noise_weights_from_asd(
    mdss=mdss,
    sqrtSX_list=[sqrtSX_H1, sqrtSX_L1, sqrtSX_X0],
)

# --- SNR calculator ---
snr_calc = SignalToNoiseRatio(mdss=mdss, noise_weights=noise_weights)
logger.info(f"Effective reference PSD: Sn = {snr_calc.effective_PSD:.2e} Hz^-1")

# --- Compute unit-depth SNR^2 distribution ---
rho02 = np.array(
    [
        draw_unit_depth_rho2_from_isotropic_population(
            snr_calc=snr_calc,
            rng=rng,
        )
        for _ in trange(nsamples, desc="Drawing rho02 samples")
    ],
    dtype=float,
)
logger.info(f"rho0^2 range: [{rho02.min():.2f}, {rho02.max():.2f}]")
logger.info(f"rho0^2 median: {np.median(rho02):.2f}")

# --- Build PDF from histogram for pfd computation ---
counts, bin_edges = np.histogram(rho02, bins=hist_bins, density=True)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# --- False-dismissal probability ---
logger.info(f"\n{'Depth (1/√Hz)':>16s}  {'p_fd':>12s}")
logger.info("-" * 32)
for depth in depths:
    pfd = pfd_Fstatistic(
        twoF_threshold=twoF_threshold,
        depth=depth,
        num_segments=num_segments,
        unitD_rho2_bins=bin_centers,
        unitD_rho2_pdf=counts,
    )
    logger.info(f"{depth:16.1f}  {pfd:12.6e}")

# --- Matplotlib histogram ---
fig, ax = plt.subplots()
ax.hist(rho02, bins=hist_bins, density=True, alpha=0.7, histtype="step")
ax.set_xlabel(r"Unit-depth SNR$^2$ ($\rho_0^2$)")
ax.set_ylabel("Probability density")
ax.set_title(
    f"H1 ($\\sqrt{{S_X}}$={sqrtSX_H1:.1e}) + "
    f"L1 ($\\sqrt{{S_X}}$={sqrtSX_L1:.1e}) + "
    f"X0 ($\\sqrt{{S_X}}$={sqrtSX_X0:.1e}), "
    f"{nsamples} draws"
)
fig.tight_layout()
fig.savefig("result.pdf")
