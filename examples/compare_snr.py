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


def compute_unitD_rho2_distribution(snr_calc, population, Sn_ref):
    """Compute unit-depth SNR^2 for each source in the population.

    At unit depth (D = 1), h0 = sqrt(Sn_ref).  We set amplitudes
    accordingly and compute SNR^2 directly.
    """
    h0_unit = np.sqrt(Sn_ref)
    nsamples = len(population["Alpha"])
    rho02 = np.empty(nsamples)

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


dict_dets = {"H1": 1e-23, "L1": 5e-24}
combos = ["H1", "L1", "H1,L1"]

Tobs_s = 365 * 86400  # days -> seconds
tstart = 1_238_166_018  # GPS start (O3 epoch)
Tsft = 1800
timestamps_array = np.arange(tstart, tstart + Tobs_s, Tsft)

rng = np.random.default_rng(42)
population = draw_isotropic_population(100, rng)

rho02 = []
for icombo in combos:
    dict_ts = {}
    
    for idet in icombo.split(","): dict_ts[idet] = timestamps_array 
    
    mds = MultiDetectorStates(
        timestamps=dict_ts,
        T_sft=1800,
    )

    mdss = mds.Series

    if len(icombo.split(","))>1:
        sqrtSn_list = []
        for idet in icombo.split(","): sqrtSn_list.append( dict_dets[idet] )
        noise_weights = make_multi_noise_weights_from_asd(
            mdss=mdss,
            sqrtSX_list=sqrtSn_list,
        )
        Sn_ref = Tsft / noise_weights.Sinv_Tsft
        snr_calc = SignalToNoiseRatio(mdss=mdss, noise_weights=noise_weights)
    
    else:
        Sn_ref = dict_dets[icombo]**2
        snr_calc = SignalToNoiseRatio(mdss=mdss, assumeSqrtSX=Sn_ref**.5)

    
    rho02.append( compute_unitD_rho2_distribution(snr_calc, population, Sn_ref) )

combos = np.array(combos)
rho02 = np.array(rho02)
harm_mean_psd = 2 / ( 1/dict_dets["H1"] + 1/dict_dets["L1"] )
joint_snr_from_single_det = ( rho02[combos=="H1"][0] * dict_dets["H1"] + rho02[combos=="L1"][0] * dict_dets["L1"] ) / harm_mean_psd
snr_ratio = joint_snr_from_single_det / rho02[combos=="H1,L1"][0]
mean = np.mean(snr_ratio) ; std = np.std(snr_ratio)
print(f"Ratio between sum from single-det & directly joint analysis is: {mean:.3f} \pm {std:.3f}")