import numpy as np
import pytest

from cows3.detectorstates import MultiDetectorStates
from cows3.noise import make_multi_noise_weights_from_asd
from cows3.snr import SignalToNoiseRatio


def make_isotropic_population(nsamples, rng):
    return {
        "Alpha": rng.uniform(0, 2 * np.pi, nsamples),
        "Delta": np.arcsin(rng.uniform(-1, 1, nsamples)),
        "cosi": rng.uniform(-1, 1, nsamples),
        "psi": rng.uniform(-np.pi / 4, np.pi / 4, nsamples),
        "phi0": rng.uniform(0, 2 * np.pi, nsamples),
    }


def compute_unit_depth_rho2(snr_object, population, noise_level):
    h0_unit = np.sqrt(noise_level)
    rho02 = np.empty(len(population["Alpha"]))

    for ii in range(len(population["Alpha"])):
        cosi = population["cosi"][ii]
        aPlus = 0.5 * h0_unit * (1 + cosi**2)
        aCross = h0_unit * cosi
        rho02[ii] = snr_object.compute_snr2(
            Alpha=float(population["Alpha"][ii]),
            Delta=float(population["Delta"][ii]),
            psi=float(population["psi"][ii]),
            phi0=float(population["phi0"][ii]),
            aPlus=aPlus,
            aCross=aCross,
        )

    return rho02


@pytest.fixture
def signal_params():
    return {
        "aPlus": 0.5 * 1e-23,
        "aCross": 1e-23,
        "psi": 0,
        "phi0": 0,
        "Alpha": 0,
        "Delta": 0,
    }


@pytest.fixture
def mds():

    Tsft = 1_800
    tstart = 700_000_000
    ts = np.arange(tstart, tstart + 4 * Tsft, Tsft)

    return MultiDetectorStates(
        timestamps={detector: ts for detector in ["H1", "L1"]},
        T_sft=Tsft,
    )


@pytest.fixture
def snr_object(mds):
    return SignalToNoiseRatio(
        mdss=mds.Series,
        assumeSqrtSX=1e-23,
    )


def test_signal_to_noise_ratio(signal_params, snr_object):
    snr2 = snr_object.compute_snr2(**signal_params)

    assert np.isfinite(snr2)
    assert snr2 > 0


def test_make_multi_noise_weights_from_asd():
    Tsft = 1_800
    tstart = 700_000_000
    ts = np.arange(tstart, tstart + 4 * Tsft, Tsft)
    mdss = MultiDetectorStates(
        timestamps={detector: ts for detector in ["H1", "L1"]},
        T_sft=Tsft,
    ).Series

    weights = make_multi_noise_weights_from_asd(
        mdss=mdss,
        sqrtSX_list=[1e-23, 1.2e-23],
    )

    assert weights.length == 2
    assert weights.Sinv_Tsft > 0
    assert weights.data[0].length == mdss.data[0].length
    assert weights.data[1].length == mdss.data[1].length
    assert weights.data[0].data[0] > weights.data[1].data[0]


@pytest.mark.parametrize(
    "sqrtSX_list,T_sft",
    [([], 1800), ([1e-23], 1800), ([1e-23, -1e-23], 1800)],
)
def test_make_multi_noise_weights_from_asd_validation(sqrtSX_list, T_sft):
    ts = np.arange(700_000_000, 700_000_000 + 4 * T_sft, T_sft)
    mdss = MultiDetectorStates(
        timestamps={detector: ts for detector in ["H1", "L1"]},
        T_sft=T_sft,
    ).Series

    with pytest.raises(ValueError):
        make_multi_noise_weights_from_asd(mdss=mdss, sqrtSX_list=sqrtSX_list)


def test_signal_to_noise_ratio_with_explicit_noise_weights(signal_params, mds):
    weights = make_multi_noise_weights_from_asd(
        mdss=mds.Series,
        sqrtSX_list=[1e-23, 1.2e-23],
    )

    snr_object = SignalToNoiseRatio(
        mdss=mds.Series,
        noise_weights=weights,
    )

    snr2 = snr_object.compute_snr2(**signal_params)
    assert np.isfinite(snr2)
    assert snr2 > 0


@pytest.mark.parametrize(
    "detectors,sqrtSX_list",
    [
        (["H1"], [1e-23]),
        (["H1", "L1"], [1e-23, 1.05e-23]),
        (["H1", "L1", "V1"], [1e-23, 1.05e-23, 1.1e-23]),
    ],
)
def test_multi_detector_network_snr2_matches_reweighted_single_detector_estimate(
    detectors, sqrtSX_list
):
    Tsft = 1_800
    tstart = 1_238_166_018
    timestamps = np.arange(tstart, tstart + 10 * 86400, Tsft)
    population = make_isotropic_population(100, np.random.default_rng(42))

    single_detector_rho2 = []
    for detector, sqrtSX in zip(detectors, sqrtSX_list):
        detector_mds = MultiDetectorStates(
            timestamps={detector: timestamps},
            T_sft=Tsft,
        )
        single_detector_rho2.append(
            compute_unit_depth_rho2(
                SignalToNoiseRatio(mdss=detector_mds.Series, assumeSqrtSX=sqrtSX),
                population,
                sqrtSX**2,
            )
        )

    combined_mds = MultiDetectorStates(
        timestamps={detector: timestamps for detector in detectors},
        T_sft=Tsft,
    )
    weights = make_multi_noise_weights_from_asd(
        mdss=combined_mds.Series,
        sqrtSX_list=sqrtSX_list,
    )
    combined_rho2 = compute_unit_depth_rho2(
        SignalToNoiseRatio(mdss=combined_mds.Series, noise_weights=weights),
        population,
        Tsft / weights.Sinv_Tsft,
    )

    harmonic_mean_sqrtSX = len(sqrtSX_list) / np.sum(1 / np.asarray(sqrtSX_list))
    reweighted_single_detector_estimate = (
        np.sum(
            np.asarray(
                [rho2 * sqrtSX for rho2, sqrtSX in zip(single_detector_rho2, sqrtSX_list)]
            ),
            axis=0,
        )
        / harmonic_mean_sqrtSX
    )
    ratio = reweighted_single_detector_estimate / combined_rho2

    assert np.all(np.isfinite(ratio))
    np.testing.assert_allclose(np.mean(ratio), 1.0, atol=0.05)
