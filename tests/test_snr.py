import numpy as np
import pytest

from cows3.detectorstates import MultiDetectorStates
from cows3.noise import make_multi_noise_weights_from_asd
from cows3.snr import SignalToNoiseRatio


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


def test_SignalToNoiseRatio(signal_params, snr_object):
    params = {
        "aPlus": 0.5 * 1e-23,
        "aCross": 1e-23,
        "psi": 0,
        "phi0": 0,
        "Alpha": 0,
        "Delta": 0,
    }

    snr_object.compute_snr2(**signal_params)


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


def test_SignalToNoiseRatio_with_explicit_noise_weights(signal_params, mds):
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
