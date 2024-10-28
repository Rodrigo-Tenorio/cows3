import numpy as np
import pytest

from cows3.detectorstates import MultiDetectorStates
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
