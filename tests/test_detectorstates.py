import pytest
import numpy as np

from cows3.detectorstates import MultiDetectorStates


@pytest.fixture
def timestamps():
    return {
        "L1": 1238166018 + np.arange(0, 10, 3),
        "H1": 1238166018 + np.arange(0, 10, 2),
    }


@pytest.fixture
def wrong_timestamps():
    return {
        "AB": 1238166018 + np.arange(0, 10, 3),
    }


@pytest.fixture
def Tsft():
    return 1800


@pytest.fixture
def time_offset():
    return 0.0


def test_get_multi_detector_states(timestamps, Tsft, time_offset):
    mdss = MultiDetectorStates(
        timestamps=timestamps, T_sft=Tsft, t_offset=time_offset
    ).Series

    assert mdss.length == len(timestamps)
    for ind, ifo in enumerate(timestamps):
        assert mdss.data[ind].length == timestamps[ifo].size
        assert mdss.data[ind].detector.frDetector.prefix == ifo
        assert mdss.data[ind].deltaT == Tsft

        for gps_ind in range(mdss.data[ind].length):
            mdss.data[ind].data[gps_ind].tGPS.gpsSeconds == timestamps[ifo][gps_ind]


def test_wrong_timestamps(wrong_timestamps, Tsft, time_offset):
    with pytest.raises(RuntimeError) as e_info:
        mds = MultiDetectorStates(
            timestamps=wrong_timestamps, T_sft=Tsft, t_offset=time_offset
        )


def test_extract_detector_velocities(timestamps, Tsft, time_offset):
    mds = MultiDetectorStates(timestamps=timestamps, T_sft=Tsft, t_offset=time_offset)

    velocities = mds.velocities
    mdss = mds.Series

    assert len(velocities) == len(timestamps)
    assert all(key in velocities for key in timestamps)
    for ifo_ind in range(len(timestamps)):
        shape_to_test = velocities[mdss.data[ifo_ind].detector.frDetector.prefix].shape
        assert shape_to_test[0] == 3
        assert shape_to_test[1] == mdss.data[ifo_ind].length
