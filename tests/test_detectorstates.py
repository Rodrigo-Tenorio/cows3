import pytest
import numpy as np

from cows3.detectorstates import get_multi_detector_states


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
    mds = get_multi_detector_states(
        timestamps=timestamps, Tsft=Tsft, time_offset=time_offset
    )

    assert mds.length == len(timestamps)
    for ind, ifo in enumerate(timestamps):
        assert mds.data[ind].length == timestamps[ifo].size
        assert mds.data[ind].detector.frDetector.prefix == ifo
        assert mds.data[ind].deltaT == Tsft

        for gps_ind in range(mds.data[ind].length):
            mds.data[ind].data[gps_ind].tGPS.gpsSeconds == timestamps[ifo][gps_ind]


def test_wrong_timestamps(wrong_timestamps, Tsft, time_offset):
    with pytest.raises(RuntimeError) as e_info:
        mds = get_multi_detector_states(
            timestamps=wrong_timestamps, Tsft=Tsft, time_offset=time_offset
        )
