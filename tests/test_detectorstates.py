import pytest
import numpy as np

from cows3.detectorstates import MultiDetectorStates, extract_detector_velocities


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
    mds = MultiDetectorStates(timestamps=timestamps, Tsft=Tsft)(time_offset=time_offset)

    assert mds.length == len(timestamps)
    for ind, ifo in enumerate(timestamps):
        assert mds.data[ind].length == timestamps[ifo].size
        assert mds.data[ind].detector.frDetector.prefix == ifo
        assert mds.data[ind].deltaT == Tsft

        for gps_ind in range(mds.data[ind].length):
            mds.data[ind].data[gps_ind].tGPS.gpsSeconds == timestamps[ifo][gps_ind]


def test_wrong_timestamps(wrong_timestamps, Tsft, time_offset):
    with pytest.raises(RuntimeError) as e_info:
        mds = MultiDetectorStates(timestamps=wrong_timestamps, Tsft=Tsft)(
            time_offset=time_offset
        )


def test_extract_detector_velocities(timestamps, Tsft, time_offset):
    mds = MultiDetectorStates(timestamps=timestamps, Tsft=Tsft)(time_offset=time_offset)
    velocities = extract_detector_velocities(mds)

    assert len(velocities) == len(timestamps)
    assert all(key in velocities for key in timestamps)
    for ifo_ind in range(len(timestamps)):
        shape_to_test = velocities[mds.data[ifo_ind].detector.frDetector.prefix].shape
        assert shape_to_test[0] == 3
        assert shape_to_test[1] == mds.data[ifo_ind].length
