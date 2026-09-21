import importlib.util
import pytest
import numpy as np
import lalpulsar

from cows3.detectorstates import CustomIFO, MultiDetectorStates


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


def test_customifo_x0_matches_h1_velocities(Tsft, time_offset):
    source = lalpulsar.GetSiteInfo("H1")

    CustomIFO(
        name="X0_H1_clone",
        prefix="X0",
        vertexLatitudeRadians=source.frDetector.vertexLatitudeRadians,
        vertexLongitudeRadians=source.frDetector.vertexLongitudeRadians,
        vertexElevation=source.frDetector.vertexElevation,
        xArmAzimuthRadians=source.frDetector.xArmAzimuthRadians,
        yArmAzimuthRadians=source.frDetector.yArmAzimuthRadians,
        xArmAltitudeRadians=source.frDetector.xArmAltitudeRadians,
        yArmAltitudeRadians=source.frDetector.yArmAltitudeRadians,
        LALDetectorType=source.type,
    )

    ts = 1238166018 + np.arange(0, 10, 2)
    mds = MultiDetectorStates(
        timestamps={"H1": ts, "X0": ts},
        T_sft=Tsft,
        t_offset=time_offset,
    )

    assert "H1" in mds.velocities
    assert "X0" in mds.velocities
    assert mds.velocities["X0"] == pytest.approx(mds.velocities["H1"])


@pytest.fixture
def imported_customifo_module(tmp_path):
    module_path = tmp_path / "custom_ifo_module.py"
    module_path.write_text(
        "\n".join(
            [
                "import lalpulsar",
                "from cows3.detectorstates import CustomIFO",
                "",
                'source = lalpulsar.GetSiteInfo("H1")',
                "IMPORTED_IFO = CustomIFO(",
                '    name="X1_H1_clone",',
                '    prefix="X1",',
                "    vertexLatitudeRadians=source.frDetector.vertexLatitudeRadians,",
                "    vertexLongitudeRadians=source.frDetector.vertexLongitudeRadians,",
                "    vertexElevation=source.frDetector.vertexElevation,",
                "    xArmAzimuthRadians=source.frDetector.xArmAzimuthRadians,",
                "    yArmAzimuthRadians=source.frDetector.yArmAzimuthRadians,",
                "    xArmAltitudeRadians=source.frDetector.xArmAltitudeRadians,",
                "    yArmAltitudeRadians=source.frDetector.yArmAltitudeRadians,",
                "    LALDetectorType=source.type,",
                ")",
            ]
        )
    )
    return module_path


def test_imported_customifo_is_available_to_detectorstates(
    imported_customifo_module, Tsft, time_offset, monkeypatch
):
    spec = importlib.util.spec_from_file_location(
        "tests.custom_ifo_module", imported_customifo_module
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(__import__("sys").modules, spec.name, module)
    spec.loader.exec_module(module)

    ts = 1238166018 + np.arange(0, 10, 2)
    mds = MultiDetectorStates(
        timestamps={"X1": ts},
        T_sft=Tsft,
        t_offset=time_offset,
    )

    assert module.IMPORTED_IFO.prefix == "X1"
    assert mds.Series.data[0].detector.frDetector.prefix == "X1"
