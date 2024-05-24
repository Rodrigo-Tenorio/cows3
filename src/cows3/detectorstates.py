import logging

import lal
import lalpulsar
import numpy as np

from solar_system_ephemerides import body_ephemeris_path

logger = logging.getLogger(__name__)

DEFAULT_EPHEMERIS = lalpulsar.InitBarycenter(
    str(body_ephemeris_path("earth")),
    str(body_ephemeris_path("sun")),
)


def get_multi_detector_states(
    timestamps: dict[str, np.array],
    Tsft: float,
    time_offset: float | None,
    ephemeris: lalpulsar.EphemerisData = DEFAULT_EPHEMERIS,
) -> lalpulsar.MultiDetectorStateSeries:
    """
    Python interface to `XLALGetMultiDetectorStates` and
    `XLALGetMultiDetectorStatesFromMultiSFTs`.

    Parameters
    ----------
    timestamps:
        Dictionary containing the GPS timestamps at which detector
        states will be retrieved.
        Keys MUST be two-character detector names as described in LALSuite;
        values MUST be numpy arrays containing the timestamps.
        E.g. for an observing run from GPS 1 to GPS 5 using LIGO Hanford
        and LIGO Livingston:
        ```
        timestamps = {
            "H1": np.array([1, 2, 3, 4, 5]),
            "L1": np.array([1, 2, 3, 4, 5])
        }
        ```
    Tsft:
        Timestamp covered for each timestamp. It does not need to coincide
        with the separation between consecutive timestamps.
    time_offset:
        Time offset with respect to the timestamp at which the detector
        state will be retrieved. Defaults to LALSuite's behaviouur.
    """

    multi_detector = lalpulsar.MultiLALDetector()
    lalpulsar.ParseMultiLALDetector(multi_detector, [*timestamps])

    multi_timestamps = lalpulsar.CreateMultiLIGOTimeGPSVector(multi_detector.length)
    for ind, ifo in enumerate(timestamps):
        seconds_array = np.floor(timestamps[ifo])
        nanoseconds_array = np.floor(1e9 * (timestamps[ifo] - seconds_array))

        multi_timestamps.data[ind] = lalpulsar.CreateTimestampVector(
            seconds_array.shape[0]
        )

        multi_timestamps.data[ind].deltaT = Tsft

        for ts_ind in range(multi_timestamps.data[ind].length):
            multi_timestamps.data[ind].data[ts_ind] = lal.LIGOTimeGPS(
                int(seconds_array[ts_ind]), int(nanoseconds_array[ts_ind])
            )

    return lalpulsar.GetMultiDetectorStates(
        multiTS=multi_timestamps,
        multiIFO=multi_detector,
        edat=ephemeris,
        tOffset=time_offset or 0.5 * Tsft,
    )