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


def extract_detector_velocities(
    multi_detector_states_series: lalpulsar.MultiDetectorStateSeries,
) -> dict[str, np.ndarray]:
    """
    Extracts detector velocity vectors from a MultiDetetorStateSeries
    into numpy arrays.

    Parameters
    ----------
    multi_detector_states_series:
        Self-explanatory

    Returns
    -------
    velocities:
        Dictionary. Keys refer to detector's 2-character prefix,
        values are (3, num_timestamps) numpy arrays.
    """
    velocities = {}

    for ifo_ind in range(multi_detector_states_series.length):
        ifo_name = multi_detector_states_series.data[ifo_ind].detector.frDetector.prefix
        velocities[ifo_name] = np.vstack(
            [data.vDetector for data in multi_detector_states_series.data[ifo_ind].data]
        ).T

    return velocities


class MultiDetectorStates:
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
        Time period covered for each timestamp. Does not need to coincide
        with the separation between consecutive timestamps.
    time_offset:
        Time offset with respect to the timestamp at which the detector
        state will be retrieved. Defaults to LALSuite's behaviour.
    ephemeris:
        Default uses `solar_system_ephemerides` to get lalsuite's default.
    """

    def __init__(
        self,
        timestamps: dict[str, np.array],
        Tsft: int | float,
        ephemeris: lalpulsar.EphemerisData = DEFAULT_EPHEMERIS,
    ):
        self.timestamps = timestamps
        self.Tsft = Tsft
        self.ephemeris = ephemeris

    @property
    def timestamps(self) -> dict:
        return self._timestamps

    @property
    def Tsft(self) -> int | float:
        return self._Tsft

    @property
    def multi_lal_detector(self) -> lalpulsar.MultiLALDetector:
        return self._multi_lal_detector

    @property
    def multi_timestamps(self) -> lalpulsar.MultiLIGOTimeGPSVector:
        return self._multi_timestamps

    @timestamps.setter
    def timestamps(self, new_timestamps: dict):

        self._timestamps = new_timestamps

        self._multi_lal_detector = lalpulsar.MultiLALDetector()
        lalpulsar.ParseMultiLALDetector(self._multi_lal_detector, [*self._timestamps])

        self._multi_timestamps = lalpulsar.CreateMultiLIGOTimeGPSVector(
            self._multi_lal_detector.length
        )
        for ind, ifo in enumerate(new_timestamps):
            seconds_array = np.floor(new_timestamps[ifo])
            nanoseconds_array = np.floor(1e9 * (new_timestamps[ifo] - seconds_array))

            self._multi_timestamps.data[ind] = lalpulsar.CreateTimestampVector(
                seconds_array.shape[0]
            )

            for ts_ind in range(self._multi_timestamps.data[ind].length):
                self._multi_timestamps.data[ind].data[ts_ind] = lal.LIGOTimeGPS(
                    int(seconds_array[ts_ind]), int(nanoseconds_array[ts_ind])
                )

    @Tsft.setter
    def Tsft(self, new_Tsft: int | float):
        self._Tsft = new_Tsft
        for ifo_ind in range(self.multi_timestamps.length):
            self._multi_timestamps.data[ifo_ind].deltaT = self._Tsft

    def __call__(
        self, time_offset: float | None = None
    ) -> lalpulsar.MultiDetectorStateSeries:
        return lalpulsar.GetMultiDetectorStates(
            multiTS=self.multi_timestamps,
            multiIFO=self.multi_lal_detector,
            edat=self.ephemeris,
            tOffset=time_offset or 0.5 * self.Tsft,
        )
