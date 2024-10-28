import logging

import lal
import lalpulsar
import numpy as np

from .ephemeris import DEFAULT_EPHEMERIS

logger = logging.getLogger(__name__)


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
    T_sft:
        Time period covered for each timestamp. Does not need to coincide
        with the separation between consecutive timestamps. It will be floored
        using `int`.
    t_offset:
        Time offset with respect to the timestamp at which the detector
        state will be retrieved. Defaults to LALSuite's behaviour.
    ephemeris:
        Default uses `solar_system_ephemerides` to get lalsuite's default.
    """

    def __init__(
        self,
        timestamps: dict[str, np.array],
        T_sft: int,
        t_offset: int | None = None,
        ephemeris: lalpulsar.EphemerisData = DEFAULT_EPHEMERIS,
    ):
        self.timestamps = timestamps
        self.T_sft = T_sft
        self.t_offset = t_offset
        self.ephemeris = ephemeris

    @property
    def Series(self) -> lalpulsar.MultiDetectorStateSeries:
        """
        Return lalpulsar.MultiDetectorStateSeries constructed using the instance's attributes.
        """
        return lalpulsar.GetMultiDetectorStates(
            multiTS=self.multi_timestamps,
            multiIFO=self.multi_lal_detector,
            edat=self.ephemeris,
            tOffset=self.t_offset,
        )

    @property
    def velocities(self) -> dict[str, np.ndarray]:
        """
        Extracts detector velocity vectors into numpy arrays.

        Returns
        -------
        velocities:
            Dictionary. Keys refer to detector's 2-character prefix,
            values are (3, num_timestamps) numpy arrays.
        """

        mdss = self.Series

        velocities = {}

        for ifo_ind in range(mdss.length):
            ifo_name = mdss.data[ifo_ind].detector.frDetector.prefix
            velocities[ifo_name] = np.vstack(
                [data.vDetector for data in mdss.data[ifo_ind].data]
            ).T

        return velocities

    @property
    def timestamps(self) -> dict[str, np.ndarray]:
        return self._timestamps

    @property
    def T_sft(self) -> int:
        return self._T_sft

    @property
    def t_offset(self) -> int | float:
        return self._t_offset

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

    @T_sft.setter
    def T_sft(self, new_T_sft: int):
        self._T_sft = int(new_T_sft)
        for ifo_ind in range(self.multi_timestamps.length):
            self._multi_timestamps.data[ifo_ind].deltaT = self._T_sft

    @t_offset.setter
    def t_offset(self, new_t_offset: int | None):
        self._t_offset = new_t_offset if new_t_offset is not None else 0.5 * self.T_sft
