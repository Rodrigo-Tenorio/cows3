import logging
import re
from dataclasses import dataclass

import lal
import lalpulsar
import numpy as np

from .ephemeris import DEFAULT_EPHEMERIS

logger = logging.getLogger(__name__)


@dataclass
class CustomIFO:
    """Register a custom interferometer in LALSuite when instantiated.

        Parameters map directly to ``lal.FrDetector`` fields:

        - ``name``: detector name string.
        - ``prefix``: 2-character detector prefix for CW naming.
        - ``latitude_rad`` / ``longitude_rad``: geodetic coordinates in radians.
        - ``elevation_m``: height above reference ellipsoid in meters.
        - ``xarm_azimuth_rad`` / ``yarm_azimuth_rad``: arm azimuths in radians,
            clockwise from North.
        - ``xarm_alt_rad`` / ``yarm_alt_rad``: arm altitude angles in radians,
            measured upward from local tangent plane.
        - ``xarm_midpoint_m`` / ``yarm_midpoint_m``: distance from vertex to arm
            midpoint in meters (for a 10 km arm, use 5000 m).
        - ``detector_type``: one of LAL's ``LALDETECTORTYPE_*`` constants.

    Notes
    -----
    For CW codes, the detector prefix is validated by LALPulsar's
    special-detector registry and must follow the pattern [XYZ][0-9]
    (for example X0, X2, Y1, Z9).
    """

    name: str
    prefix: str
    latitude_rad: float
    longitude_rad: float
    elevation_m: float
    xarm_azimuth_rad: float
    yarm_azimuth_rad: float
    xarm_alt_rad: float
    yarm_alt_rad: float
    xarm_midpoint_m: float = 0.0
    yarm_midpoint_m: float = 0.0
    detector_type: int = lal.LALDETECTORTYPE_IFODIFF

    def __post_init__(self):
        if not re.fullmatch(r"[XYZ][0-9]", self.prefix):
            raise ValueError(
                "CustomIFO.prefix must match [XYZ][0-9] for special CW detectors."
            )

        fr_detector = lal.FrDetector()
        fr_detector.name = self.name
        fr_detector.prefix = self.prefix
        fr_detector.vertexLatitudeRadians = self.latitude_rad
        fr_detector.vertexLongitudeRadians = self.longitude_rad
        fr_detector.vertexElevation = self.elevation_m
        fr_detector.xArmAzimuthRadians = self.xarm_azimuth_rad
        fr_detector.yArmAzimuthRadians = self.yarm_azimuth_rad
        fr_detector.xArmAltitudeRadians = self.xarm_alt_rad
        fr_detector.yArmAltitudeRadians = self.yarm_alt_rad
        fr_detector.xArmMidpoint = self.xarm_midpoint_m
        fr_detector.yArmMidpoint = self.yarm_midpoint_m

        # Geometry/consistency checks beyond naming and enum selection are delegated to LAL.
        detector = lal.CreateDetector(None, fr_detector, self.detector_type)
        lalpulsar.RegisterSpecialCWDetector(detector)


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

    Notes
    -----
    Custom detectors should be registered in advance by instantiating
    ``CustomIFO`` with the desired detector definition.
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