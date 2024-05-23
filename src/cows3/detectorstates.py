import logging

import lal
import lalpulsar
import numpy as np

logger = logging.getLogger(__name__)


def get_multi_detector_states(
    timestamps: Dict[str, np.array], Tsft: int | float, time_offset: int | float
) -> lalpulsar.MultiDetectorStates:
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
        if time_offset is None:
            time_offset = 0.5 * Tsft

        self._parse_timestamps_and_detectors(timestamps, Tsft, detectors)
        return lalpulsar.GetMultiDetectorStates(
            self.multi_timestamps,
            self.multi_detector,
            self.ephems, # FIXME
            tOffset=time_offset or 0.5 * Tsft,

def  

# class MultiDetectorStates:
#    """
#    Python interface to XLALGetMultiDetectorStates and XLALGetMultiDetectorStatesFromMultiSFTs.
#    """
#
#    def __init__(self):
#        self.ephems = lalpulsar.InitBarycenter(*get_ephemeris_files())
#
#    def get_multi_detector_states(
#        self, timestamps, Tsft, detectors=None, time_offset=None
#    ):
#        """
#        Parameters
#        ----------
#        timestamps: array-like or dict
#            GPS timestamps at which detector states will be retrieved.
#            If array, use the same set of timestamps for all detectors,
#            which must be explicitly given by the user via `detectors`.
#            If dictionary, each key should correspond to a valid detector name
#            to be parsed by XLALParseMultiLALDetector and the associated value
#            should be an array-like set of GPS timestamps for each individual detector.
#        Tsft: float
#            Timespan covered by each timestamp. It does not need to coincide with the
#            separation between consecutive timestamps.
#        detectors: list[str] or comma-separated string
#            List of detectors to be parsed using XLALParseMultiLALDetector.
#            Conflicts with dictionary of `timestamps`, required otherwise.
#        time_offset: float
#            Timestamp offset to retrieve detector states.
#            Defaults to LALSuite's default of using the central time of an STF (SFT's timestamp + Tsft/2).
#
#        Returns
#        -------
#        multi_detector_states: lalpulsar.MultiDetectorStateSeries
#            Resulting multi-detector states produced by XLALGetMultiDetectorStates
#        """
#        if time_offset is None:
#            time_offset = 0.5 * Tsft
#
#        self._parse_timestamps_and_detectors(timestamps, Tsft, detectors)
#        return lalpulsar.GetMultiDetectorStates(
#            self.multi_timestamps,
#            self.multi_detector,
#            self.ephems,
#            time_offset,
#        )
#
#    def get_multi_detector_states_from_sfts(
#        self,
#        sftfilepath,
#        central_frequency,
#        time_offset=None,
#        frequency_wing_bins=1,
#        sft_constraint=None,
#        return_sfts=False,
#    ):
#        """
#        Parameters
#        ----------
#        sftfilepath: str
#            Path to SFT files in a format compatible with XLALSFTdataFind.
#        central_frequency: float
#            Frequency [Hz] around which SFT data will be retrieved.
#            This option is only relevant if further information is to be
#            retrieved from the SFTs (i.e. `return_sfts=True`).
#        time_offset: float
#            Timestamp offset to retrieve detector states.
#            Defaults to LALSuite's default of using the central time of an STF (SFT's timestamp + Tsft/2).
#        frequency_wing_bins: int
#            Frequency bins around the central frequency to retrieve from
#            SFT data. Bin size is determined using the SFT baseline time
#            as obtained from the catalog.
#            This option is only relevant if further information is to be
#            retrieved from the SFTs (i.e. `return_sfts=True`).
#        sft_constraint: lalpulsar.SFTConstraint
#            Optional argument to specify further constraints in XLALSFTdataFind.
#        return_sfts: bool
#            If True, also return the loaded SFTs. This is useful to compute further
#            quantities such as noise weights.
#
#        Returns
#        -------
#        multi_detector_states: lalpulsar.MultiDetectorStateSeries
#            Resulting multi-detector states produced by XLALGetMultiDetectorStatesFromMultiSFTs
#        multi_sfts: lalpulsar.MultiSFTVector
#            Only if `return_sfts` is True.
#            MultiSFTVector produced by XLALLoadMultiSFTs along the specified frequency band.
#        """
#        # FIXME: Use MultiCatalogView once lalsuite implements the proper
#        # SWIG wrapper around XLALLoadMultiSFTsFromView.
#        sft_catalog = lalpulsar.SFTdataFind(sftfilepath, sft_constraint)
#        df = sft_catalog.data[0].header.deltaF
#        wing_Hz = df * frequency_wing_bins
#        multi_sfts = lalpulsar.LoadMultiSFTs(
#            sft_catalog,
#            fMin=central_frequency - wing_Hz,
#            fMax=central_frequency + wing_Hz,
#        )
#
#        if time_offset is None:
#            time_offset = 0.5 / df
#
#        multi_detector_states = lalpulsar.GetMultiDetectorStatesFromMultiSFTs(
#            multiSFTs=multi_sfts, edat=self.ephems, tOffset=time_offset
#        )
#        if return_sfts:
#            return multi_detector_states, multi_sfts
#        else:
#            return multi_detector_states
#
#    def _parse_timestamps_and_detectors(self, timestamps, Tsft, detectors):
#        """
#        Checks consistency between timestamps and detectors.
#
#        If `timestamps` is a dictionary, gets detector names from the keys
#        and makes sure `detectors` is None.
#
#        Otherwise, formats `detectors` into a list and makes sure `timestamps`
#        is a 1D array containing numbers.
#        """
#
#        if isinstance(timestamps, dict):
#            if detectors is not None:
#                raise ValueError("`timestamps`' keys are redundant with `detectors`.")
#            for ifo in timestamps:
#                try:
#                    lalpulsar.FindCWDetector(name=ifo, exactMatch=True)
#                except Exception:
#                    raise ValueError(
#                        f"Invalid detector name {ifo} in timestamps. "
#                        "Each key should contain a single detector, "
#                        "no comma-separated strings allowed."
#                    )
#
#            logger.debug("Retrieving detectors from timestamps dictionary.")
#            detectors = list(timestamps.keys())
#            timestamps = (np.array(ts) for ts in timestamps.values())
#
#        elif detectors is not None:
#            if isinstance(detectors, str):
#                logger.debug("Converting `detectors` string to list")
#                detectors = detectors.replace(" ", "").split(",")
#
#            logger.debug("Checking integrity of `timestamps`")
#            ts = np.array(timestamps)
#            if ts.dtype == np.dtype("O") or ts.ndim > 1:
#                raise ValueError("`timestamps` is not a 1D list of numerical values")
#            timestamps = (ts for ifo in detectors)
#
#        self.multi_detector = lalpulsar.MultiLALDetector()
#        lalpulsar.ParseMultiLALDetector(self.multi_detector, detectors)
#
#        self.multi_timestamps = lalpulsar.CreateMultiLIGOTimeGPSVector(
#            self.multi_detector.length
#        )
#        for ind, ts in enumerate(timestamps):
#            self.multi_timestamps.data[ind] = self._numpy_array_to_LIGOTimeGPSVector(
#                ts, Tsft
#            )
#
#    @staticmethod
#    def _numpy_array_to_LIGOTimeGPSVector(numpy_array, Tsft=None):
#        """
#        Maps a numpy array of floats into a LIGOTimeGPS array using `np.floor`
#        to separate seconds and nanoseconds.
#        """
#
#        if numpy_array.ndim != 1:
#            raise ValueError(
#                f"Time stamps array must be 1D: Current one has {numpy_array.ndim}."
#            )
#
#        seconds_array = np.floor(numpy_array)
#        nanoseconds_array = np.floor(1e9 * (numpy_array - seconds_array))
#
#        time_gps_vector = lalpulsar.CreateTimestampVector(numpy_array.shape[0])
#        for ind in range(time_gps_vector.length):
#            time_gps_vector.data[ind] = lal.LIGOTimeGPS(
#                int(seconds_array[ind]), int(nanoseconds_array[ind])
#            )
#            time_gps_vector.deltaT = Tsft or 0
#
#        return time_gps_vector
# """
