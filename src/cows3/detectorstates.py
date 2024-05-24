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
