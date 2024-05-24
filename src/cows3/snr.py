from functools import cache
import logging

import lalpulsar
import numpy as np

logger = logging.getLogger(__name__)

def compute_Mmunu(
    Alpha: float,
    Delta: float,
    multi_detector_states: lalpulsar.MultiDetectorStateSeries,
    multi_noise_weights: lalpulsar.MultiNoiseWeights,
):
    """
    Compute Mmunu matrix at a specific sky position using the detector states
    (and possible noise weights) given at initialization time.
    If no noise weigths were given, unit weights are assumed.

    Parameters
    ----------
   Alpha: float
       Right ascension (equatorial longitude) of the signal in radians.
    Delta: float
       Declination (equatorial latitude) of the signal in radians.

    Returns
    -------
   Mmunu: lalpulsar.AntennaPatternMatrix
        Mmunu matrix encoding the response of the given detector network
        to a CW at the specified sky position.
   """
    sky = lal.SkyPosition()
    sky.longitude = Alpha
    sky.latitude = Delta
    sky.system = lal.COORDINATESYSTEM_EQUATORIAL
    lal.NormalizeSkyPosition(sky.longitude, sky.latitude)

    Mmunu = lalpulsar.ComputeMultiAMCoeffs(
        multiDetStates=multi_detector_states,
        multiWeights=multi_noise_weights,
        skypos=sky,
    ).Mmunu

    if self.noise_weights is None:
        Mmunu.Sinv_Tsft = self.Sinv_Tsft

    return Mmunu


# class SignalToNoiseRatio:
#    r"""Compute the optimal SNR of a CW signal as expected in Gaussian noise.
#
#    The definition of SNR (shortcut for "optimal signal-to-noise ratio")
#    is taken from Eq. (76) of https://dcc.ligo.org/T0900149-v6/public and is
#    such that :math:`\langle 2\mathcal{F}\rangle = 4 + \textrm{SNR}^2`,
#    where  :math:`\langle 2\mathcal{F}\rangle` represents the expected
#    value over noise realizations of twice the F-statistic of a template
#    perfectly matched to an existing signal in the data.
#
#    Computing :math:`\textrm{SNR}^2` requires two quantities:
#
#    * | The antenna pattern matrix :math:`\mathcal{M}`, which depends on the sky position :math:`\vec{n}`
#      | and polarization angle :math:`\psi` and encodes the effect of the detector's antenna pattern response
#      | over the course of the observing run.
#    * | The JKS amplitude parameters :math:`(\mathcal{A}^0, \mathcal{A}^1, \mathcal{A}^2, \mathcal{A}^3)`
#      | [JKS1998]_ which are functions of the CW's amplitude parameters :math:`(h_0,\cos\iota, \psi, \phi_0)` or,
#      | alternatively, :math:`(A_{+}, A_{\times}, \psi, \phi_0)`.
#
#    .. [JKS1998] `Jaranowski, Krolak, Schuz Phys. Rev. D58 063001, 1998 <https://arxiv.org/abs/gr-qc/9804014>`_
#
#    Parameters
#    ----------
#    detector_states: lalpulsar.MultiDetectorStateSeries
#        MultiDetectorStateSeries as produced by DetectorStates.
#        Provides the required information to compute the antenna pattern contribution.
#    noise_weights: Union[lalpulsar.MultiNoiseWeights, None]
#        Optional, incompatible with `assumeSqrtSX`.
#        Can be computed from SFTs using `SignalToNoiseRatio.from_sfts`.
#        Noise weights to account for a varying noise floor or unequal noise
#        floors in different detectors.
#    assumeSqrtSX: float
#        Optional, incompatible with `noise_weights`.
#        Single-sided amplitude spectral density (ASD) of the detector noise.
#        This value is used for all detectors, meaning it's not currently possible to manually
#        specify different noise floors without creating SFT files.
#        (To be improved in the future; developer note:
#        will require SWIG constructor for MultiNoiseWeights.)
#    """
#
#    detector_states: lalpulsar.MultiDetectorStateSeries = field()
#    noise_weights: Union[lalpulsar.MultiNoiseWeights, None] = field(default=None)
#    assumeSqrtSX: float = field(default=None)
#
#    def __attrs_post_init__(self):
#        have_noise_weights = self.noise_weights is not None
#        have_sqrtSX = self.assumeSqrtSX is not None
#
#        if have_noise_weights == have_sqrtSX:
#            raise ValueError(
#                "Need either `assumeSqrtSX` or `noise_weights` to account for background noise."
#            )
#
#        self.Tsft = self.detector_states.data[0].deltaT
#        if have_sqrtSX:
#            self.Sinv_Tsft = self.Tsft / self.assumeSqrtSX**2
#        else:
#            self.Sinv_Tsft = None
#
#    @classmethod
#    def from_sfts(
#        cls,
#        F0,
#        sftfilepath,
#        time_offset=None,
#        running_median_window=lalpulsar.FstatOptionalArgsDefaults.runningMedianWindow,
#        sft_constraint=None,
#    ):
#        """
#        Alternative constructor to retrieve detector states and noise weights from SFT files.
#        This method is based on
#        :py:meth:`DetectorStates.multi_detector_states_from_sfts`.
#        This is currently the other way in which varying / different noise floors can be used
#        when computing SNRs.
#
#        Parameters
#        ----------
#
#        F0: float
#            Central frequency [Hz] to retrieve from the SFT files to compute noise weights.
#        sftfilepath: str
#            Path to SFT files in a format compatible with XLALSFTdataFind.
#        time_offset: float
#            Timestamp offset to retrieve detector states.
#            Defaults to LALSuite's default of using the central time of an STF (SFT's timestamp + Tsft/2).
#        running_median_window: int
#            Window used to compute the running-median noise floor estimation.
#            Default value is consistent with that used in PredictFstat executable.
#        sft_constraint: lalpulsar.SFTConstraint
#            Optional argument to specify further constraints in XLALSFTdataFind.
#        """
#
#        (
#            detector_states,
#            multi_sfts,
#        ) = DetectorStates().get_multi_detector_states_from_sfts(
#            sftfilepath=sftfilepath,
#            central_frequency=F0,
#            frequency_wing_bins=running_median_window // 2
#            + 10,  # PredictFstat.c:Line 414
#            time_offset=time_offset,
#            sft_constraint=sft_constraint,
#            return_sfts=True,
#        )
#        multi_rng_med = lalpulsar.NormalizeMultiSFTVect(
#            multi_sfts, running_median_window, None
#        )
#        noise_weights = lalpulsar.ComputeMultiNoiseWeights(
#            multi_rng_med, running_median_window, 0
#        )
#
#        return cls(detector_states=detector_states, noise_weights=noise_weights)
#
#    def compute_snr2(
#        self, Alpha, Delta, psi, phi, h0=None, cosi=None, aPlus=None, aCross=None
#    ):
#        r"""
#        Compute the :math:`\textrm{SNR}^2` of a CW signal using XLALComputeOptimalSNR2FromMmunu.
#        Parameters correspond to the standard ones used to describe a CW
#        (see e.g. Eqs. (16), (26), (30) of https://dcc.ligo.org/T0900149-v6/public ).
#
#        Mind that this function returns *squared* SNR
#        (Eq. (76) of https://dcc.ligo.org/T0900149-v6/public ),
#        which can be directly related to the expected F-statistic as
#        :math:`\langle 2\mathcal{F}\rangle = 4 + \textrm{SNR}^2`.
#
#        Parameters
#        ----------
#        Alpha: float
#            Right ascension (equatorial longitude) of the signal in radians.
#        Delta: float
#            Declination (equatorial latitude) of the signal in radians.
#        psi: float
#            Polarization angle.
#        h0: float
#            Nominal GW amplitude. Must be given together with `cosi`
#            and conflicts with `aPlus` and `aCross`.
#        cosi: float
#            Cosine of the source inclination w.r.t. line of sight.
#            Must be given together with `h0`
#            and conflicts with `aPlus` and `aCross`.
#        aPlus: float
#            Plus polarization amplitude. Must be given with `aCross`
#            and conflicts with `h0` and `cosi`.
#        aCross: float
#            Cross polarization amplitude. Must be given with `aPlus`
#            and conflicts with `h0` and `cosi`.
#
#        Returns
#        -------
#        SNR^2: float
#            Squared signal-to-noise ratio of a CW signal consistent
#            with the specified parameters in the specified detector
#            network.
#        """
#        aPlus, aCross = self._convert_amplitude_parameters(
#            h0=h0, cosi=cosi, aPlus=aPlus, aCross=aCross
#        )
#
#        Aphys = lalpulsar.PulsarAmplitudeParams()
#        Aphys.psi = psi
#        Aphys.phi0 = phi
#        Aphys.aPlus = aPlus
#        Aphys.aCross = aCross
#
#        M = self.compute_Mmunu(Alpha=Alpha, Delta=Delta)
#
#        return lalpulsar.ComputeOptimalSNR2FromMmunu(Aphys, M)
#
#    def compute_h0_from_snr2(
#        self,
#        Alpha,
#        Delta,
#        psi,
#        phi,
#        cosi,
#        snr2,
#    ):
#        r"""
#        Convert the :math:`\textrm{SNR}^2` of a CW signal to a corresponding amplitude
#        :math:`h_0` given the source orientation.
#        Parameters correspond to the standard ones used to describe a CW
#        (see e.g. Eqs. (16), (26), (30) of https://dcc.ligo.org/T0900149-v6/public ).
#
#        This function returns "inverts" Eq. (77) of
#        https://dcc.ligo.org/T0900149-v6/public by computing the overall prefactor
#        on :math:`h_0` using `self.compute_snr2(h0=1, ...)`.
#
#        Parameters
#        ----------
#        Alpha: float
#            Right ascension (equatorial longitude) of the signal in radians.
#        Delta: float
#            Declination (equatorial latitude) of the signal in radians.
#        psi: float
#            Polarization angle.
#        cosi: float
#            Cosine of the source inclination w.r.t. line of sight.
#            Must be given together with `h0`
#            and conflicts with `aPlus` and `aCross`.
#        snr2: float
#            Squared signal-to-noise ratio of a CW signal
#            in the specified detector network.
#
#        Returns
#        -------
#        h0: float
#            Nominal GW amplitude.
#        """
#        conversion_factor = self.compute_snr2(
#            Alpha=Alpha, Delta=Delta, psi=psi, phi=phi, cosi=cosi, h0=1.0
#        )
#        return np.sqrt(snr2 / conversion_factor)
#
#    def compute_twoF(self, *args, **kwargs):
#        r"""
#        Compute the expected :math:`2\mathcal{F}` value of a CW signal from the result of `compute_snr2`.
#
#        .. math:: \langle 2\mathcal{F}\rangle = 4 + \textrm{SNR}^2
#        .. math:: \sigma_{2\mathcal{F}} =  \sqrt{8 + 4 \textrm{SNR}^2}
#
#        Input parameters are passed untouched to `self.compute_snr2`.
#        See corresponding docstring for a list of valid parameters.
#
#        Returns
#        -------
#        expected_2F:
#            Expected value of a non-central chi-squared distribution with
#            four degrees of freedom and non-centrality parameter given by SNR^2.
#        stdev_2F:
#            Standard deviation of a non-central chi-squared distribution with
#            four degrees of freedom and non-centrality parameter given by SNR^2.
#        """
#        snr2 = self.compute_snr2(*args, **kwargs)
#        expected_2F = snr2 + 4.0
#        stdev_2F = np.sqrt(8.0 + 4.0 * snr2)
#        return expected_2F, stdev_2F
#
#    def compute_Mmunu(self, Alpha, Delta):
#        """
#        Compute Mmunu matrix at a specific sky position using the detector states
#        (and possible noise weights) given at initialization time.
#        If no noise weigths were given, unit weights are assumed.
#
#        Parameters
#        ----------
#        Alpha: float
#            Right ascension (equatorial longitude) of the signal in radians.
#        Delta: float
#            Declination (equatorial latitude) of the signal in radians.
#
#        Returns
#        -------
#        Mmunu: lalpulsar.AntennaPatternMatrix
#            Mmunu matrix encoding the response of the given detector network
#            to a CW at the specified sky position.
#        """
#
#        sky = lal.SkyPosition()
#        sky.longitude = Alpha
#        sky.latitude = Delta
#        sky.system = lal.COORDINATESYSTEM_EQUATORIAL
#        lal.NormalizeSkyPosition(sky.longitude, sky.latitude)
#
#        Mmunu = lalpulsar.ComputeMultiAMCoeffs(
#            multiDetStates=self.detector_states,
#            multiWeights=self.noise_weights,
#            skypos=sky,
#        ).Mmunu
#
#        if self.noise_weights is None:
#            Mmunu.Sinv_Tsft = self.Sinv_Tsft
#
#        return Mmunu
#
#    def _convert_amplitude_parameters(self, h0, cosi, aPlus, aCross):
#        """
#        Internal method to check and convert the given amplitude parameters
#        into the required format.
#        """
#        h0_cosi = h0 is not None and cosi is not None
#        aPlusCross = aPlus is not None and aCross is not None
#
#        if h0_cosi == aPlusCross:
#            raise ValueError("Need either (h0, cosi) or (aPlus, aCross), but not both")
#
#        if h0_cosi:
#            aPlus, aCross = convert_h0_cosi_to_aPlus_aCross(h0, cosi)
#
#        return aPlus, aCross
#
