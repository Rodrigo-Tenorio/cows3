from functools import cache
import logging

import lal
import lalpulsar
import numpy as np

logger = logging.getLogger(__name__)


class SignalToNoiseRatio:
    r"""Compute the optimal SNR of a CW signal as expected in Gaussian noise.

    The definition of SNR (shortcut for "optimal signal-to-noise ratio")
    is taken from Eq. (76) of https://dcc.ligo.org/T0900149-v6/public and is
    such that :math:`\langle 2\mathcal{F}\rangle = 4 + \textrm{SNR}^2`,
    where  :math:`\langle 2\mathcal{F}\rangle` represents the expected
    value over noise realizations of twice the F-statistic of a template
    perfectly matched to an existing signal in the data.

    Computing :math:`\textrm{SNR}^2` requires two quantities:

    * | The antenna pattern matrix :math:`\mathcal{M}`, which depends on the sky position :math:`\vec{n}`
      | and polarization angle :math:`\psi` and encodes the effect of the detector's antenna pattern response
      | over the course of the observing run.
    * | The JKS amplitude parameters :math:`(\mathcal{A}^0, \mathcal{A}^1, \mathcal{A}^2, \mathcal{A}^3)`
      | [JKS1998]_ which are functions of the CW's amplitude parameters :math:`(h_0,\cos\iota, \psi, \phi_0)` or,
      | alternatively, :math:`(A_{+}, A_{\times}, \psi, \phi_0)`.

    .. [JKS1998] `Jaranowski, Krolak, Schuz Phys. Rev. D58 063001, 1998 <https://arxiv.org/abs/gr-qc/9804014>`_

    Parameters
    ----------
    mdss: lalpulsar.MultiDetectorStateSeries
        MultiDetectorStateSeries as produced by DetectorStates.
        Provides the required information to compute the antenna pattern contribution.
    noise_weights: Union[lalpulsar.MultiNoiseWeights, None]
        Optional, incompatible with `assumeSqrtSX`.
        Can be computed from SFTs using `SignalToNoiseRatio.from_sfts`.
        Noise weights to account for a varying noise floor or unequal noise
        floors in different detectors.
    assumeSqrtSX: float
        Optional, incompatible with `noise_weights`.
        Single-sided amplitude spectral density (ASD) of the detector noise.
        This value is used for all detectors, meaning it's not currently possible to manually
        specify different noise floors without creating SFT files.
        (To be improved in the future; developer note:
        will require SWIG constructor for MultiNoiseWeights.)

    This code is a subset of that under `snr.py` in [PyFstat](https://github.com/PyFstat/PyFstat).

    """

    def __init__(
        self,
        mdss: lalpulsar.MultiDetectorStateSeries,
        noise_weights: lalpulsar.MultiNoiseWeights | None = None,
        assumeSqrtSX: float | None = None,
    ):

        self.mdss = mdss
        self.noise_weights = noise_weights
        self.assumeSqrtSX = assumeSqrtSX

    @property
    def mdss(self) -> lalpulsar.MultiDetectorStateSeries:
        return self._mdss

    @property
    def noise_weights(self) -> lalpulsar.MultiNoiseWeights:
        return self._noise_weights

    @property
    def assumeSqrtSX(self) -> float | None:
        return self._assumeSqrtSX

    @mdss.setter
    def mdss(self, new_mdss: lalpulsar.MultiDetectorStateSeries):
        self._mdss = new_mdss
        self._T_sft = new_mdss.data[0].deltaT

    @noise_weights.setter
    def noise_weights(self, new_weights: lalpulsar.MultiNoiseWeights):
        if (new_weights is not None) and (
            getattr(self, "assumeSqrtSX", None) is not None
        ):
            raise ValueError(
                "Cannot set `noise_weights` if `assumeSqrtSX is already set!"
            )
        self._noise_weights = new_weights
        self._Sinv_Tsft = None

    @assumeSqrtSX.setter
    def assumeSqrtSX(self, new_sqrtSX: float):
        if getattr(self, "noise_weights", None) is not None:
            raise ValueError(
                "Cannot set `assumeSqrtSX' if `noise_weights` is already set!"
            )
        self._assumeSqrtSX = new_sqrtSX
        self._T_sft = self.mdss.data[0].deltaT
        self._Sinv_Tsft = self._T_sft / new_sqrtSX**2

    def compute_snr2(
        self,
        Alpha: float,
        Delta: float,
        psi: float,
        phi0: float,
        aPlus: float,
        aCross: float,
    ) -> float:
        r"""
        Compute the :math:`\textrm{SNR}^2` of a CW signal using XLALComputeOptimalSNR2FromMmunu.
        Parameters correspond to the standard ones used to describe a CW
        (see e.g. Eqs. (16), (26), (30) of https://dcc.ligo.org/T0900149-v6/public ).

        Mind that this function returns *squared* SNR
        (Eq. (76) of https://dcc.ligo.org/T0900149-v6/public ),
        which can be directly related to the expected F-statistic as
        :math:`\langle 2\mathcal{F}\rangle = 4 + \textrm{SNR}^2`.

        Parameters
        ----------
        Alpha: float
            Right ascension (equatorial longitude) of the signal in radians.
        Delta: float
            Declination (equatorial latitude) of the signal in radians.
        psi: float
            Polarization angle.
        phi0: float
            Initial phase.
        aPlus: float
            Plus polarization amplitude, equal to `0.5 * h0 (1 + cosi^2)`.
        aCross: float
            Cross polarization amplitude, equal to `h0 * cosi`.

        Returns
        -------
        SNR^2: float
            Squared signal-to-noise ratio of a CW signal consistent
            with the specified parameters in the specified detector
            network.
        """

        Aphys = lalpulsar.PulsarAmplitudeParams()
        Aphys.psi = psi
        Aphys.phi0 = phi0
        Aphys.aPlus = aPlus
        Aphys.aCross = aCross

        M = self.compute_Mmunu(Alpha=Alpha, Delta=Delta)

        return lalpulsar.ComputeOptimalSNR2FromMmunu(Aphys, M)

    def compute_Mmunu(self, Alpha: float, Delta: float) -> float:
        """
        Compute Mmunu matrix at a specific sky position using the detector states
        (and possible noise weights) given at initialization time.
        If no noise weights were given, unit weights are assumed.

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
            multiDetStates=self.mdss,
            multiWeights=self.noise_weights,
            skypos=sky,
        ).Mmunu

        if self.noise_weights is None:
            Mmunu.Sinv_Tsft = self._Sinv_Tsft

        return Mmunu
