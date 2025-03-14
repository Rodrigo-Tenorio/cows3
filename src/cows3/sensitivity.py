import numpy as np
from scipy import integrate, stats


def pfd_Fstatistic(
    twoF_threshold: float | np.ndarray,
    depth: float | np.ndarray,
    num_segments: int | np.ndarray,
    unitD_rho2_bins: np.ndarray,
    unitD_rho2_pdf: np.ndarray,
):
    """
    Compute the false-dismissal probability for a semicoherent F-statistic
    for a population of signals at a sensitivity depth `depth` given
    a unit-depth SNR^2 `unitD_rho2_pdf`.
    Integration is performed across the last axis of the arrays.

    twoF_threshold:
        Threshold at which the false-dismissal probability will be computed.
        This threshold corresponds to the semicoherent twoF statistic,
        which is distributed as a chi-squared distribution with `4 * num_segments`
        degrees of freedom in Gaussian noise.
    depth:
        Depth of the signal population.
    num_segments:
        Number of segments over which the semicoherent F-statistic is computed.
    unitD_rho2_bins:
        Values at which the unit-depth SNR^2 pdf is evaluated
    unitD_rho2_pdf:
        PDF of the unitary-depth signal distribution. This is the
        ``geometric factor'' R in Dreissigacker+ arXiv:1808.024
    """
    raise NotImplemented("There is a mistake in this formula. Will refactor")
    cdf_2F_rho2 = stats.ncx2(df=4 * num_segments, nc=unitD_rho2_bins).cdf(
        twoF_threshold
    )

    return integrate.simpson(y=unitD_rho2_pdf * cdf_2F_rho2, x=unitD_rho2_pdf)
