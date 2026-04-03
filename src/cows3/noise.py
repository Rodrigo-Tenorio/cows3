from collections.abc import Sequence

import lal
import lalpulsar


def make_multi_noise_weights_from_asd(
    mdss: lalpulsar.MultiDetectorStateSeries,
    sqrtSX_list: Sequence[float],
) -> lalpulsar.MultiNoiseWeights:
    """Construct ``lalpulsar.MultiNoiseWeights`` from per-detector ASD values.

    Parameters
    ----------
    mdss: lalpulsar.MultiDetectorStateSeries
        Detector states used to define detector and per-SFT dimensions.
    sqrtSX_list: Sequence[float]
        Per-detector single-sided ASDs in detector order.

    Returns
    -------
    lalpulsar.MultiNoiseWeights
        Multi-detector noise weights with one entry per SFT for each detector.
    """

    if len(sqrtSX_list) == 0:
        raise ValueError("`sqrtSX_list` must contain at least one detector ASD value.")
    if len(sqrtSX_list) != mdss.length:
        raise ValueError(
            "`sqrtSX_list` length must match the number of detectors in `mdss`."
        )
    if any(sqrtSX <= 0 for sqrtSX in sqrtSX_list):
        raise ValueError("All `sqrtSX_list` entries must be positive.")

    multi_noise_weights = lalpulsar.CreateMultiNoiseWeights(mdss.length)

    # Compute the global normalization first so per-SFT weights are O(1),
    # following the convention used by LAL's multi-detector weighting.
    sinv_tsft_per_ifo = [
        mdss.data[ifo_ind].deltaT / float(sqrtSX) ** 2
        for ifo_ind, sqrtSX in enumerate(sqrtSX_list)
    ]

    total_num_sfts = sum(mdss.data[ifo_ind].length for ifo_ind in range(mdss.length))
    avg_sinv_tsft = (
        sum(
            sinv_tsft_per_ifo[ifo_ind] * mdss.data[ifo_ind].length
            for ifo_ind in range(mdss.length)
        )
        / total_num_sfts
    )
    multi_noise_weights.Sinv_Tsft = avg_sinv_tsft
    multi_noise_weights.isNotNormalized = False

    for ifo_ind, sqrtSX in enumerate(sqrtSX_list):
        weights = lal.CreateREAL8Vector(mdss.data[ifo_ind].length)
        weights.data[:] = sinv_tsft_per_ifo[ifo_ind] / avg_sinv_tsft
        multi_noise_weights.data[ifo_ind] = weights

    return multi_noise_weights