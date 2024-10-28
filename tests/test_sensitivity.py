import pytest

import numpy as np
from cows3.sensitivity import pfd_Fstatistic


@pytest.fixture
def unitD_rho2():
    bins = np.arange(1, 11)
    pdf = np.ones(bins.size) / bins.size
    return bins, pdf


def test_pfd_Fstatistic(unitD_rho2):
    pfd_Fstatistic(
        twoF_threshold=100,
        depth=10,
        num_segments=25,
        unitD_rho2_bins=unitD_rho2[0],
        unitD_rho2_pdf=unitD_rho2[1],
    )
