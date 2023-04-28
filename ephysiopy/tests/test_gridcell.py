import numpy as np
from ephysiopy.common.gridcell import SAC
from ephysiopy.tests.test_binning import standard_Ratemap


def test_get_measures_and_show(basic_ratemap):
    R = standard_Ratemap
    nodwell = ~np.isfinite(basic_ratemap)
    sac = R.autoCorr2D(basic_ratemap, nodwell)
    S = SAC()
    measures = S.getMeasures(sac)
    assert isinstance(measures, dict)
    S.show(sac, measures)
