import numpy as np
from ephysiopy.common.gridcell import SAC


def test_get_measures_and_show(basic_ratemap):
    S = SAC()
    nodwell = ~np.isfinite(basic_ratemap)
    sac = S.autoCorr2D(basic_ratemap, nodwell)
    measures = S.getMeasures(sac)
    assert isinstance(measures, dict)
    S.show(sac, measures)
