import numpy as np
from ephysiopy.common.gridcell import SAC
from ephysiopy.tests.test_binning import standard_Ratemap


# def test_get_measures_and_show(standard_Ratemap):
#     n_pos = len(standard_Ratemap.pos_weights)
#     spk_weights = np.random.rand(n_pos)
#     spk_weights[spk_weights >= 0.95] = 1
#     basic_ratemap, _ = standard_Ratemap.getMap(spk_weights)
#     nodwell = ~np.isfinite(basic_ratemap)
#     sac = standard_Ratemap.autoCorr2D(basic_ratemap, nodwell)
#     S = SAC()
#     measures = S.getMeasures(sac)
#     assert isinstance(measures, dict)
#     S.show(sac, measures)
