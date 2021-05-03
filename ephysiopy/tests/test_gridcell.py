import pytest
import numpy as np
import matplotlib.pylab as plt
from ephysiopy.common.gridcell import SAC


def test_auto_corr_2D(basic_ratemap):
    S = SAC()
    nodwell = ~np.isfinite(basic_ratemap)
    sac = S.autoCorr2D(basic_ratemap, nodwell)
    assert(isinstance(sac, np.ndarray))


def test_cross_corr_2D(basic_ratemap):
    A = basic_ratemap
    B = np.rot90(np.rot90(A))
    A_dwell = ~np.isfinite(A)
    B_dwell = ~np.isfinite(B)
    S = SAC()
    cc = S.crossCorr2D(A, B, A_dwell, B_dwell)
    assert(isinstance(cc, np.ndarray))


def test_t_win_SAC(basic_xy):
    x, y = basic_xy
    xy = np.array([x, y])
    t = np.random.rand(xy.shape[1])
    spk_idx = np.nonzero(t > 0.95)[0]
    S = SAC()
    H = S.t_win_SAC(xy, spk_idx)
    assert(isinstance(H, np.ndarray))


@pytest.mark.mpl_image_compare
def test_get_measures_and_show(basic_ratemap):
    S = SAC()
    nodwell = ~np.isfinite(basic_ratemap)
    sac = S.autoCorr2D(basic_ratemap, nodwell)
    measures = S.getMeasures(sac)
    assert(isinstance(measures, dict))
    S.show(sac, measures)
    fig = plt.gcf()
    return fig
