import pytest

from ephysiopy.common.binning import RateMap
import numpy as np


@pytest.fixture
def standard_Ratemap(basic_PosCalcs):
    '''Returns a Ratemap instance with a random walk as x,y'''
    P = basic_PosCalcs
    xy, hdir = P.postprocesspos()
    # only have 10 seconds of spiking data so limit the pos stuff to that too
    P.xy = P.xy[:, 0:10*P.sample_rate]
    P.dir = P.dir[0:10*P.sample_rate]
    P.speed = P.speed[0:10*P.sample_rate]
    P.npos = 10*P.sample_rate
    return RateMap(P.xy, P.dir, P.speed)


def test_calc_bin_size(standard_Ratemap):
    bs = standard_Ratemap.__calcBinSize__()
    assert(isinstance(bs, np.ndarray))


def test_get_map(standard_Ratemap):
    # A large number of the methods in RateMap are
    # called within the method getMap()
    n_pos = len(standard_Ratemap.pos_weights)
    spk_weights = np.random.rand(n_pos)
    spk_weights[spk_weights >= 0.95] = 1
    spk_weights[spk_weights >= 0.99] = 2
    spk_weights[spk_weights >= 0.99] = 3
    spk_weights[spk_weights < 0.95] = 0

    vars_2_bin = ['xy', 'dir', 'speed']
    map_types = ['rate', 'pos']
    smoothing_when = ['after', 'before']
    # There is a member variable to smooth before or after dividing
    # binned spikes by the spatial variable that needs to be set in the
    # iteration below

    for var in vars_2_bin:
        for map_type in map_types:
            for when2smooth in smoothing_when:
                rmap, bin_edges = standard_Ratemap.getMap(
                    spk_weights, varType=var, mapType=map_type,
                    smoothing=when2smooth)
                assert(isinstance(rmap, np.ndarray))


def test_get_adaptive_map(standard_Ratemap):
    n_pos = len(standard_Ratemap.pos_weights)
    spk_weights = np.random.rand(n_pos)
    spk_weights[spk_weights >= 0.95] = 1
    spk_weights[spk_weights >= 0.99] = 2
    spk_weights[spk_weights >= 0.99] = 3
    spk_weights[spk_weights < 0.95] = 0

    rmap, _ = standard_Ratemap.getMap(
        spk_weights)
    pos_binned, _ = standard_Ratemap.getMap(
        spk_weights, mapType='pos')
    pos_binned[~np.isfinite(pos_binned)] = 0
    smthdRate, smthdSpk, smthdPos = standard_Ratemap.getAdaptiveMap(
        rmap, pos_binned)
    assert(isinstance(smthdRate, np.ndarray))


def test_auto_corr_2D(basic_ratemap, standard_Ratemap):
    nodwell = ~np.isfinite(basic_ratemap)
    SAC = standard_Ratemap.autoCorr2D(basic_ratemap, nodwell)
    assert(isinstance(SAC, np.ndarray))


def test_cross_corr_2D(basic_ratemap, standard_Ratemap):
    A = basic_ratemap
    B = np.rot90(np.rot90(A))
    A_dwell = ~np.isfinite(A)
    B_dwell = ~np.isfinite(B)
    cc = standard_Ratemap.crossCorr2D(A, B, A_dwell, B_dwell)
    assert(isinstance(cc, np.ndarray))


def test_t_win_SAC(basic_xy, standard_Ratemap):
    x, y = basic_xy
    xy = np.array([x, y])
    t = np.random.rand(xy.shape[1])
    spk_idx = np.nonzero(t > 0.95)[0]
    H = standard_Ratemap.tWinSAC(xy, spk_idx)
    assert(isinstance(H, np.ndarray))
