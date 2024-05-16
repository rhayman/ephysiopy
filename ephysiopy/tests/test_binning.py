from typing import Sequence

import numpy as np
import pytest
from ephysiopy.common.binning import RateMap, VariableToBin, MapType
from ephysiopy.common.utils import BinnedData
from ephysiopy.tests.conftest import *


@pytest.fixture
def standard_Ratemap(basic_PosCalcs):
    """Returns a Ratemap instance with a random walk as x,y"""
    P = basic_PosCalcs
    P.postprocesspos(tracker_params={"AxonaBadValue": 1023})
    # only have 10 seconds of spiking data so limit the pos stuff to that too
    # P.xy = P.xy[:, 0:10 * P.sample_rate]
    # P.dir = P.dir[0:10 * P.sample_rate]
    # P.speed = P.speed[0:10 * P.sample_rate]
    # P.npos = 10 * P.sample_rate
    return RateMap(P)


def test_calc_bin_size(standard_Ratemap):
    bs = standard_Ratemap._calc_bin_edges()
    print(f"Using {standard_Ratemap.ppm} ppm")
    assert isinstance(bs, (np.ndarray, list, Sequence))


# def test_bin_data(standard_Ratemap):
#     R = standard_Ratemap
#     xy = getattr(R, "xy")
#     xy_bins = R.binedges
#     hd = getattr(R, "dir")
#     R.inCms
#     R.inCms = True
#     hd_bins = np.arange(0, 360 + R.binsize, R.binsize)
#     samples = [xy, hd]
#     bins = [xy_bins, hd_bins]
#     pw2d = np.zeros(shape=[2, len(hd)])
#     pw2d[0, :] = R.pos_weights
#     pw2d[1, :] = R.pos_weights
#     pw = [R.pos_weights, pw2d]
#     keep = np.arange(len(hd))
#     for sample in zip(samples, bins, pw):
#         ret = R._bin_data(sample[0], sample[1], sample[2])
#         assert isinstance(ret, tuple)
#         assert isinstance(ret[0][0], np.ndarray)
#     breakpoint()
#     R._bin_data(xy, xy_bins)
#     R.pos_weights = np.random.randn(100)
#     R.smoothingType = "gaussian"


def test_get_map(standard_Ratemap):
    # A large number of the methods in RateMap are
    # called within the method get_map()
    n_pos = len(standard_Ratemap.pos_weights)
    spk_weights = np.random.rand(n_pos)
    spk_weights[spk_weights >= 0.95] = 1
    spk_weights[spk_weights >= 0.99] = 2
    spk_weights[spk_weights >= 0.99] = 3
    spk_weights[spk_weights < 0.95] = 0

    vars_2_bin = [VariableToBin.XY, VariableToBin.DIR, VariableToBin.SPEED]
    map_types = [MapType.RATE, MapType.POS]
    smoothing_when = ["after", "before"]
    do_smooth = [True, False]
    # There is a member variable to smooth before or after dividing
    # binned spikes by the spatial variable that needs to be set in the
    # iteration below

    for var in vars_2_bin:
        for map_type in map_types:
            for smooth in do_smooth:
                for when2smooth in smoothing_when:
                    standard_Ratemap.whenToSmooth = when2smooth
                    ret = standard_Ratemap.get_map(
                        spk_weights, var_type=var, map_type=map_type, smoothing=smooth
                    )
                    assert isinstance(ret, BinnedData)


def test_get_adaptive_map(standard_Ratemap):
    n_pos = len(standard_Ratemap.pos_weights)
    spk_weights = np.random.rand(n_pos)
    spk_weights[spk_weights >= 0.95] = 1
    spk_weights[spk_weights >= 0.99] = 2
    spk_weights[spk_weights >= 0.99] = 3
    spk_weights[spk_weights < 0.95] = 0

    rmap = standard_Ratemap.get_map(spk_weights)
    pos_binned = standard_Ratemap.get_map(spk_weights, map_type=MapType.POS)
    smthdRate = standard_Ratemap.getAdaptiveMap(
        rmap.binned_data[0], pos_binned.binned_data[0]
    )
    assert isinstance(smthdRate, tuple)
    assert isinstance(smthdRate[0], np.ndarray)


def test_cross_corr_2D(basic_BinnedData, standard_Ratemap):
    A = basic_BinnedData
    B = basic_BinnedData
    B.binned_data[0] = np.rot90(np.rot90(B.binned_data[0]))
    cc = standard_Ratemap.crossCorr2D(A, B)
    assert isinstance(cc, BinnedData)


def test_t_win_SAC(basic_xy, standard_Ratemap):
    x, y = basic_xy
    xy = np.array([x, y])
    t = np.random.rand(xy.shape[1])
    spk_idx = np.nonzero(t > 0.95)[0]
    H = standard_Ratemap.tWinSAC(xy, spk_idx)
    assert isinstance(H, BinnedData)
