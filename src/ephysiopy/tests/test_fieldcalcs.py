import numpy as np
import pytest
from ephysiopy.common import fieldcalcs
from ephysiopy.common.utils import corr_maps


def test_limit_to_one(basic_ratemap):
    _, middle_field, _ = fieldcalcs.limit_to_one(basic_ratemap)
    assert isinstance(middle_field, np.ndarray)
    basic_ratemap[1::, :] = np.nan
    fieldcalcs.limit_to_one(basic_ratemap)


def test_get_border_score(basic_ratemap):
    fieldcalcs.border_score(basic_ratemap)
    fieldcalcs.border_score(basic_ratemap, shape="circle")
    rmap_copy = basic_ratemap.copy()
    rmap_copy[1::, :] = np.nan
    rmap_copy[:, 2::] = np.nan
    fieldcalcs.border_score(rmap_copy)
    rmap_copy = basic_ratemap.copy()
    rmap_copy[1:-1, 1:-1] = 0
    fieldcalcs.border_score(rmap_copy)
    fieldcalcs.border_score(rmap_copy, minArea=1)


def test_corr_maps(basic_ratemap):
    flipped_map = np.rot90(basic_ratemap)
    cc = corr_maps(basic_ratemap, flipped_map)
    assert isinstance(cc, float)
    flipped_map = flipped_map[1::, :]
    corr_maps(basic_ratemap, flipped_map, maptype="grid")
    corr_maps(flipped_map, basic_ratemap, maptype="grid")
    flipped_map[:, :] = np.nan
    flipped_map[0, 0] = 0
    corr_maps(flipped_map, basic_ratemap)


def test_coherence(basic_BinnedData):
    M = basic_BinnedData
    blurred = fieldcalcs.blur_image(M, n=15)
    coh = fieldcalcs.coherence(M.binned_data[0], blurred.binned_data[0])
    assert isinstance(coh, float)


def test_kldiv_dir():
    t = np.linspace(0, 2 * np.pi, 100)
    y = np.cos(t)
    kldiv = fieldcalcs.kldiv_dir(y)
    assert isinstance(kldiv, float)


def test_kldiv():
    n = 100
    X = np.linspace(0, 2 * np.pi, n)
    y1 = np.cos(X)
    y2 = np.ones_like(y1) / n
    fieldcalcs.kldiv(X, y1, y2)
    X = X[1::]
    with pytest.raises(ValueError):
        fieldcalcs.kldiv(X, y1, y2)
    X = np.linspace(0, 2 * np.pi, n)
    fieldcalcs.kldiv(X, y1, y2, variant="js")
    fieldcalcs.kldiv(X, y1, y2, variant="sym")
    with pytest.warns(UserWarning):
        fieldcalcs.kldiv(X, y1, y2, variant="error")
    # make probabilites sum to > 1
    y2 = np.ones_like(y1)
    with pytest.warns(UserWarning):
        fieldcalcs.kldiv(X, y1, y2, variant="js")


def test_skaggs_info(basic_ratemap):
    dwell_times = np.random.rand(basic_ratemap.shape[0], basic_ratemap.shape[1])
    dwell_times = dwell_times / np.sum(dwell_times)
    dwell_times = dwell_times * 10
    skaggs = fieldcalcs.skaggs_info(basic_ratemap, dwell_times)
    assert isinstance(skaggs, float)
    fieldcalcs.skaggs_info(basic_ratemap, dwell_times, sample_rate=30)
    basic_ratemap[:, :] = 0
    fieldcalcs.skaggs_info(basic_ratemap, dwell_times)
