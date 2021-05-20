import numpy as np
import pytest
from ephysiopy.common import fieldcalcs


def test_limit_to_one(basic_ratemap):
    _, middle_field, _ = fieldcalcs.limit_to_one(basic_ratemap)
    assert(isinstance(middle_field, np.ndarray))
    basic_ratemap[1::, :] = np.nan
    fieldcalcs.limit_to_one(basic_ratemap)


def test_global_threshold(basic_ratemap):
    fieldcalcs.global_threshold(basic_ratemap)


def test_local_threshold(basic_ratemap):
    A = fieldcalcs.local_threshold(basic_ratemap)
    assert(isinstance(A, np.ndarray))


def test_get_border_score(basic_ratemap):
    fieldcalcs.border_score(basic_ratemap)
    fieldcalcs.border_score(basic_ratemap, shape='circle')
    rmap_copy = basic_ratemap.copy()
    rmap_copy[1::, :] = np.nan
    rmap_copy[:, 2::] = np.nan
    fieldcalcs.border_score(rmap_copy)
    rmap_copy = basic_ratemap.copy()
    rmap_copy[1:-1, 1:-1] = 0
    fieldcalcs.border_score(rmap_copy)
    fieldcalcs.border_score(rmap_copy, minArea=1)


def test_field_props(basic_ratemap):
    fp = fieldcalcs.field_props(basic_ratemap)
    assert(isinstance(fp, dict))
    fieldcalcs.field_props(
        basic_ratemap, clear_border=True,
        neighbours=100,
        calc_angs=True,
        min_distance=5)
    # test something that should fail as it's poorly formed
    x, y = np.indices((80, 80))
    x1, y1, x2, y2 = 28, 28, 44, 52
    r1, r2 = 16, 20
    mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2
    mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2
    image = np.logical_or(mask_circle1, mask_circle2)
    from ephysiopy.common.utils import blurImage
    im = blurImage(image, 15)
    im[im < 0.1] = 0
    fieldcalcs.field_props(im)


def test_corr_maps(basic_ratemap):
    flipped_map = np.rot90(basic_ratemap)
    cc = fieldcalcs.corr_maps(basic_ratemap, flipped_map)
    assert(isinstance(cc, float))
    flipped_map = flipped_map[1::, :]
    fieldcalcs.corr_maps(basic_ratemap, flipped_map, maptype='grid')
    fieldcalcs.corr_maps(flipped_map, basic_ratemap, maptype='grid')
    flipped_map[:, :] = np.nan
    flipped_map[0, 0] = 0
    fieldcalcs.corr_maps(flipped_map, basic_ratemap)


def test_coherence(basic_ratemap):
    blurred = fieldcalcs.blurImage(basic_ratemap, n=15)
    coh = fieldcalcs.coherence(basic_ratemap, blurred)
    assert(isinstance(coh, float))


def test_kldiv_dir():
    t = np.linspace(0, 2*np.pi, 100)
    y = np.cos(t)
    kldiv = fieldcalcs.kldiv_dir(y)
    assert(isinstance(kldiv, float))


def test_kldiv():
    n = 100
    X = np.linspace(0, 2*np.pi, n)
    y1 = np.cos(X)
    y2 = np.ones_like(y1) / n
    fieldcalcs.kldiv(X, y1, y2)
    X = X[1::]
    with pytest.raises(ValueError):
        fieldcalcs.kldiv(X, y1, y2)
    X = np.linspace(0, 2*np.pi, n)
    y2 = np.ones_like(y1)
    fieldcalcs.kldiv(X, y1, y2, variant='js')
    fieldcalcs.kldiv(X, y1, y2, variant='sym')
    fieldcalcs.kldiv(X, y1, y2, variant='error')


def test_skaggs_info(basic_ratemap):
    dwell_times = np.random.rand(
        basic_ratemap.shape[0], basic_ratemap.shape[1])
    dwell_times = dwell_times / np.sum(dwell_times)
    dwell_times = dwell_times * 10
    skaggs = fieldcalcs.skaggs_info(basic_ratemap, dwell_times)
    assert(isinstance(skaggs, float))
    fieldcalcs.skaggs_info(basic_ratemap, dwell_times, sample_rate=30)
    basic_ratemap[:, :] = 0
    fieldcalcs.skaggs_info(basic_ratemap, dwell_times)


def test_grid_field_measures(basic_ratemap):
    # Set allProps to True to try the ellipse fitting stuff
    measures = fieldcalcs.grid_field_props(basic_ratemap, allProps=True)
    assert(isinstance(measures, dict))
    fieldcalcs.grid_field_props(
        basic_ratemap, min_distance=10,
        maxima='single',
        step=15)


def test_deform_SAC(basic_ratemap):
    from ephysiopy.common.binning import RateMap
    R = RateMap()
    sac = R.autoCorr2D(
        basic_ratemap, ~np.isfinite(basic_ratemap))
    from skimage import transform
    A = transform.AffineTransform(
        scale=[1, 1.15], translation=[0, -15])
    sac = transform.warp(sac, A.inverse)
    deformed_SAC = fieldcalcs.deform_SAC(sac)
    assert(isinstance(deformed_SAC, np.ndarray))
    A = np.zeros_like(basic_ratemap)
    A[3:10, 3:8] = 10
    nodwell = ~np.isfinite(A)
    sac = R.autoCorr2D(A, nodwell)
    fieldcalcs.deform_SAC(sac)
    fieldcalcs.deform_SAC(
        sac, np.array([[3,9],[10,2]]), np.array([[1,9],[10,2]]))


def test_get_grid_orientation(basic_ratemap):
    from ephysiopy.common.gridcell import SAC
    S = SAC()
    nodwell = ~np.isfinite(basic_ratemap)
    sac = S.autoCorr2D(basic_ratemap, nodwell)
    measures = fieldcalcs.grid_field_props(sac, allProps=True)
    peak_coords = measures['closest_peak_coords']
    fieldcalcs.grid_orientation(peak_coords, np.arange(len(peak_coords)))
    A = np.zeros_like(basic_ratemap)
    A[3:10, 3:8] = 10
    nodwell = ~np.isfinite(A)
    sac = S.autoCorr2D(A, nodwell)
    measures = fieldcalcs.grid_field_props(sac, allProps=True)
    peak_coords = measures['closest_peak_coords']
    fieldcalcs.grid_orientation(peak_coords, np.arange(len(peak_coords)))