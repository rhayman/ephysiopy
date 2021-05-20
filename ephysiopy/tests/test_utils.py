import numpy as np
import pytest
from ephysiopy.common import utils


def test_smooth():
    x = np.random.rand(100)
    x = list(x)
    with pytest.raises(ValueError):
        utils.smooth(np.atleast_2d(x))
    with pytest.raises(ValueError):
        utils.smooth(x, window_len=len(x)+1)
    utils.smooth(x, window_len=2)
    utils.smooth(x, window_len=6)
    with pytest.raises(ValueError):
        utils.smooth(x, window='deliberate_error')
    utils.smooth(x, window='flat')
    y = utils.smooth(x, window='hamming')
    assert(isinstance(y, np.ndarray))


def test_blur_image(basic_ratemap):
    filt = ['box', 'gaussian']
    rmap1D = basic_ratemap[0, :]
    rmap2D = basic_ratemap
    rmap3D = np.atleast_3d(rmap2D)
    rmaps = [rmap1D, rmap2D, rmap3D]
    b = utils.blurImage(rmap2D, 3, 4)
    for f in filt:
        for rmap in rmaps:
            b = utils.blurImage(rmap, 3, ftype=f)
            assert(isinstance(b, np.ndarray))


def test_count_to():
    n = [0, 0, 3, 0, 0, 2, 0, 2, 1]
    n = np.array(n)
    with pytest.raises(Exception):
        utils.count_to(np.atleast_2d(n))
    y = utils.count_to(n)
    assert(isinstance(y, np.ndarray))


def test_repeat_ind():
    n = [0, 0, 3, 0, 0, 2, 0, 2, 1]
    n = np.array(n)
    with pytest.raises(Exception):
        utils.repeat_ind(np.atleast_2d(n))
    res = utils.repeat_ind(n)
    assert(isinstance(res, np.ndarray))


def test_rect():
    from numpy.random import default_rng
    rng = default_rng()
    x = rng.vonmises(0, 0.1, 100)
    y = rng.vonmises(0, 0.1, 100)
    utils.rect(x, y)
    r, _ = utils.rect(
        np.rad2deg(x),
        np.rad2deg(y),
        deg=True)
    assert(isinstance(r, np.ndarray))


def test_polar():
    x = np.random.randint(0, 10, 20)
    y = np.random.randint(0, 10, 20)
    r, _ = utils.polar(x, y)
    assert(isinstance(r, np.ndarray))
    r, _ = utils.polar(x, y, deg=True)
    assert(isinstance(r, np.ndarray))


def test_bwperim(basic_ratemap):
    with pytest.raises(ValueError):
        utils.bwperim(basic_ratemap, n=2)
    res = utils.bwperim(basic_ratemap, n=8)
    assert(isinstance(res, np.ndarray))
