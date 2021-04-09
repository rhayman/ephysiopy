import pytest
import numpy as np
from ephysiopy.common.ephys_generic import PosCalcsGeneric


@pytest.fixture
def basic_PosCalcs(basic_xy):
    '''
    Returns a PosCalcsGeneric instance initialised with some random
    walk xy data
    '''
    x = basic_xy[0]
    y = basic_xy[1]
    ppm = 300  # pixels per metre value
    return PosCalcsGeneric(x, y, ppm)


def test_speedfilter(basic_PosCalcs, basic_xy):
    xy = np.array([basic_xy[0], basic_xy[1]])
    new_xy = basic_PosCalcs.speedfilter(xy)
    assert(new_xy.ndim == 2)
    assert(xy.shape == new_xy.shape)


def test_interpnans(basic_PosCalcs, basic_xy):
    xy = np.array([basic_xy[0], basic_xy[1]])
    new_xy = basic_PosCalcs.interpnans(xy)
    assert(new_xy.ndim == 2)
    assert(xy.shape == new_xy.shape)


def test_smoothPos(basic_PosCalcs, basic_xy):
    xy = np.array([basic_xy[0], basic_xy[1]])
    new_xy = basic_PosCalcs.smoothPos(xy)
    assert(new_xy.ndim == 2)
    assert(xy.shape == new_xy.shape)


def test_calcSpeed(basic_PosCalcs, basic_xy):
    xy = np.array([basic_xy[0], basic_xy[1]])
    basic_PosCalcs.calcSpeed(xy)
    speed = basic_PosCalcs.speed
    assert(speed.ndim == 1)
    assert(xy.shape[1] == speed.shape[0])


# postprocesspos calls the functions in the above 4 tests
def test_postprocesspos(basic_PosCalcs):
    xy, hdir = basic_PosCalcs.postprocesspos({})
    assert(xy.ndim == 2)
    assert(hdir.ndim == 1)
    assert(xy.shape[1] == hdir.shape[0])


def test_upsamplePos(basic_PosCalcs, basic_xy):
    xy = np.array([basic_xy[0], basic_xy[1]])
    # default upsample rate is 50Hz (for Axona)
    # default sample rate for me is 30Hz
    new_xy = basic_PosCalcs.upsamplePos(xy)
    new_len = np.ceil(xy.shape[1] * (30 / 50.)).astype(int)
    assert(new_xy.ndim == 2)
    assert(new_xy.shape[1] == new_len)
