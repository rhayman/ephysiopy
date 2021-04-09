import pytest

from ephysiopy.common.binning import RateMap
import numpy as np


@pytest.fixture
def standard_ratemap(basic_xy):
    '''Returns a Ratemap instance with a random walk as x,y'''
    xy = np.array([basic_xy[0], basic_xy[1]])
    return RateMap(xy)
