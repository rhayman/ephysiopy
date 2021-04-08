import pytest

from ephysiopy.common.binning import RateMap
import os
import numpy as np

@pytest.fixture
def standard_ratemap():
    '''Returns a Ratemap instance with a random walk as x,y'''
    xy_test_data_path = os.path.join("../data/random_walk_xy.npy")
    xy = np.load(xy_test_data_path)
    return RateMap(xy)

