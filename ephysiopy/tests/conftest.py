import pytest

import os
import numpy as np


@pytest.fixture(scope="module")
def basic_xy():
    '''
    Returns a random 2D walk as x, y tuple
    '''
    xy_test_data_path = os.path.join("../data/random_walk_xy.npy")
    xy = np.load(xy_test_data_path)
    x = xy[0, :]
    y = xy[:, 1]
    return x, y
