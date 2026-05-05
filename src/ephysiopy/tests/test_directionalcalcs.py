import numpy as np
import pytest
from ephysiopy.common.directionalcalcs import HeadDirectionCalcs


@pytest.fixture
def sample_head_directions():
    # Create a masked array with some head direction data (degrees 0-360)
    data = np.array([10, 20, 30, 40, 350, 0, 90, 180, 270, 360])
    mask = np.array([False] * len(data))
    return np.ma.MaskedArray(data, mask=mask)


def test_init(sample_head_directions):
    hd = HeadDirectionCalcs(sample_head_directions)
    assert hasattr(hd, "head_directions")


def test_mean(sample_head_directions):
    hd = HeadDirectionCalcs(sample_head_directions)
    mean = hd.mean()
    assert isinstance(mean, float)


def test_variability(sample_head_directions):
    hd = HeadDirectionCalcs(sample_head_directions)
    var = hd.variability()
    assert isinstance(var, float)


def test_consistency(sample_head_directions):
    hd = HeadDirectionCalcs(sample_head_directions)
    cons = hd.consistency()
    assert isinstance(cons, float)


def test_dispersion(sample_head_directions):
    hd = HeadDirectionCalcs(sample_head_directions)
    disp = hd.dispersion()
    assert isinstance(disp, float)


def test_entropy(sample_head_directions):
    hd = HeadDirectionCalcs(sample_head_directions)
    ent = hd.entropy()
    assert isinstance(ent, float)


def test_kurtosis(sample_head_directions):
    hd = HeadDirectionCalcs(sample_head_directions)
    kurt = hd.kurtosis()
    assert isinstance(kurt, float)


def test_rayleigh_test(sample_head_directions):
    hd = HeadDirectionCalcs(sample_head_directions)
    result = hd.rayleigh_test()
    assert hasattr(result, "pval")


def test_omnibus_test(sample_head_directions):
    hd = HeadDirectionCalcs(sample_head_directions)
    result = hd.omnibus_test()
    assert hasattr(result, "pval")


def test_plot(sample_head_directions):
    hd = HeadDirectionCalcs(sample_head_directions)
    ax = hd.plot()
    assert hasattr(ax, "plot")
