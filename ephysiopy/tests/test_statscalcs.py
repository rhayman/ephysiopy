import numpy as np


def test_circ_r():
    t = np.linspace(0, 2*np.pi, 100)
    y = np.cos(t)
    from ephysiopy.common.statscalcs import circ_r
    r = circ_r(y)
    assert(isinstance(r, float))
    r = circ_r(y, d=0.2)
    assert(isinstance(r, float))


def test_mean_resultant_vector():
    t = np.linspace(0, 2*np.pi, 100)
    y = np.cos(t)
    from ephysiopy.common.statscalcs import mean_resultant_vector
    r, th = mean_resultant_vector(y)
    assert(isinstance(r, float))
    assert(isinstance(th, float))


def test_V_test():
    t = np.linspace(-np.pi, 4*np.pi, 100)
    y = np.cos(t)
    from ephysiopy.common.statscalcs import V_test
    v = V_test(y, 3.0)
    assert(isinstance(v, float))


def test_duplicates_as_complex():
    x = [9.9, 9.9, 12.3, 15.2, 15.2, 15.2]
    from ephysiopy.common.statscalcs import duplicates_as_complex
    ret = duplicates_as_complex(x)
    assert(isinstance(ret, np.ndarray))


def test_watsons_U2():
    t = np.linspace(0, 2*np.pi, 100)
    a = np.cos(t)
    b = np.roll(a, 20)
    from ephysiopy.common.statscalcs import watsonsU2
    u2 = watsonsU2(a, b)
    assert(isinstance(u2, float))


def test_watsons_U2_n():
    from numpy.random import default_rng
    rng = default_rng()
    # draw 100 samples from a von mises distribution
    # with a mode of 0 and a dispersion of 0.1
    # this should give a non-significant result for the
    # Watson U2n test
    y = np.rad2deg(rng.vonmises(0, 0.1, 100))
    from ephysiopy.common.statscalcs import watsonsU2n
    v = watsonsU2n(y)
    assert(isinstance(v, float))
    # draw 100 samples from a von mises distribution
    # with a mode of 0 and a dispersion of 0.6
    # this should give a significant result for the
    # Watson U2n test
    y = np.rad2deg(rng.vonmises(0, 0.6, 100))
    v = watsonsU2n(y)
    assert(isinstance(v, float))


def test_watson_williams():
    t = np.linspace(0, 2*np.pi, 100)
    a = np.cos(t)
    b = np.roll(a, 20)
    from ephysiopy.common.statscalcs import watsonWilliams
    u2 = watsonWilliams(a, b)
    assert(isinstance(u2, float))
