import numpy as np
from scipy import optimize
from scipy.stats import norm
from astropy.stats.circstats import rayleightest
from dataclasses import dataclass


def circ_r(alpha, w=None, d=0, axis=0):
    """
    Computes the mean resultant vector length for circular data.

    Args:
        alpha (array or list): Sample of angles in radians.
        w (array or list): Counts in the case of binned data.
            Must be same length as alpha.
        d (array or list, optional): Spacing of bin centres for binned data; if
            supplied, correction factor is used to correct for bias in
            estimation of r, in radians.
        axis (int, optional): The dimension along which to compute.
            Default is 0.

    Returns:
        r (float): The mean resultant vector length.
    """

    if w is None:
        w = np.ones_like(alpha, dtype=float)
    # TODO: error check for size constancy
    r = np.sum(w * np.exp(1j * alpha))
    r = np.abs(r) / np.sum(w)
    if d != 0:
        c = d / 2.0 / np.sin(d / 2.0)
        r = c * r
    return r


def mean_resultant_vector(angles):
    """
    Calculate the mean resultant length and direction for angles.

    Args:
        angles (np.array): Sample of angles in radians.

    Returns:
        r (float): The mean resultant vector length.
        th (float): The mean resultant vector direction.

    Notes:
    Taken from Directional Statistics by Mardia & Jupp, 2000
    """
    if len(angles) == 0:
        return 0, 0
    S = np.sum(np.sin(angles)) * (1 / float(len(angles)))
    C = np.sum(np.cos(angles)) * (1 / float(len(angles)))
    r = np.hypot(S, C)
    th = np.arctan2(S, C)
    return r, th


def rayleigh_test(angles: np.ndarray) -> float:
    """
    Perform the Rayleigh test for uniformity of circular data.

    Args:
        angles (array_like): Vector of angular values in radians.

    Returns:
        Z (float): The Rayleigh test statistic.
        p_value (float): The p-value for the test.
    """
    angles = np.asarray(angles)
    return rayleightest(angles)


def V_test(angles, test_direction):
    """
    The Watson U2 tests whether the observed angles have a tendency to
    cluster around a given angle indicating a lack of randomness in the
    distribution. Also known as the modified Rayleigh test.

    Args:
        angles (array_like): Vector of angular values in degrees.
        test_direction (int): A single angular value in degrees.

    Notes:
        For grouped data the length of the mean vector must be adjusted,
        and for axial data all angles must be doubled.
    """
    n = len(angles)
    x_hat = np.sum(np.cos(np.radians(angles))) / float(n)
    y_hat = np.sum(np.sin(np.radians(angles))) / float(n)
    r = np.sqrt(x_hat**2 + y_hat**2)
    theta_hat = np.degrees(np.arctan(y_hat / x_hat))
    v_squiggle = r * np.cos(np.radians(theta_hat) - np.radians(test_direction))
    V = np.sqrt(2 * n) * v_squiggle
    return V


def duplicates_as_complex(x, already_sorted=False):
    """
    Finds duplicates in x

    Args:
        x (array_like): The list to find duplicates in.
        already_sorted (bool, optional): Whether x is already sorted.
            Default False.

    Returns:
        x (array_like): A complex array where the complex part is the count of
            the number of duplicates of the real value.

    Examples:
        >>>	x = [9.9, 9.9, 12.3, 15.2, 15.2, 15.2]
        >>> ret = duplicates_as_complex(x)
        >>>	print(ret)
        [9.9+0j, 9.9+1j,  12.3+0j, 15.2+0j, 15.2+1j, 15.2+2j]
    """

    if not already_sorted:
        x = np.sort(x)
    is_start = np.empty(len(x), dtype=bool)
    is_start[0], is_start[1:] = True, x[:-1] != x[1:]
    labels = np.cumsum(is_start) - 1
    sub_idx = np.arange(len(x)) - np.nonzero(is_start)[0][labels]
    return x + 1j * sub_idx


def watsonsU2(a, b):
    """
    Tests whether two samples from circular observations differ significantly
    from each other with regard to mean direction or angular variance.

    Args:
        a, b (array_like): The two samples to be tested

    Returns:
        U2 (float): The test statistic

    Notes:
        Both samples must come from a continuous distribution. In the case of
        grouping the class interval should not exceed 5.
        Taken from '100 Statistical Tests' G.J.Kanji, 2006 Sage Publications
    """

    a = np.sort(np.ravel(a))
    b = np.sort(np.ravel(b))
    n_a = len(a)
    n_b = len(b)
    N = float(n_a + n_b)
    a_complex = duplicates_as_complex(a, True)
    b_complex = duplicates_as_complex(b, True)
    a_and_b = np.union1d(a_complex, b_complex)

    # get index for a
    a_ind = np.zeros(len(a_and_b), dtype=int)
    a_ind[np.searchsorted(a_and_b, a_complex)] = 1
    a_ind = np.cumsum(a_ind)

    # same for b
    b_ind = np.zeros(len(a_and_b), dtype=int)
    b_ind[np.searchsorted(a_and_b, b_complex)] = 1
    b_ind = np.cumsum(b_ind)

    d_k = (a_ind / float(n_a)) - (b_ind / float(n_b))

    d_k_sq = d_k**2

    U2 = ((n_a * n_b) / N**2) * (np.sum(d_k_sq) - ((np.sum(d_k) ** 2) / N))
    return U2


def watsonsU2n(angles):
    """
    Tests whether the given distribution fits a random sample of angular
    values.

    Args:
        angles (array_like): The angular samples.

    Returns:
        U2n (float): The test statistic.

    Notes:
        This test is suitable for both unimodal and the multimodal cases.
        It can be used as a test for randomness.
        Taken from '100 Statistical Tests' G.J.Kanji, 2006 Sage Publications.
    """

    angles = np.sort(angles)
    n = len(angles)
    Vi = angles / float(360)
    sum_Vi = np.sum(Vi)
    sum_sq_Vi = np.sum(Vi**2)
    Ci = (2 * np.arange(1, n + 1)) - 1
    sum_Ci_Vi_ov_n = np.sum(Ci * Vi / n)
    V_bar = (1 / float(n)) * sum_Vi
    U2n = sum_sq_Vi - sum_Ci_Vi_ov_n + \
        (n * (1 / float(3) - (V_bar - 0.5) ** 2))
    test_vals = {
        "0.1": 0.152,
        "0.05": 0.187,
        "0.025": 0.221,
        "0.01": 0.267,
        "0.005": 0.302,
    }
    for key, val in test_vals.items():
        if U2n > val:
            print(
                "The Watsons U2 statistic is {0} which is \
                greater than\n the critical value of {1} at p={2}".format(
                    U2n, val, key
                )
            )
        else:
            print(
                "The Watsons U2 statistic is not \
                significant at p={0}".format(
                    key
                )
            )
    return U2n


def watsonWilliams(a, b):
    """
    The Watson-Williams F test tests whether a set of mean directions are
    equal given that the concentrations are unknown, but equal, given that
    the groups each follow a von Mises distribution.

    Args:
        a, b (array_like): The directional samples

    Returns:
        F_stat (float): The F-statistic
    """

    n = len(a)
    m = len(b)
    N = n + m
    # v_1 = 1 # needed to do p-value lookup in table of critical values
    #  of F distribution
    # v_2 = N - 2 # needed to do p-value lookup in table of critical values
    # of F distribution
    C_1 = np.sum(np.cos(np.radians(a)))
    S_1 = np.sum(np.sin(np.radians(a)))
    C_2 = np.sum(np.cos(np.radians(b)))
    S_2 = np.sum(np.sin(np.radians(b)))
    C = C_1 + C_2
    S = S_1 + S_2
    R_1 = np.hypot(C_1, S_1)
    R_2 = np.hypot(C_2, S_2)
    R = np.hypot(C, S)
    R_hat = (R_1 + R_2) / float(N)
    from ephysiopy.common.mle_von_mises_vals import vals

    mle_von_mises = np.array(vals)
    mle_von_mises = np.sort(mle_von_mises, 0)
    k_hat = mle_von_mises[(np.abs(mle_von_mises[:, 0] - R_hat)).argmin(), 1]
    g = 1 - (3 / 8 * k_hat)
    F = g * (N - 2) * ((R_1 + R_2 - R) / (N - (R_1 + R_2)))
    return F


@dataclass
class CircStatsResults:
    """
    Dataclass to hold results from circular statistics
    """

    rho: float = np.nan
    p: float = np.nan
    rho_boot: float = np.nan
    p_shuffled: float = np.nan
    ci: float = np.nan
    slope = np.nan
    intercept = np.nan

    def __post_init__(self):
        if isinstance(self.ci, tuple):
            self.ci_lower, self.ci_upper = self.ci
        else:
            self.ci_lower = self.ci_upper = self.ci

        # ensure that p is a float
        if isinstance(self.p, np.ndarray):
            self.p = float(self.p)

        # ensure that p_shuffled is a float
        if isinstance(self.p_shuffled, np.ndarray):
            self.p_shuffled = float(self.p_shuffled)

    def __repr__(self):
        return (
            f"$\\rho$={self.rho:.3f}\np={self.p},\n"
            f"p_shuf={self.p_shuffled:.3f}\n"
            f"ci=({self.ci_lower:.3f}, {self.ci_upper:.3f})\n"
            f"slope={self.slope:.3f}\n"
            f"intercept={self.intercept:.3f}\n"
        )


@dataclass(frozen=True)
class RegressionResults:
    name: str
    phase: np.ndarray
    regressor: np.ndarray
    stats: CircStatsResults


def ccc(t, p):
    """
    Calculates correlation between two random circular variables

    Parameters
    ----------
    t : np.ndarray
        The first variable
    p : np.ndarray
        The second variable

    Returns
    -------
    float
        The correlation between the two variables
    """
    n = len(t)
    A = np.sum(np.cos(t) * np.cos(p))
    B = np.sum(np.sin(t) * np.sin(p))
    C = np.sum(np.cos(t) * np.sin(p))
    D = np.sum(np.sin(t) * np.cos(p))
    E = np.sum(np.cos(2 * t))
    F = np.sum(np.sin(2 * t))
    G = np.sum(np.cos(2 * p))
    H = np.sum(np.sin(2 * p))
    rho = 4 * (A * B - C * D) / \
        np.sqrt((n**2 - E**2 - F**2) * (n**2 - G**2 - H**2))
    return rho


def ccc_jack(t, p):
    """
    Function used to calculate jackknife estimates of correlation
    between two circular random variables

    Parameters
    ----------
    t : np.ndarray
        The first variable
    p : np.ndarray
        The second variable

    Returns
    -------
    np.ndarray
        The jackknife estimates of the correlation between the two variables
    """
    n = len(t) - 1
    A = np.cos(t) * np.cos(p)
    A = np.sum(A) - A
    B = np.sin(t) * np.sin(p)
    B = np.sum(B) - B
    C = np.cos(t) * np.sin(p)
    C = np.sum(C) - C
    D = np.sin(t) * np.cos(p)
    D = np.sum(D) - D
    E = np.cos(2 * t)
    E = np.sum(E) - E
    F = np.sin(2 * t)
    F = np.sum(F) - F
    G = np.cos(2 * p)
    G = np.sum(G) - G
    H = np.sin(2 * p)
    H = np.sum(H) - H
    rho = 4 * (A * B - C * D) / \
        np.sqrt((n**2 - E**2 - F**2) * (n**2 - G**2 - H**2))
    return rho


def circCircCorrTLinear(theta, phi, regressor=1000, alpha=0.05, hyp=0, conf=True):
    """
    An almost direct copy from AJs Matlab fcn to perform correlation
    between 2 circular random variables.

    Returns the correlation value (rho), p-value, bootstrapped correlation
    values, shuffled p values and correlation values.

    Parameters
    ----------
    theta, phi : np.ndarray
        The two circular variables to correlate (in radians)
    regressor : int, default=1000
        number of permutations to use to calculate p-value from randomisation
        and bootstrap estimation of confidence intervals.
        Leave empty to calculate p-value analytically (NB confidence
        intervals will not be calculated).
    alpha : float, default=0.05
        hypothesis test level e.g. 0.05, 0.01 etc.
    hyp : int, default=0
        hypothesis to test; -1/ 0 / 1 (-ve correlated / correlated in either
        direction / positively correlated).
    conf : bool, default=True
        True or False to calculate confidence intervals via
        jackknife or bootstrap.

    References
    ----------
    Fisher (1993), Statistical Analysis of Circular Data,
        Cambridge University Press, ISBN: 0 521 56890 0
    """
    theta = theta.ravel()
    phi = phi.ravel()

    if not len(theta) == len(phi):
        print("theta and phi not same length - try again!")
        raise ValueError()

    # estimate correlation
    rho = ccc(theta, phi)
    n = len(theta)

    # derive p-values
    if regressor:
        p_shuff = shuffledPVal(theta, phi, rho, regressor, hyp)
        p = np.nan

    # estimtate ci's for correlation
    if n >= 25 and conf:
        # obtain jackknife estimates of rho and its ci's
        rho_jack = ccc_jack(theta, phi)
        rho_jack = n * rho - (n - 1) * rho_jack
        rho_boot = np.mean(rho_jack)
        rho_jack_std = np.std(rho_jack)
        ci = (
            rho_boot - (1 / np.sqrt(n)) * rho_jack_std *
            norm.ppf(alpha / 2, (0, 1))[0],
            rho_boot + (1 / np.sqrt(n)) * rho_jack_std *
            norm.ppf(alpha / 2, (0, 1))[0],
        )
    elif conf and regressor and n < 25 and n > 4:
        from sklearn.utils import resample

        # set up the bootstrapping parameters
        boot_samples = []
        for i in range(regressor):
            theta_sample = resample(theta, replace=True)
            phi_sample = resample(phi, replace=True)
            boot_samples.append(
                ccc(
                    theta_sample[np.isfinite(theta_sample)],
                    phi_sample[np.isfinite(phi_sample)],
                )
            )
        rho_boot = np.nanmean(boot_samples)
        # confidence intervals
        p = ((1.0 - alpha) / 2.0) * 100
        lower = np.nanmax([0.0, np.nanpercentile(boot_samples, p)])
        p = (alpha + ((1.0 - alpha) / 2.0)) * 100
        upper = np.nanmin([1.0, np.nanpercentile(boot_samples, p)])

        ci = (lower, upper)
    else:
        rho_boot = np.nan
        ci = np.nan

    return CircStatsResults(rho, p, rho_boot, p_shuff, ci)


def shuffledPVal(theta, phi, rho, regressor, hyp):
    """
    Calculates shuffled p-values for correlation

    Parameters
    ----------
    theta, phi : np.ndarray
        The two circular variables to correlate (in radians)

    Returns
    -------
    float
        The shuffled p-value for the correlation between the two variables
    """
    n = len(theta)
    idx = np.zeros((n, regressor))
    for i in range(regressor):
        idx[:, i] = np.random.permutation(np.arange(n))

    thetaPerms = theta[idx.astype(int)]

    A = np.dot(np.cos(phi), np.cos(thetaPerms))
    B = np.dot(np.sin(phi), np.sin(thetaPerms))
    C = np.dot(np.sin(phi), np.cos(thetaPerms))
    D = np.dot(np.cos(phi), np.sin(thetaPerms))
    E = np.sum(np.cos(2 * theta))
    F = np.sum(np.sin(2 * theta))
    G = np.sum(np.cos(2 * phi))
    H = np.sum(np.sin(2 * phi))

    rho_sim = 4 * (A * B - C * D) / \
        np.sqrt((n**2 - E**2 - F**2) * (n**2 - G**2 - H**2))

    if hyp == 1:
        p_shuff = np.sum(rho_sim >= rho) / float(regressor)
    elif hyp == -1:
        p_shuff = np.sum(rho_sim <= rho) / float(regressor)
    elif hyp == 0:
        p_shuff = np.sum(np.fabs(rho_sim) > np.fabs(rho)) / float(regressor)
    else:
        p_shuff = np.nan

    return p_shuff


# TODO: Rarely the minimisation function fails due to
# some unbounded condition ValueError - I think this is
# due to bad input - e.g. x is all nan or something similar
def circRegress(x, t):
    """
    Finds approximation to circular-linear regression for phase precession.

    Parameters
    ----------
    x, t : np.ndarray
        The linear variable and the phase variable (in radians)

    Notes
    -----
    Neither x nor t can contain NaNs, must be paired (of equal length).
    """
    # transform the linear co-variate to the range -1 to 1
    if not np.any(x) or not np.any(t):
        return x, t
    mnx = np.mean(x)
    xn = x - mnx
    mxx = np.max(np.fabs(xn))
    xn = xn / mxx
    # keep tn between 0 and 2pi
    tn = np.remainder(t, 2 * np.pi)
    # constrain max slope to give at most 720 degrees of phase precession
    # over the field
    max_slope = (2 * np.pi) / (np.max(xn) - np.min(xn))

    # perform slope optimisation and find intercept
    def _cost(m, x, t):
        return -np.abs(np.sum(np.exp(1j * (t - m * x)))) / len(t - m * x)

    try:
        slope = optimize.fminbound(
            _cost, -1 * max_slope, max_slope, args=(xn, tn))
    except ValueError:
        return np.nan, np.nan
    intercept = np.arctan2(
        np.sum(np.sin(tn - slope * xn)), np.sum(np.cos(tn - slope * xn))
    )
    intercept = intercept + ((0 - slope) * (mnx / mxx))
    slope = slope / mxx
    return slope, intercept
