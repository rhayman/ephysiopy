import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from ephysiopy.common.binning import RateMap
from ephysiopy.common.ephys_generic import PosCalcsGeneric
from ephysiopy.common.rhythmicity import LFPOscillations
from ephysiopy.common.utils import bwperim
from ephysiopy.common.utils import count_runs_and_unique_numbers, flatten_list
from ephysiopy.visualise.plotting import stripAxes
from scipy import ndimage, optimize, signal
from scipy.stats import norm, circmean
import skimage
from collections import defaultdict


@stripAxes
def _stripAx(ax):
    return ax


jet_cmap = matplotlib.colormaps["jet"]

subaxis_title_fontsize = 10
cbar_fontsize = 8
cbar_tick_fontsize = 6


def labelledCumSum(X, L):
    # check if inputs are masked and save for masking
    # output and unmask the input
    x_mask = None
    if np.ma.is_masked(X):
        x_mask = X.mask
        X = X.data
    l_mask = None
    if np.ma.is_masked(L):
        l_mask = L.mask
        L = L.data
    orig_mask = np.logical_or(x_mask, l_mask)
    X = np.ravel(X)
    L = np.ravel(L)
    if len(X) != len(L):
        print("The two inputs need to be of the same length")
        return
    X[np.isnan(X)] = 0
    S = np.cumsum(X)

    mask = L.astype(bool)
    LL = L[:-1] != L[1::]
    LL = np.insert(LL, 0, True)
    isStart = np.logical_and(mask, LL)
    startInds = np.nonzero(isStart)[0]
    if len(startInds) == 0:
        return S
    if startInds[0] == 0:
        S_starts = S[startInds[1::] - 1]
        S_starts = np.insert(S_starts, 0, 0)
    else:
        S_starts = S[startInds - 1]

    L_safe = np.cumsum(isStart)
    S[mask] = S[mask] - S_starts[L_safe[mask] - 1]
    zero_label_idx = L == 0
    out_mask = np.logical_or(zero_label_idx, orig_mask)
    S = np.ma.MaskedArray(S, mask=out_mask)
    return S


def cart2pol(x, y):
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    return r, th


def pol2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def applyFilter2Labels(M, x):
    """
    M is a logical mask specifying which label numbers to keep
    x is an array of positive integer labels

    This method sets the undesired labels to 0 and renumbers the remaining
    labels 1 to n when n is the number of trues in M
    """
    newVals = M * np.cumsum(M)
    x[x > 0] = newVals[x[x > 0] - 1]
    return x


def getLabelStarts(x):
    x = np.ravel(x)
    xx = np.ones(len(x) + 1)
    xx[1::] = x
    xx = xx[:-1] != xx[1::]
    xx[0] = True
    return np.nonzero(np.logical_and(x, xx))[0]


def getLabelEnds(x):
    x = np.ravel(x)
    xx = np.ones(len(x) + 1)
    xx[:-1] = x
    xx = xx[:-1] != xx[1::]
    xx[-1] = True
    return np.nonzero(np.logical_and(x, xx))[0]


def circ_abs(x):
    return np.abs(np.mod(x + np.pi, 2 * np.pi) - np.pi)


def labelContigNonZeroRuns(x):
    x = np.ravel(x)
    xx = np.ones(len(x) + 1)
    xx[1::] = x
    xx = xx[:-1] != xx[1::]
    xx[0] = True
    L = np.cumsum(np.logical_and(x, xx))
    L[np.logical_not(x)] = 0
    return L


def getPhaseOfMinSpiking(spkPhase):
    kernelLen = 180
    kernelSig = kernelLen / 4

    k = signal.windows.gaussian(kernelLen, kernelSig)
    bins = np.arange(-179.5, 180, 1)
    phaseDist, _ = np.histogram(spkPhase / np.pi * 180, bins=bins)
    phaseDist = ndimage.convolve(phaseDist, k)
    phaseMin = bins[
        int(np.ceil(np.nanmean(np.nonzero(phaseDist == np.min(phaseDist))[0])))
    ]
    return phaseMin


def fixAngle(a):
    """
    Ensure angles lie between -pi and pi
    a must be in radians
    """
    b = np.mod(a + np.pi, 2 * np.pi) - np.pi
    return b


def ccc(t, p):
    """
    Calculates correlation between two random circular variables
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
    rho = 4 * (A * B - C * D) / np.sqrt((n**2 - E**2 - F**2) * (n**2 - G**2 - H**2))
    return rho


def ccc_jack(t, p):
    """
    Function used to calculate jackknife estimates of correlation
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
    rho = 4 * (A * B - C * D) / np.sqrt((n**2 - E**2 - F**2) * (n**2 - G**2 - H**2))
    return rho


def plot_spikes_in_runs_per_field(
    field_label: np.ndarray,
    run_starts: np.ndarray,
    run_ends: np.ndarray,
    spikes_in_time: np.ndarray,
    ttls_in_time: np.ndarray = None,
    **kwargs,
):
    """
    Debug plotting to show spikes per run per field found in the ratemap
    as a raster plot

    Args:
    field_label (np.ndarray): The field labels for each position bin
        a vector
    run_start_stop_idx (np.ndarray): The start and stop indices of each run
        has shape (n_runs, 2)
    spikes_in_time (np.ndarray): The number of spikes in each position bin
        a vector

    kwargs:
    separate_plots (bool): If True then each field will be plotted in a
    separate figure

    single_axes (bool): If True will plot all the runs/ spikes in a single
    axis with fields delimited by horizontal lines

    Returns:
    fig, axes (tuple): The figure and axes objects
    """
    spikes_in_time = np.ravel(spikes_in_time)
    assert len(spikes_in_time) == len(ttls_in_time)
    run_start_stop_idx = np.array([run_starts, run_ends]).T
    run_field_id = field_label[run_start_stop_idx[:, 0]]
    runs_per_field = np.histogram(run_field_id, bins=range(1, max(run_field_id) + 2))[0]
    max_run_len = np.max(run_start_stop_idx[:, 1] - run_start_stop_idx[:, 0])
    all_slices = np.array([slice(r[0], r[1]) for r in run_start_stop_idx])
    # create the figure window first then do the iteration through fields etc
    master_raster_arr = []
    # a grey colour for the background i.e. how long the run was
    grey = np.array([0.8627, 0.8627, 0.8627, 1])
    # iterate through each field then pull out the
    max_spikes = np.nanmax(spikes_in_time).astype(int) + 1
    orig_cmap = matplotlib.colormaps["spring"].resampled(max_spikes)
    cmap = orig_cmap(np.linspace(0, 1, max_spikes))
    cmap[0, :] = grey
    newcmap = ListedColormap(cmap)
    # some lists to hold the outputs
    # spike count for each run through the field
    master_raster_arr = []
    # list for count of total number of spikes per field
    spikes_per_run = []
    # counts of ttl puleses emitted during each run
    if ttls_in_time is not None:
        ttls_per_field = []
    # collect all the per field spiking, ttls etc first then plot
    # in a separate iteration
    for i, field_id in enumerate(np.unique(run_field_id)):
        # create a temporary array to hold the raster for this fields runs
        raster_arr = np.zeros(shape=(runs_per_field[i], max_run_len)) * np.nan
        ttl_arr = np.zeros(shape=(runs_per_field[i], max_run_len)) * np.nan
        # get the indices into the time binned spikes of the runs
        i_field_slices = all_slices[run_field_id == field_id]
        # breakpoint()
        for j, s in enumerate(i_field_slices):
            i_run_len = s.stop - s.start
            raster_arr[j, 0:i_run_len] = spikes_in_time[s]
            if ttls_in_time is not None:
                ttl_arr[j, 0:i_run_len] = ttls_in_time[s]
        spikes_per_run.append(int(np.nansum(raster_arr)))
        if ttls_in_time is not None:
            ttls_per_field.append(ttl_arr)
        master_raster_arr.append(raster_arr)

    if "separate_plots" in kwargs.keys():
        for i, field_id in enumerate(np.unique(run_field_id)):
            _, ax = plt.subplots(1, 1)
            ax.imshow(master_raster_arr[i], cmap=newcmap, aspect="auto")
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_ylabel(f"Field {field_id}")
    elif "single_axes" in kwargs.keys():
        # deal with master_raster_arr here
        _, ax = plt.subplots(1, 1)
        if ttls_in_time is not None:
            ttls = np.array(flatten_list(ttls_per_field))
            ax.imshow(ttls, cmap=matplotlib.colormaps["bone"])
        spiking_arr = np.array(flatten_list(master_raster_arr))
        ax.imshow(spiking_arr, cmap=newcmap, alpha=0.6)
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.hlines(np.cumsum(runs_per_field)[:-1], 0, max_run_len, "k")
        ax.set_xlim(0, max_run_len)
        ytick_locs = np.insert(np.cumsum(runs_per_field), 0, 0)
        ytick_locs = np.diff(ytick_locs) // 2 + ytick_locs[:-1]
        ax.set_yticks(ytick_locs, list(map(str, np.unique(run_field_id))))
        ax.set_ylabel("Field ID", rotation=90, labelpad=10)
        ax.set_xlabel("Time (s)")
        ax.set_xticks([0, max_run_len], ["0", f"{(max_run_len)/50:.2f}"])
        axes2 = ax.twinx()
        axes2.set_yticks(ytick_locs, list(map(str, spikes_per_run)))
        axes2.set_ylim(ax.get_ylim())
        axes2.set_ylabel("Spikes per field", rotation=270, labelpad=10)

    # else:
    #     axes[i].imshow(raster_arr, cmap=newcmap, aspect="auto")
    #     axes[i].axes.get_xaxis().set_ticks([])
    #     axes[i].axes.get_yaxis().set_ticks([])
    #     axes[i].set_ylabel(f"Field {field_id}")
    #


def circCircCorrTLinear(theta, phi, k=1000, alpha=0.05, hyp=0, conf=True):
    """
    An almost direct copy from AJs Matlab fcn to perform correlation
    between 2 circular random variables.

    Returns the correlation value (rho), p-value, bootstrapped correlation
    values, shuffled p values and correlation values.

    Args:
        theta, phi (array_like): mx1 array containing circular data (radians)
            whose correlation is to be measured
        k (int, optional): number of permutations to use to calculate p-value
            from randomisation and bootstrap estimation of confidence
            intervals.
            Leave empty to calculate p-value analytically (NB confidence
            intervals will not be calculated). Default is 1000.
        alpha (float, optional): hypothesis test level e.g. 0.05, 0.01 etc.
            Default is 0.05.
        hyp (int, optional): hypothesis to test; -1/ 0 / 1 (-ve correlated /
            correlated in either direction / positively correlated).
            Default is 0.
        conf (bool, optional): True or False to calculate confidence intervals
            via jackknife or bootstrap. Default is True.

    References:
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
    if k:
        p_shuff = shuffledPVal(theta, phi, rho, k, hyp)
        p = np.nan

    # estimtate ci's for correlation
    if n >= 25 and conf:
        # obtain jackknife estimates of rho and its ci's
        rho_jack = ccc_jack(theta, phi)
        rho_jack = n * rho - (n - 1) * rho_jack
        rho_boot = np.mean(rho_jack)
        rho_jack_std = np.std(rho_jack)
        ci = (
            rho_boot - (1 / np.sqrt(n)) * rho_jack_std * norm.ppf(alpha / 2, (0, 1))[0],
            rho_boot + (1 / np.sqrt(n)) * rho_jack_std * norm.ppf(alpha / 2, (0, 1))[0],
        )
    elif conf and k and n < 25 and n > 4:
        from sklearn.utils import resample

        # set up the bootstrapping parameters
        boot_samples = []
        for i in range(k):
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
        lower = np.nanmax(0.0, np.nanpercentile(boot_samples, p))
        p = (alpha + ((1.0 - alpha) / 2.0)) * 100
        upper = np.nanmin(1.0, np.nanpercentile(boot_samples, p))

        ci = (lower, upper)
    else:
        rho_boot = np.nan
        ci = np.nan

    return rho, p, rho_boot, p_shuff, ci


def shuffledPVal(theta, phi, rho, k, hyp):
    """
    Calculates shuffled p-values for correlation
    """
    n = len(theta)
    idx = np.zeros((n, k))
    for i in range(k):
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

    rho_sim = 4 * (A * B - C * D) / np.sqrt((n**2 - E**2 - F**2) * (n**2 - G**2 - H**2))

    if hyp == 1:
        p_shuff = np.sum(rho_sim >= rho) / float(k)
    elif hyp == -1:
        p_shuff = np.sum(rho_sim <= rho) / float(k)
    elif hyp == 0:
        p_shuff = np.sum(np.fabs(rho_sim) > np.fabs(rho)) / float(k)
    else:
        p_shuff = np.nan

    return p_shuff


def circRegress(x, t):
    """
    Finds approximation to circular-linear regression for phase precession.

    Args:
        x (list): n-by-1 list of in-field positions (linear variable)
        t (list): n-by-1 list of phases, in degrees (converted to radians)

    Note:
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

    slope = optimize.fminbound(_cost, -1 * max_slope, max_slope, args=(xn, tn))
    intercept = np.arctan2(
        np.sum(np.sin(tn - slope * xn)), np.sum(np.cos(tn - slope * xn))
    )
    intercept = intercept + ((0 - slope) * (mnx / mxx))
    slope = slope / mxx
    return slope, intercept


# There are a lot of parameters here so lets keep them outside the main
# class and define them as a module level dictionary
phase_precession_config = {
    "pos_sample_rate": 50,
    "lfp_sample_rate": 250,
    "cms_per_bin": 1,  # bin size gets calculated in Ratemap
    "ppm": 400,
    "field_smoothing_kernel_len": 51,
    "field_smoothing_kernel_sigma": 5,
    # fractional limit of field peak to restrict fields with
    "field_threshold": 10,
    # field threshold percent - fed into fieldcalcs.local_threshold as prc
    "field_threshold_percent": 50,
    # fractional limit for restricting fields size
    "area_threshold": 0.01,
    "bins_per_cm": 2,
    "convert_xy_2_cm": False,
    # defines start/ end of theta cycle
    "allowed_min_spike_phase": np.pi,
    # percentile power below which theta cycles are rejected
    "min_power_percent_threshold": 0,
    # theta bands for min / max cycle length
    "min_theta": 6,
    "max_theta": 12,
    # kernel length for smoothing speed (boxcar)
    "speed_smoothing_window_len": 15,
    # cm/s - original value = 2.5; lowered for mice
    "minimum_allowed_run_speed": 0.5,
    "minimum_allowed_run_duration": 2,  # in seconds
    # instantaneous firing rate (ifr) smoothing constant
    "ifr_smoothing_constant": 1.0 / 3,
    "spatial_lowpass_cutoff": 3,
    "ifr_kernel_len": 1,  # ifr smoothing kernal length
    "ifr_kernel_sigma": 0.5,
    "bins_per_second": 50,  # bins per second for ifr smoothing
}


all_regressors = [
    "spk_numWithinRun",
    "pos_exptdRate_cum",
    "pos_instFR",
    "eeg_instFR",  # my addition to get eeg sampled estimate of FR
    "pos_timeInRun",
    "pos_d_cum",
    "pos_d_meanDir",
    "pos_d_currentdir",
    "spk_thetaBatchLabelInRun",
]


class phasePrecession2D(object):
    """
    Performs phase precession analysis for single unit data

    Mostly a total rip-off of code written by Ali Jeewajee for his paper on
    2D phase precession in place and grid cells [1]_

    .. [1] Jeewajee A, Barry C, Douchamps V, Manson D, Lever C, Burgess N.
        Theta phase precession of grid and place cell firing in open
        environments.
        Philos Trans R Soc Lond B Biol Sci. 2013 Dec 23;369(1635):20120532.
        doi: 10.1098/rstb.2012.0532.

    Args:
        lfp_sig (np.array): The LFP signal against which cells might precess...
        lfp_fs (int): The sampling frequency of the LFP signal
        xy (np.array): The position data as 2 x num_position_samples
        spike_ts (np.array): The times in samples at which the cell fired
        pos_ts (np.array): The times in samples at which position was captured
        pp_config (dict): Contains parameters for running the analysis.
            See phase_precession_config dict in ephysiopy.common.eegcalcs
    """

    def __init__(
        self,
        lfp_sig: np.ndarray,
        lfp_fs: int,
        xy: np.ndarray,
        spike_ts: np.ndarray,
        pos_ts: np.ndarray,
        pp_config: dict = phase_precession_config,
        regressors=None,
    ):
        # Set up the parameters
        # this sets a bunch of member attributes from the pp_config dict
        self.update_config(pp_config)
        self._pos_ts = pos_ts

        self.update_regressors(regressors)

        self.k = 1000
        self.alpha = 0.05
        self.hyp = 0
        self.conf = True

        # Process the EEG data a bit...
        self.eeg = lfp_sig
        L = LFPOscillations(lfp_sig, lfp_fs)
        filt_sig, phase, _, _ = L.getFreqPhase(lfp_sig, [6, 12], 2)
        self.filteredEEG = filt_sig
        self.phase = phase
        self.phaseAdj = np.ma.MaskedArray

        self.update_position(xy, self.ppm, cm=self.convert_xy_2_cm)
        self.update_rate_map()

        spk_times_in_pos_samples = self.getSpikePosIndices(spike_ts)
        spk_weights = np.bincount(spk_times_in_pos_samples, minlength=len(self.pos_ts))
        self.spike_times_in_pos_samples = spk_times_in_pos_samples
        self.spk_weights = spk_weights

        self.spike_ts = spike_ts

    @property
    def pos_ts(self):
        return self._pos_ts

    @pos_ts.setter
    def pos_ts(self, value):
        self._pos_ts = value

    @property
    def xy(self):
        return self.PosData.xy

    @xy.setter
    def xy(self, value):
        self.PosData.xy = value

    def update_regressors(self, reg_keys: list):
        """
        Create a dict to hold the stats values for
        each regressor
        Default regressors are:
            "spk_numWithinRun",
            "pos_exptdRate_cum",
            "pos_instFR",
            "eeg_instFR",
            "pos_timeInRun",
            "pos_d_cum",
            "pos_d_meanDir",
            "pos_d_currentdir",
            "spk_thetaBatchLabelInRun"

        NB: The regressors have differing sizes of 'values' depending on the
        type of the regressor:
        spk_* - integer values of the spike number within a run or the theta batch
                in a run, so has a length equal to the number of spikes collected
        pos_* - a bincount of some type so equal to the number of position samples
                collected
        eeg_* - only one at present, the instantaneous firing rate binned into the
                number of eeg samples so equal to that in length
        """
        if reg_keys is None:
            reg_keys = all_regressors
        else:
            assert all([k in all_regressors for k in reg_keys])

        # Create a dict to hold the stats values for
        # each regressor
        stats_dict = {
            "values": np.ma.MaskedArray,
            "pha": np.ma.MaskedArray,
            "slope": float,
            "intercept": float,
            "cor": float,
            "p": float,
            "cor_boot": float,
            "p_shuffled": float,
            "ci": float,
            "reg": float,
        }
        self.regressors = {}
        self.regressors = defaultdict(lambda: stats_dict.copy(), self.regressors)
        [self.regressors[k] for k in reg_keys]
        # each of the regressors in regressor_keys is a key with a value
        # of stats_dict

    def update_regressor_values(self, key: str, values):
        # Check whether values is a masked array and if not make it one
        self.regressors[key]["values"] = values

    def update_regressor_mask(self, key: str, indices):
        # Mask entries in the 'values' and 'pha' arrays of the relevant regressor
        self.regressors[key]["values"].mask[indices] = False

    def get_regressors(self):
        return self.regressors.keys()

    def get_regressor(self, key):
        return self.regressors[key]

    def update_config(self, pp_config):
        [setattr(self, k, pp_config[k]) for k in pp_config.keys()]

    def update_position(self, xy, ppm: float, cm: bool):
        P = PosCalcsGeneric(
            xy[0, :],
            xy[1, :],
            ppm=ppm,
            convert2cm=cm,
        )
        P.postprocesspos(tracker_params={"AxonaBadValue": 1023})
        # ... do the ratemap creation here once
        self.PosData = P

    def update_rate_map(self):
        R = RateMap(self.PosData, xyInCms=self.convert_xy_2_cm)
        R.binsize = self.cms_per_bin
        R.smooth_sz = self.field_smoothing_kernel_len
        R.ppm = self.ppm
        self.RateMap = R  # this will be used a fair bit below

    def getSpikePosIndices(self, spk_times: np.array):
        pos_times = getattr(self, "pos_ts")
        idx = np.searchsorted(pos_times, spk_times)
        idx[idx == len(pos_times)] = idx[idx == len(pos_times)] - 1
        return idx

    def performRegression(self, laserEvents=None, **kwargs):
        """
        Wrapper function for doing the actual regression which has multiple
        stages.

        Specifically here we partition fields into sub-fields, get a bunch of
        information about the position, spiking and theta data and then
        do the actual regression.

        Args:
            tetrode (int): The tetrode to examine
            cluster (int): The cluster to examine
            laserEvents (array_like, optional): The on times for laser events
            if present. Default is None
        Valid keyword args:
            plot (bool): whether to plot the results of field partitions, the regression(s)
                etc
        See Also:
            ephysiopy.common.eegcalcs.phasePrecession.partitionFields()
            ephysiopy.common.eegcalcs.phasePrecession.getPosProps()
            ephysiopy.common.eegcalcs.phasePrecession.getThetaProps()
            ephysiopy.common.eegcalcs.phasePrecession.getSpikeProps()
            ephysiopy.common.eegcalcs.phasePrecession._ppRegress()
        """
        do_plot = kwargs.get("plot", False)

        # Partition fields
        peaksXY, _, labels, _ = self.partitionFields()

        # split into runs
        posD, runD = self.getPosProps(
            labels, peaksXY, laserEvents=laserEvents, plot=do_plot
        )
        self.posdict = posD
        self.rundict = runD

        # get theta cycles, amplitudes, phase etc
        self.getThetaProps()
        # get the indices of spikes for various metrics such as
        # theta cycle, run etc
        spkD = self.getSpikeProps(
            posD["runLabel"], runD["meanDir"], runD["runDurationInPosBins"]
        )
        self.spkdict = spkD
        # at this point the 'values' and 'pha' arrays in the regressors dict are all
        # npos elements long and are masked arrays. keep as masked arrays and just modify the
        # masks instead of truncating the data
        # Do the regressions
        self._ppRegress(spkD)

        # Plot the results if asked
        if do_plot:
            n_regressors = len(self.get_regressors())
            n_rows = np.ceil(n_regressors / 2).astype(int)
            if n_regressors == 1:
                fig, ax = plt.subplots(1, 1, figsize=(3, 5))
            else:
                fig, ax = plt.subplots(2, n_rows, figsize=(10, 10))
            fig.canvas.manager.set_window_title("Regression results")
            if isinstance(ax, list | np.ndarray):
                ax = flatten_list(ax)
            if n_regressors == 1:
                ax = [ax]
            for ra in zip(self.get_regressors(), ax):
                self.plotRegressor(ra[0], ra[1])

    def partitionFields(self, plot: bool = False) -> tuple:
        """
        Partitions fields.

        Partitions spikes into fields by finding the watersheds around the
        peaks of a super-smoothed ratemap

        Args:
            spike_ts (np.array): The ratemap to partition
            ftype (str): 'p' or 'g' denoting place or grid cells
              - not implemented yet
            plot (bool): Whether to produce a debugging plot or not

        Returns:
            peaksXY (array_like): The xy coordinates of the peak rates in
            each field
            peaksRate (array_like): The peak rates in peaksXY
            labels (numpy.ndarray): An array of the labels corresponding to
            each field (starting at 1)
            rmap (numpy.ndarray): The ratemap of the tetrode / cluster
        """
        rmap = self.RateMap.get_map(self.spk_weights)
        ye, xe = rmap.bin_edges
        rmap = rmap.binned_data[0]
        nan_idx = np.isnan(rmap)
        rmap[nan_idx] = 0
        # start image processing:
        # get some markers
        from ephysiopy.common import fieldcalcs

        markers = fieldcalcs.local_threshold(rmap, prc=self.field_threshold_percent)
        # clear the edges / any invalid positions again
        markers[nan_idx] = 0
        # label these markers so each blob has a unique id
        labels = ndimage.label(markers)[0]
        # labels is now a labelled int array from 0 to however many fields have
        # been detected
        # get the number of spikes in each field - NB this is done against a
        # flattened array so we need to figure out which count corresponds to
        # which particular field id using np.unique
        fieldId, _ = np.unique(labels, return_index=True)
        fieldId = fieldId[1::]
        # TODO: come back to this as may need to know field id ordering
        peakCoords = np.array(
            ndimage.maximum_position(rmap, labels=labels, index=fieldId)
        ).astype(int)
        # breakpoint()

        peaksXY = np.vstack((xe[peakCoords[:, 1]], ye[peakCoords[:, 0]]))

        # find the peak rate at each of the centre of the detected fields to
        # subsequently threshold the field at some fraction of the peak value
        # use a labeled_comprehension to do this
        def fn(val, pos):
            return pos[val < (np.max(val) * (self.field_threshold / 100))]

        indices = ndimage.labeled_comprehension(
            rmap, labels, None, fn, np.ndarray, 0, True
        )
        labels[np.unravel_index(indices, labels.shape)] = 0
        # as a result of this some of the labeled areas will have been removed
        # and /or made smaller so we should fill in the holes,
        # remove any that are too small and re-label
        labels, n_labels = ndimage.label(ndimage.binary_fill_holes(labels))
        min_field_size = np.ceil(np.prod(labels.shape) * self.area_threshold).astype(
            int
        )
        # breakpoint()
        labels = skimage.morphology.remove_small_objects(
            labels, min_size=min_field_size, connectivity=2
        )
        # relable the fields
        labels = skimage.segmentation.relabel_sequential(labels)[0]

        # re-calculate the peakCoords array as we may have removed some
        # objects
        fieldId, _ = np.unique(labels, return_index=True)
        fieldId = fieldId[1::]
        peakCoords = np.array(
            ndimage.maximum_position(rmap, labels=labels, index=fieldId)
        ).astype(int)
        peaksXY = np.vstack((xe[peakCoords[:, 1]], ye[peakCoords[:, 0]]))
        peakRates = rmap[peakCoords[:, 0], peakCoords[:, 1]]
        peakLabels = labels[peakCoords[:, 0], peakCoords[:, 1]]
        peaksXY = peaksXY[:, peakLabels - 1]
        peaksRate = peakRates[peakLabels - 1]
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(211)
            rmapM = np.ma.masked_where(rmap == 0, rmap)
            ax.pcolormesh(
                xe, ye, rmapM, cmap=matplotlib.colormaps["jet"], edgecolors="face"
            )
            ax.set_title("Smoothed ratemap + peaks", fontsize=subaxis_title_fontsize)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set_aspect("equal")
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.plot(peaksXY[0], peaksXY[1], "ko")
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)

            ax = fig.add_subplot(212)
            labelsM = np.ma.masked_where(labels == 0, labels)
            ax.pcolormesh(xe, ye, labelsM, edgecolors="face")
            ax.plot(peaksXY[0], peaksXY[1], "ko")
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)

            ax.set_title("Labelled restricted fields", fontsize=subaxis_title_fontsize)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set_aspect("equal")

        return peaksXY, peaksRate, labels, rmap

    def getPosProps(
        self,
        labels: np.ndarray,
        peaksXY: np.ndarray,
        laserEvents: np.ndarray = None,
        plot: bool = False,
        **kwargs,
    ) -> dict:
        """
        Uses the output of partitionFields and returns vectors the same
        length as pos.

        Args:
            tetrode, cluster (int): The tetrode / cluster to examine
            peaksXY (array_like): The x-y coords of the peaks in the ratemap
            laserEvents (array_like): The position indices of on events
            (laser on)

        Returns:
            pos_dict, run_dict (dict): Contains a whole bunch of information
            for the whole trial and also on a run-by-run basis (run_dict).
            See the end of this function for all the key / value pairs.
        """

        spikeTS = self.spike_ts  # in seconds
        xy = self.RateMap.xy
        xydir = self.RateMap.dir
        spd = self.RateMap.speed
        spkPosInd = np.ceil(spikeTS * self.pos_sample_rate).astype(int)
        spkEEGInd = np.ceil(spikeTS * self.lfp_sample_rate).astype(int)
        nPos = xy.shape[1]
        spkPosInd[spkPosInd > nPos] = nPos - 1
        xy_old = xy.copy()
        xydir = np.squeeze(xydir)

        rmap = self.RateMap.get_map(self.spk_weights)
        ye, xe = rmap.bin_edges
        rmap = rmap.binned_data[0]
        """
        Lets remap the xy coordinates to lie between -1 and +1
        which should make things easier later when dealing with
        the unit circle
        """

        # The large number of bins combined with the super-smoothed ratemap
        # will lead to fields labelled with lots of small holes in. Fill those
        # gaps in here and calculate the perimeter of the fields based on that
        # labelled image
        labels, n_labels = ndimage.label(ndimage.binary_fill_holes(labels))

        rmap_zeros = rmap.copy()
        rmap_zeros[np.isnan(rmap)] = 0
        xBins = np.digitize(xy[0], xe[:-1])
        yBins = np.digitize(xy[1], ye[:-1])
        fieldLabel = labels[yBins - 1, xBins - 1]

        fieldPerimMask = bwperim(labels)

        peaksYXBins = np.array(
            ndimage.maximum_position(
                rmap_zeros, labels=labels, index=np.unique(labels)[1::]
            )
        ).astype(int)

        # define a couple of functions to add to skimage.measure.regionprops
        # 1) return the perimeter of the region as an array of bool
        def bw_perim(mask):
            return bwperim(mask)

        # 2) return the maximum index of the intensity image for the region as a tuple
        # NB there is a lot of overlap between the call to regionprops here and
        # the fieldcalcs.local_threshold function - that function uses many of the
        # same functions used in the surrounding code here and could be refactored
        def max_index(mask, intensity):
            return np.array(
                np.unravel_index(np.argmax(intensity, axis=None), intensity.shape)
            )

        field_props = skimage.measure.regionprops(
            labels, rmap_zeros, extra_properties=(max_index, bw_perim)
        )
        # some arrays to hold results
        perim_angle_from_peak = np.zeros_like(rmap) * np.nan
        perim_dist_from_peak = np.zeros_like(rmap) * np.nan
        pos_angle_from_peak = np.zeros((nPos)) * np.nan
        pos_dist_from_peak = np.zeros((nPos)) * np.nan
        pos_r_unsmoothed = np.zeros((nPos)) * np.nan

        for field in field_props:
            field_id = field.label
            i_xy = xy[:, fieldLabel == field_id]
            field.xy = i_xy
            if np.any(i_xy):
                # get the distances and angles of each point on the perimeter to the field peak
                perim_coords = np.nonzero(field.bw_perim)
                field.perim_coords = perim_coords
                perim_minus_field_max = (
                    perim_coords[0] - field.max_index[0],
                    perim_coords[1] - field.max_index[1],
                )

                i_perim_angle = np.arctan2(
                    perim_minus_field_max[0], perim_minus_field_max[1]
                )
                i_perim_dist = np.hypot(
                    perim_minus_field_max[0], perim_minus_field_max[1]
                )
                perim_angle_from_peak[fieldPerimMask == field_id] = i_perim_angle
                perim_dist_from_peak[fieldPerimMask == field_id] = i_perim_dist

                # get the distances and angles of each position coordinate within the field to the field peak
                xy_minus_field_max = (
                    i_xy[0] - xe[peaksYXBins[field_id - 1, 1]],
                    i_xy[1] - ye[peaksYXBins[field_id - 1, 0]],
                )
                i_pos_angle = np.arctan2(xy_minus_field_max[1], xy_minus_field_max[0])
                i_pos_dist = np.hypot(xy_minus_field_max[0], xy_minus_field_max[1])
                pos_angle_from_peak[fieldLabel == field_id] = i_pos_angle
                pos_dist_from_peak[fieldLabel == field_id] = i_pos_dist

                i_angle_df = circ_abs(
                    i_perim_angle[:, np.newaxis] - i_pos_angle[np.newaxis, :]
                )
                i_perim_idx = np.argmin(i_angle_df, 0)
                tmp = (
                    perim_coords[1][i_perim_idx] - field.max_index[1],
                    perim_coords[0][i_perim_idx] - field.max_index[0],
                )
                i_perim_dist_to_peak = np.hypot(tmp[0], tmp[1])
                # calculate the ratio of the distance from the field peak to the position sample
                # and the distance from the field peak to the point on the perimeter that is most
                # colinear with the position sample
                pos_r_unsmoothed[fieldLabel == field_id] = (
                    i_pos_dist / i_perim_dist_to_peak
                )

        # the skimage find_boundaries method combined with the labelled mask
        # strive to make some of the values in thisDistFromPos2Peak larger than
        # those in thisDistFromPerim2Peak which means that some of the vals in
        # posRUnsmthd larger than 1 which means the values in xy_new later are
        # wrong - so lets cap any value > 1 to 1. The same cap is applied later
        # to rho when calculating the angular values. Print out a warning
        # message letting the user know how many values have been capped
        print(
            "\n\n{:.2%} posRUnsmthd values have been capped to 1\n\n".format(
                np.sum(pos_r_unsmoothed >= 1) / pos_r_unsmoothed.size
            )
        )
        runs_count, _ = count_runs_and_unique_numbers(fieldLabel)
        for k in runs_count.keys():
            if k != 0:
                print(f"Field {k} has {runs_count[k]} potential runs through it")
        pos_r_unsmoothed[pos_r_unsmoothed >= 1] = 1
        # label non-zero contiguous runs with a unique id
        runLabel = labelContigNonZeroRuns(fieldLabel)
        isRun = runLabel > 0
        runStartIdx = getLabelStarts(runLabel)
        runEndIdx = getLabelEnds(runLabel)
        # find runs that are too short, have low speed or too few spikes
        no_spike_runs = np.ones(len(runStartIdx), dtype=bool)
        spkRunLabels = runLabel[spkPosInd] - 1
        no_spike_runs[spkRunLabels[spkRunLabels > 0]] = False
        runDurationInPosBins = runEndIdx - runStartIdx + 1
        runsMinSpeed = []
        runId = np.unique(runLabel)[1::]
        for run in runId:
            runsMinSpeed.append(np.min(spd[runLabel == run]))
        runsMinSpeed = np.array(runsMinSpeed)
        slow_runs = runsMinSpeed < self.minimum_allowed_run_speed
        short_runs = runDurationInPosBins < self.minimum_allowed_run_duration
        badRuns = np.logical_or(
            np.logical_or(slow_runs, short_runs),
            no_spike_runs,
        )
        # output some info about the runs that are being removed
        field_slow_runs = fieldLabel[runStartIdx][slow_runs]
        field_short_runs = fieldLabel[runStartIdx][short_runs]
        field_no_spike_runs = fieldLabel[runStartIdx][no_spike_runs]

        def print_lost_runs(runs, run_type):
            counts, field_ids = np.histogram(runs, bins=np.unique(fieldLabel) + 1)
            print("\n")
            print(f"Runs lost due to {run_type}:")
            for i, field in enumerate(list(field_ids[:-1])):
                print(f"Field {field} has lost {counts[i]} runs")

        print_lost_runs(field_slow_runs, "slow speed")
        print_lost_runs(field_short_runs, "short duration")
        print_lost_runs(field_no_spike_runs, "no spikes")
        badRuns = np.squeeze(badRuns)
        runLabel = np.ma.MaskedArray(applyFilter2Labels(~badRuns, runLabel))
        runStartIdx = runStartIdx[~badRuns]
        runEndIdx = runEndIdx[~badRuns]  # + 1
        runsMinSpeed = runsMinSpeed[~badRuns]
        runDurationInPosBins = runDurationInPosBins[~badRuns]
        isRun = runLabel > 0

        # output how many runs are left after filtering
        print(f"\n\n{len(runStartIdx)} total runs left after filtering\n\n")
        counts, field_ids = np.histogram(
            fieldLabel[runStartIdx], bins=np.unique(fieldLabel) + 1
        )
        spikes_in_time = np.bincount(
            (spikeTS * self.pos_sample_rate).astype(int), minlength=nPos
        )
        spikes_per_run = [
            sum(spikes_in_time[run[0] : run[1]]) for run in zip(runStartIdx, runEndIdx)
        ]
        # breakpoint()
        for i, c in enumerate(counts):
            print(f"Field {field_ids[i]} has {c} runs through it")

        # for each of the fields extracted above using regionprops
        # add each run through the field to the field object as a list
        # of runs
        field_runs_xy = {k: [] for k in np.unique(fieldLabel)}
        for run in zip(runStartIdx, runEndIdx):
            field_id = fieldLabel[run[0]]
            this_run = xy[:, run[0] : run[1]]
            field_runs_xy[field_id].append(this_run)
        # add this list of runs to the field_props list
        for field in field_props:
            id = field.label
            field.xy_runs = field_runs_xy[id]

        # calculate mean direction for each run
        meanDir = np.array(
            [circmean(np.deg2rad(xydir)[runLabel == i]) for i in np.unique(runLabel)]
        )

        # caculate angular distance between the runs main direction and the
        # pos's direction to the peak centre
        pos_phi_unsmoothed = np.ones_like(fieldLabel) * np.nan
        pos_phi_unsmoothed[isRun] = (
            pos_angle_from_peak[isRun] - meanDir[runLabel[isRun] - 1]
        )

        # smooth r and phi in cartesian space
        # convert to cartesian coords first
        pos_x_unsmoothed, pos_y_unsmoothed = pol2cart(
            pos_r_unsmoothed, pos_phi_unsmoothed
        )
        pos_xy_unsmoothed = np.vstack((pos_x_unsmoothed, pos_y_unsmoothed))

        filtLen = np.squeeze(
            np.floor((runEndIdx - runStartIdx + 1) * self.ifr_smoothing_constant)
        )
        xy_new = np.zeros_like(xy_old) * np.nan
        for i in range(len(runStartIdx)):
            if filtLen[i] > 2:
                filt = signal.firwin(
                    int(filtLen[i] - 1),
                    cutoff=self.spatial_lowpass_cutoff / self.pos_sample_rate * 2,
                    window="blackman",
                )
                xy_new[:, runStartIdx[i] : runEndIdx[i]] = signal.filtfilt(
                    filt,
                    [1],
                    pos_xy_unsmoothed[:, runStartIdx[i] : runEndIdx[i]],
                    axis=1,
                )
        rho, phi = cart2pol(xy_new[0], xy_new[1])
        rho[rho > 1] = 1

        # calculate the direction of the smoothed data
        xydir_new = np.arctan2(np.diff(xy_new[1]), np.diff(xy_new[0]))
        xydir_new = np.append(xydir_new, xydir_new[-1])
        xydir_new[runEndIdx] = xydir_new[runEndIdx - 1]

        # for each of the fields extracted above using regionprops
        # add each run through the field to the field object as a list
        # of runs
        field_runs_xy = {k: [] for k in np.unique(fieldLabel)}
        field_runs_rho_phi = {k: [] for k in np.unique(fieldLabel)}
        for run in zip(runStartIdx, runEndIdx):
            field_id = fieldLabel[run[0]]
            this_run = xy_new[:, run[0] : run[1]]
            field_runs_xy[field_id].append(this_run)
            this_run_rho_phi = np.vstack((rho[run[0] : run[1]], phi[run[0] : run[1]]))
            field_runs_rho_phi[field_id].append(this_run_rho_phi)
        # add this list of runs to the field_props list
        for field in field_props:
            id = field.label
            field.xy_runs = field_runs_xy[id]
            field.rho_phi_runs = field_runs_rho_phi[id]

        # project the distance value onto the current direction
        # GOOD
        if "pos_d_currentdir" in self.regressors.keys():
            d_currentdir = rho * np.cos(xydir_new - phi)
            self.update_regressor_values("pos_d_currentdir", d_currentdir)

        # calculate the cumulative distance travelled on each run
        # only goes from 0-1
        if "pos_d_cum" in self.regressors.keys():
            dr = np.sqrt(np.diff(np.power(rho, 2), 1))
            d_cumulative = labelledCumSum(np.insert(dr, 0, 0), runLabel)
            self.update_regressor_values("pos_d_cum", d_cumulative)

        # calculate cumulative sum of the expected normalised firing rate
        # only goes from 0-1
        if "pos_exptdRate_cum" in self.regressors.keys():
            # breakpoint()
            exptdRate_cumulative = labelledCumSum(1 - rho, runLabel)
            self.update_regressor_values("pos_exptdRate_cum", exptdRate_cumulative)

        # direction projected onto the run mean direction is just the x coord
        # good - remembering that xy_new is rho,phi
        if "pos_d_meanDir" in self.regressors.keys():
            d_meandir = xy_new[0]
            self.update_regressor_values("pos_d_meanDir", d_meandir)

        # smooth binned spikes to get an instantaneous firing rate
        # set up the smoothing kernel
        # all up at 1.0
        if "pos_instFR" in self.regressors.keys():
            kernLenInBins = np.round(self.ifr_kernel_len * self.bins_per_second)
            kernSig = self.ifr_kernel_sigma * self.bins_per_second
            k = signal.windows.gaussian(kernLenInBins, kernSig)
            # get a count of spikes to smooth over
            spkCount = np.bincount(spkPosInd, minlength=self.PosData.npos)
            # apply the smoothing kernel
            instFiringRate = signal.convolve(spkCount, k, mode="same")
            instFiringRate = np.ma.MaskedArray(instFiringRate, mask=~isRun)
            self.update_regressor_values("pos_instFR", instFiringRate)

        if "eeg_instFR" in self.regressors.keys():
            kernLenInBins = np.round(self.ifr_kernel_len * self.bins_per_second)
            kernSig = self.ifr_kernel_sigma * self.bins_per_second
            k = signal.windows.gaussian(kernLenInBins, kernSig)
            # get a count of spikes to smooth over
            spkCount = np.bincount(spkEEGInd, minlength=len(self.phase))
            # apply the smoothing kernel
            instFiringRate = signal.convolve(spkCount, k, mode="same")
            isRunEEG = np.repeat(
                isRun, int(self.lfp_sample_rate / self.pos_sample_rate)
            )
            instFiringRate = np.ma.MaskedArray(instFiringRate, mask=~isRunEEG)
            self.update_regressor_values("eeg_instFR", instFiringRate)

        # find time spent within run
        # only goes from 0-1
        if "pos_timeInRun" in self.regressors.keys():
            time = np.ones(nPos)
            time = labelledCumSum(time, runLabel)
            timeInRun = time / self.pos_sample_rate
            self.update_regressor_values("pos_timeInRun", timeInRun)
        fieldNum = fieldLabel[runStartIdx]
        mnSpd = np.squeeze(np.zeros_like(fieldNum, dtype=float))
        np.add.at(mnSpd, runLabel[isRun] - 1, spd[isRun])
        nPts = np.bincount(runLabel[isRun] - 1, minlength=len(mnSpd))
        np.divide.at(mnSpd, np.arange(len(mnSpd)), nPts)
        centralPeripheral = np.squeeze(np.zeros_like(fieldNum, dtype=float))
        np.add.at(centralPeripheral, runLabel[isRun] - 1, xy_new[1, isRun])
        np.divide.at(centralPeripheral, np.arange(len(nPts)), nPts)
        if plot:
            # FIGURE LEVEL PREPARATION
            fig = plt.figure()
            ax = fig.add_subplot(221)
            fig.canvas.manager.set_window_title("Field partitioning and runs")
            # get the outline of the arena for plotting
            # NB this is really just the outline of the area the animal
            # covered during the session
            # breakpoint()
            outline = np.isfinite(rmap)
            outline = ndimage.binary_fill_holes(outline)
            outline = bwperim(outline)
            outline = np.ma.masked_where(~outline, outline)

            # PLOT 1) the xy data with the runs through the fields
            cmap = matplotlib.colormaps["Set1"].resampled(np.max(fieldLabel))
            for field in field_props:
                for run in field.xy_runs:
                    ax.plot(run[0], run[1], color=cmap(field.label - 1))
            ax.set_title("Unit circle x-y", fontsize=subaxis_title_fontsize)
            ax.set_aspect("equal")
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            _stripAx(ax)

            # PLOT 2) the field perimeters and the peak locations coloured by field
            ax1 = fig.add_subplot(222)
            # add the outline of the arena
            ax1.pcolormesh(xe, ye, outline)
            fieldPerimMask_m = np.ma.MaskedArray(
                fieldPerimMask, mask=fieldPerimMask == 0
            )
            fpm = ax1.pcolormesh(xe, ye, fieldPerimMask_m, cmap=cmap, edgecolors="face")
            for field in field_props:
                ax1.plot(
                    xe[peaksYXBins[field.label - 1, 1]],
                    ye[peaksYXBins[field.label - 1, 0]],
                    marker="o",
                    color=cmap(field.label - 1),
                )
            cbar = plt.colorbar(fpm)
            cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
            cbar.ax.set_yticks(
                np.linspace(1.5, np.max(fieldLabel) - 0.5, np.max(fieldLabel))
            )
            cbar.ax.set_yticklabels(list(map(str, np.unique(fieldLabel)[1::])))
            cbar.ax.set_ylabel(
                "Field id", rotation=-90, va="bottom", size=cbar_fontsize
            )
            ax1.set_ylim(np.min(ye), np.max(ye))
            ax1.set_xlim(np.min(xe), np.max(xe))
            ax1.set_title(
                "Field perim and\n laser on events", fontsize=subaxis_title_fontsize
            )
            # if laserEvents is not None:
            # validOns = np.setdiff1d(
            #     laserEvents, np.nonzero(~np.isnan(r))[0])
            # ax1.plot(xy[0, validOns], xy[1, validOns], "rx")
            ax1.set_aspect("equal")
            _stripAx(ax1)

            # PLOT 3) the runs through the fields coloured by angle and distance
            angleCMInd = np.round(perim_angle_from_peak / np.pi * 180) + 180
            angleCMInd[angleCMInd == 0] = 360
            im = np.zeros_like(fieldPerimMask)
            fl_counts, fl_bins = np.histogram(fieldLabel, bins=np.unique(labels) + 1)
            for fl in fl_bins:
                xi, yi = np.nonzero(fieldPerimMask == fl)
                im[xi, yi] = angleCMInd[xi, yi]
            imM = np.ma.MaskedArray(im, mask=fieldPerimMask == 0, copy=True)
            # create custom colormap
            cmap = matplotlib.colormaps["jet_r"]
            # add the runs through the fields
            runVals = np.zeros_like(rmap)
            runVals[yBins[isRun] - 1, xBins[isRun] - 1] = rho[isRun]
            runVals = np.ma.masked_where(runVals == 0, runVals)
            ax = fig.add_subplot(223)
            ax.pcolormesh(xe, ye, outline)
            imm = ax.pcolormesh(
                xe,
                ye,
                runVals,
                cmap=cmap,
                edgecolors="face",
                shading="auto",
            )
            cbar = plt.colorbar(imm, orientation="horizontal")
            cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
            cbar.ax.set_xlabel(
                "Normalised distance to field centre",
                rotation=0,
                ha="center",
                size=cbar_fontsize,
            )
            ax.set_aspect("equal")

            # a cyclic colormap for the angular values
            cmap = matplotlib.colormaps["hsv"]

            imm = ax.pcolormesh(
                xe, ye, imM, cmap=cmap, edgecolors="face", shading="auto"
            )
            cbar = plt.colorbar(imm)
            cbar.ax.set_ylabel(
                "Angle to field centre", rotation=-90, va="bottom", size=cbar_fontsize
            )
            cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
            ax.set_title("Runs by distance and angle", fontsize=subaxis_title_fontsize)
            ax.set_ylim(np.min(ye), np.max(ye))
            ax.set_xlim(np.min(xe), np.max(xe))
            _stripAx(ax)

            # PLOT 4) the smoothed ratemap
            ax = fig.add_subplot(224)
            ax.pcolormesh(xe, ye, outline)
            vmax = np.nanmax(np.ravel(rmap))
            rmapM = np.ma.masked_where(rmap == 0, rmap)
            ax.pcolormesh(
                xe,
                ye,
                rmapM,
                cmap=jet_cmap,
                edgecolors="face",
                shading="auto",
                vmax=vmax,
            )
            for field in field_props:
                ax.plot(
                    xe[peaksYXBins[field.label - 1, 1]],
                    ye[peaksYXBins[field.label - 1, 0]],
                    marker="o",
                    color="k",
                )
            ax.set_xlim(np.min(xe), np.max(xe))
            ax.set_ylim(np.min(ye), np.max(ye))
            ax.set_aspect("equal")
            ax.set_title("Smoothed ratemap", fontsize=subaxis_title_fontsize)
            _stripAx(ax)

        posKeys = (
            "xy",
            "xydir",
            "rho",
            "phi",
            "labels",
            "fieldPerimMask",
            "runLabel",
            "fieldLabel",
            "peaksYXBins",
            "xe",
            "ye",
            "fieldPerimMask",
            "perim_angle_from_peak",
            "pos_angle_from_peak",
            "pos_dist_from_peak",
        )
        runsKeys = (
            "runStartIdx",
            "runEndIdx",
            "runDurationInPosBins",
            "runsMinSpeed",
            "meanDir",
            "mnSpd",
            "xy_new",
            "pos_x_unsmoothed",
            "pos_y_unsmoothed",
            "centralPeripheral",
            "spikes_per_run",
        )
        posDict = dict.fromkeys(posKeys, np.nan)
        # neat trick: locals is a dict that holds all locally scoped variables
        for thiskey in posDict.keys():
            posDict[thiskey] = locals()[thiskey]
        runsDict = dict.fromkeys(runsKeys, np.nan)
        for thiskey in runsDict.keys():
            runsDict[thiskey] = locals()[thiskey]
        return posDict, runsDict

    def getThetaProps(self):
        spikeTS = self.spike_ts
        phase = np.ma.MaskedArray(self.phase, mask=True)
        filteredEEG = self.filteredEEG
        oldAmplt = filteredEEG.copy()
        # get indices of spikes into eeg
        spkEEGIdx = np.ceil(spikeTS * self.lfp_sample_rate).astype(int)
        spkEEGIdx[spkEEGIdx > len(phase)] = len(phase) - 1
        spkCount = np.bincount(spkEEGIdx, minlength=len(phase))
        spkPhase = phase.copy()
        # unmask the valid entries
        spkPhase.mask[spkEEGIdx] = False
        minSpikingPhase = getPhaseOfMinSpiking(spkPhase)
        # force phase to lie between 0 and 2PI
        phaseAdj = fixAngle(
            phase - minSpikingPhase * (np.pi / 180) + self.allowed_min_spike_phase
        )
        isNegFreq = np.diff(np.unwrap(phaseAdj)) < 0
        isNegFreq = np.append(isNegFreq, isNegFreq[-1])
        # get start of theta cycles as points where diff > pi
        phaseDf = np.diff(phaseAdj)
        cycleStarts = phaseDf[1::] < -np.pi
        cycleStarts = np.append(cycleStarts, True)
        cycleStarts = np.insert(cycleStarts, 0, True)
        cycleStarts[isNegFreq] = False
        cycleLabel = np.cumsum(cycleStarts)

        # caculate power and find low power cycles
        power = np.power(filteredEEG, 2)
        cycleTotValidPow = np.bincount(
            cycleLabel[~isNegFreq], weights=power[~isNegFreq]
        )
        cycleValidBinCount = np.bincount(cycleLabel[~isNegFreq])
        cycleValidMnPow = cycleTotValidPow / cycleValidBinCount
        powRejectThresh = np.percentile(
            cycleValidMnPow, self.min_power_percent_threshold
        )
        cycleHasBadPow = cycleValidMnPow < powRejectThresh

        # find cycles too long or too short
        allowed_theta_len = np.floor(
            (1.0 / self.min_theta) * self.lfp_sample_rate
        ).astype(int), np.ceil((1.0 / self.max_theta) * self.lfp_sample_rate).astype(
            int
        )
        cycleTotBinCount = np.bincount(cycleLabel)
        cycleHasBadLen = np.logical_or(
            cycleTotBinCount > allowed_theta_len[0],
            cycleTotBinCount < allowed_theta_len[1],
        )

        # remove data calculated as 'bad'
        isBadCycle = np.logical_or(cycleHasBadLen, cycleHasBadPow)
        isInBadCycle = isBadCycle[cycleLabel]
        isBad = np.logical_or(isInBadCycle, isNegFreq)
        phaseAdj = np.ma.MaskedArray(phaseAdj, mask=np.invert(isBad))
        self.phaseAdj = phaseAdj
        ampAdj = np.ma.MaskedArray(filteredEEG, mask=np.invert(isBad))
        cycleLabel = np.ma.MaskedArray(cycleLabel, mask=np.invert(isBad))
        self.cycleLabel = cycleLabel
        spkCount = np.ma.MaskedArray(spkCount, mask=np.invert(isBad))
        # All the values in the dict below are the same length as the
        # number of EEG samples (all are also masked arrays except oldAmplt)
        out = {
            "phase": phaseAdj,
            "amp": ampAdj,
            "cycleLabel": cycleLabel,
            "oldPhase": phase.copy(),
            "oldAmplt": oldAmplt,
            "spkCount": spkCount,
        }
        return out

    def getSpikeProps(self, runLabel, meanDir, durationInPosBins):
        # TODO: the regressor values here need updating so they are the same length
        # as the number of positions and masked in the correct places to maintain
        # consistency with the regressors added in the getPosProps method
        spikeTS = self.spike_ts
        xy = self.RateMap.xy
        phase = self.phaseAdj
        cycleLabel = self.cycleLabel
        spkEEGIdx = np.ceil(spikeTS * self.lfp_sample_rate).astype(int)
        spkEEGIdx[spkEEGIdx > len(phase)] = len(phase) - 1
        spkPosIdx = np.ceil(spikeTS * self.pos_sample_rate).astype(int)
        spkPosIdx[spkPosIdx > xy.shape[1]] = xy.shape[1] - 1
        spkRunLabel = runLabel[spkPosIdx]
        thetaCycleLabel = cycleLabel[spkEEGIdx]
        firstInTheta = thetaCycleLabel[-1:] != thetaCycleLabel[1::]
        firstInTheta = np.insert(firstInTheta, 0, True)
        lastInTheta = firstInTheta[1::]
        numWithinRun = labelledCumSum(np.ones_like(spkRunLabel), spkRunLabel)
        thetaBatchLabelInRun = labelledCumSum(firstInTheta.astype(float), spkRunLabel)
        # breakpoint()

        spkCount = np.bincount(
            spkRunLabel[spkRunLabel > 0].compressed(), minlength=len(meanDir)
        )
        rateInPosBins = spkCount[1::] / durationInPosBins.astype(float)
        # update the regressor dict from __init__ with relevant values
        # all up at 1.0
        breakpoint()
        if "spk_numWithinRun" in self.regressors.keys():
            self.update_regressor_values("spk_numWithinRun", numWithinRun)
        # all up at 1.0
        if "spk_thetaBatchLabelInRun" in self.regressors.keys():
            self.update_regressor_values(
                "spk_thetaBatchLabelInRun", thetaBatchLabelInRun
            )
        spkKeys = (
            "spikeTS",
            "spkPosIdx",
            "spkEEGIdx",
            "spkRunLabel",
            "thetaCycleLabel",
            "firstInTheta",
            "lastInTheta",
            "numWithinRun",
            "thetaBatchLabelInRun",
            "spkCount",
            "rateInPosBins",
        )
        spkDict = dict.fromkeys(spkKeys, np.nan)
        for thiskey in spkDict.keys():
            spkDict[thiskey] = locals()[thiskey]
        print(f"Total spikes available for this cluster: {len(spikeTS)}")
        print(f"Total spikes used for analysis: {np.sum(spkCount)}")
        return spkDict

    def _ppRegress(self, spkDict, whichSpk="first"):

        phase = self.phaseAdj
        newSpkRunLabel = spkDict["spkRunLabel"].copy()
        # TODO: need code to deal with splitting the data based on a group of
        # variables
        spkUsed = newSpkRunLabel > 0
        # Calling compressed() method on spkUsed gives a boolean mask with length equal to
        # the number of spikes emitted by the cluster where True is a valid spike (ie it was
        # emitted when in a receptive field detected by the getPosProps() method above)
        # breakpoint()
        if "first" in whichSpk:
            spkUsed[~spkDict["firstInTheta"]] = False
        elif "last" in whichSpk:
            if len(spkDict["lastInTheta"]) < len(spkDict["spkRunLabel"]):
                spkDict["lastInTheta"] = np.insert(spkDict["lastInTheta"], -1, False)
            spkUsed[~spkDict["lastInTheta"]] = False
        spkPosIdxUsed = spkDict["spkPosIdx"].astype(int)
        # copy self.regressors and update with spk/ pos of interest
        regressors = self.regressors.copy()
        breakpoint()
        for k in regressors.keys():
            self.update_regressor_mask(k, spkPosIdxUsed)
            # if k.startswith("spk_"):
            #     self.update_regressor_values(k, regressors[k]["values"][spkUsed])
            # elif k.startswith("pos_"):
            #     self.update_regressor_values(
            #         k, regressors[k]["values"][spkPosIdxUsed[spkUsed]]
            #     )
        # breakpoint()
        phase = phase[spkDict["spkEEGIdx"][spkUsed]]
        phase = phase.astype(np.double)
        if "mean" in whichSpk:
            goodPhase = ~np.isnan(phase)
            cycleLabels = spkDict["thetaCycleLabel"][spkUsed]
            sz = np.max(cycleLabels)
            cycleComplexPhase = np.squeeze(np.zeros(sz, dtype=np.complex))
            np.add.at(
                cycleComplexPhase,
                cycleLabels[goodPhase] - 1,
                np.exp(1j * phase[goodPhase]),
            )
            phase = np.angle(cycleComplexPhase)
            spkCountPerCycle = np.bincount(cycleLabels[goodPhase], minlength=sz)
            for k in regressors.keys():
                regressors[k]["values"] = (
                    np.bincount(
                        cycleLabels[goodPhase],
                        weights=regressors[k]["values"][goodPhase],
                        minlength=sz,
                    )
                    / spkCountPerCycle
                )

        goodPhase = ~np.isnan(phase)
        for k in regressors.keys():
            print(f"Doing regression: {k}")
            goodRegressor = ~np.isnan(regressors[k]["values"])
            if np.any(goodRegressor):
                breakpoint()
                reg = regressors[k]["values"][np.logical_and(goodRegressor, goodPhase)]
                pha = phase[np.logical_and(goodRegressor, goodPhase)]
                regressors[k]["slope"], regressors[k]["intercept"] = circRegress(
                    reg, pha
                )
                regressors[k]["pha"] = pha
                mnx = np.mean(reg)
                reg = reg - mnx
                mxx = np.max(np.abs(reg)) + np.spacing(1)
                reg = reg / mxx
                # problem regressors = instFR, pos_d_cum
                # breakpoint()
                theta = np.mod(np.abs(regressors[k]["slope"]) * reg, 2 * np.pi)
                rho, p, rho_boot, p_shuff, ci = circCircCorrTLinear(
                    theta, pha, self.k, self.alpha, self.hyp, self.conf
                )
                regressors[k]["reg"] = reg
                regressors[k]["cor"] = rho
                regressors[k]["p"] = p
                regressors[k]["cor_boot"] = rho_boot
                regressors[k]["p_shuffled"] = p_shuff
                regressors[k]["ci"] = ci

        self.reg_phase = phase
        return regressors

    def plotRegressor(self, regressor: str, ax=None):
        assert regressor in self.regressors.keys()
        if ax is None:
            fig = plt.figure(figsize=(3, 5))
            ax = fig.add_subplot(111)
        else:
            ax = ax
        vals = self.regressors[regressor]["values"]
        pha = self.reg_phase
        slope = self.regressors[regressor]["slope"]
        intercept = self.regressors[regressor]["intercept"]
        mm = (0, -4 * np.pi, -2 * np.pi, 2 * np.pi, 4 * np.pi)
        for m in mm:
            ax.plot((-1, 1), (-slope + intercept + m, slope + intercept + m), "r", lw=3)
            ax.plot(vals, pha + m, "k.")
        ax.set_xlim(-1, 1)
        xtick_locs = np.linspace(-1, 1, 3)
        ax.set_xticks(xtick_locs, list(map(str, xtick_locs)))
        ax.set_yticks(sorted(mm), ["-4π", "-2π", "0", "2π", "4π"])
        ax.set_ylim(-2 * np.pi, 4 * np.pi)
        title_str = f"{regressor} vs phase: slope = {slope:.2f}, \nintercept = {intercept:.2f}, p_shuffled = {self.regressors[regressor]['p_shuffled']:.2f}"
        ax.set_title(title_str, fontsize=subaxis_title_fontsize)
        ax.set_ylabel("Phase", fontsize=subaxis_title_fontsize)
        ax.set_xlabel("Normalised position", fontsize=subaxis_title_fontsize)
        return ax

    def plotPPRegression(self, regressorDict, regressor2plot="pos_d_cum", ax=None):

        t = self.getLFPPhaseValsForSpikeTS()
        x = self.RateMap.xy[0, self.spike_times_in_pos_samples]
        from ephysiopy.common import fieldcalcs

        rmap = self.RateMap.get_map(self.spk_weights)
        xe, ye = rmap.bin_edges
        label = fieldcalcs.field_lims(rmap)
        rmap = rmap.binned_data[0].T
        xInField = xe[label.nonzero()[1]]
        mask = np.logical_and(x > np.min(xInField), x < np.max(xInField))
        x = x[mask]
        t = t[mask]
        # keep x between -1 and +1
        mnx = np.mean(x)
        xn = x - mnx
        mxx = np.max(np.abs(xn))
        x = xn / mxx
        # keep tn between 0 and 2pi
        t = np.remainder(t, 2 * np.pi)
        slope, intercept = circRegress(x, t)
        rho, p, rho_boot, p_shuff, ci = circCircCorrTLinear(x, t)
        plt.figure()
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = ax
        ax.plot(x, t, ".", color="k")
        ax.plot(x, t + 2 * np.pi, ".", color="k")
        mm = (0, -2 * np.pi, 2 * np.pi, 4 * np.pi)
        for m in mm:
            ax.plot((-1, 1), (-slope + intercept + m, slope + intercept + m), "r", lw=3)
        ax.set_xlim((-1, 1))
        ax.set_ylim((-np.pi, 3 * np.pi))
        return {
            "slope": slope,
            "intercept": intercept,
            "rho": rho,
            "p": p,
            "rho_boot": rho_boot,
            "p_shuff": p_shuff,
            "ci": ci,
        }

    def getLFPPhaseValsForSpikeTS(self):
        ts = self.spike_times_in_pos_samples * (
            self.lfp_sample_rate / self.pos_sample_rate
        )
        ts_idx = np.array(np.floor(ts), dtype=int)
        return self.phase[ts_idx]


# Define a group of static methods for doing various operations on circular
# and labelled data
