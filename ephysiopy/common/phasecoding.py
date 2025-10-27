from ephysiopy.common.fieldcalcs import FieldProps, filter_runs
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from ephysiopy.common.rhythmicity import LFPOscillations
from ephysiopy.common.utils import (
    flatten_list,
    fixAngle,
    find_runs,
    repeat_ind,
)
from ephysiopy.common.fieldproperties import LFPSegment, fieldprops
from ephysiopy.common.fieldcalcs import (
    partitionFields,
    infill_ratemap,
)
from ephysiopy.io.recording import AxonaTrial
from scipy import ndimage, optimize, signal
from scipy.stats import norm
from collections import defaultdict
from dataclasses import dataclass
import pywt
import pycwt


jet_cmap = matplotlib.colormaps["jet"]

subaxis_title_fontsize = 10
cbar_fontsize = 8
cbar_tick_fontsize = 6


# a dataclass for holding the results of the circular correlation
# TODO: add default np.nan or None values to each member variable
@dataclass
class CircStatsResults:
    rho: float = np.nan
    p: float = np.nan
    rho_boot: float = np.nan
    p_shuffled: float = np.nan
    ci: float = np.nan

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
            f"ci=({self.ci_lower:.3f}, {self.ci_upper:.3f})"
        )


def get_cycle_labels(
    phase: np.ndarray, min_allowed_min_spike_phase: float
) -> tuple[np.ndarray, ...]:
    """
    Get the cycle labels for a given phase array

    Parameters
    ----------
    phase : np.ndarray
        The phases at which the spikes were fired.
    min_allowed_min_spike_phase : float
        The minimum allowed phase for cycles to start.

    Returns
    -------
    np.ndarray
        The cycle labels for the phase array
    """
    # force phase to lie between 0 and 2PI
    minSpikingPhase = get_phase_of_min_spiking(phase)
    phaseAdj = fixAngle(
        phase - minSpikingPhase * (np.pi / 180) + min_allowed_min_spike_phase
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

    return cycleLabel, phaseAdj


def get_phase_of_min_spiking(spkPhase: np.ndarray) -> float:
    """
    Returns the phase at which the minimum number of spikes are fired

    Parameters
    ----------
    spkPhase : np.ndarray
        The phase of the spikes

    Returns
    -------
    float
        The phase at which the minimum number of spikes are fired


    """
    kernelLen = 180
    kernelSig = kernelLen / 4

    regressor = signal.windows.gaussian(kernelLen, kernelSig)
    bins = np.arange(-179.5, 180, 1)
    phaseDist, _ = np.histogram(spkPhase / np.pi * 180, bins=bins)
    phaseDist = ndimage.convolve(phaseDist, regressor)
    phaseMin = bins[
        int(np.ceil(np.nanmean(np.nonzero(phaseDist == np.min(phaseDist))[0])))
    ]
    return phaseMin


def get_bad_cycles(
    filtered_eeg: np.ndarray,
    negative_freqs: np.ndarray,
    cycle_labels: np.array,
    min_power_percent_threshold: float,
    min_theta: float,
    max_theta: float,
    lfp_fs: float,
) -> np.ndarray:
    """
    Get the cycles that are bad based on their length and power

    Parameters
    ----------
    filtered_eeg : np.ndarray
        The filtered EEG signal
    negative_freqs : np.ndarray
        A boolean array indicating negative frequencies
    cycle_labels : np.ndarray
        The cycle labels for the phase array
    min_power_percent_threshold : float
        The minimum power percent threshold for rejecting cycles
    min_theta : float
        The minimum theta frequency
    max_theta : float
        The maximum theta frequency
    lfp_fs : float
        The sampling frequency of the LFP signal

    Returns
    -------
    np.ndarray
        A boolean array indicating bad cycles
    """
    power = np.power(filtered_eeg, 2)
    cycle_valid_power = np.bincount(
        cycle_labels[~negative_freqs], weights=power[~negative_freqs]
    )
    cycle_valid_bincount = np.bincount(cycle_labels[~negative_freqs])
    cycle_valid_mn_power = cycle_valid_power / cycle_valid_bincount
    power_rejection_thresh = np.percentile(
        cycle_valid_mn_power, min_power_percent_threshold
    )
    # get the cycles that are below the rejection threshold
    bad_power_cycle = cycle_valid_mn_power < power_rejection_thresh

    # find cycle too long or too short
    allowed_cycle_len = (
        np.floor((1.0 / max_theta) * lfp_fs).astype(int),
        np.ceil((1.0 / min_theta) * lfp_fs).astype(int),
    )
    cycle_bincount_total = np.bincount(cycle_labels)
    bad_len_cycles = np.logical_or(
        cycle_bincount_total < allowed_cycle_len[0],
        cycle_bincount_total > allowed_cycle_len[1],
    )
    bad_cycle = np.logical_or(bad_len_cycles, bad_power_cycle)
    in_bad_cycle = bad_cycle[cycle_labels]
    is_bad = np.logical_or(in_bad_cycle, negative_freqs)

    return is_bad


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
    rho = 4 * (A * B - C * D) / np.sqrt((n**2 - E**2 - F**2) * (n**2 - G**2 - H**2))
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
    rho = 4 * (A * B - C * D) / np.sqrt((n**2 - E**2 - F**2) * (n**2 - G**2 - H**2))
    return rho


def plot_spikes_in_runs_per_field(
    field_label: np.ndarray,
    run_starts: np.ndarray,
    run_ends: np.ndarray,
    spikes_in_time: np.ndarray,
    ttls_in_time: np.ndarray | None = None,
    **kwargs,
):
    """
    Debug plotting to show spikes per run per field found in the ratemap
    as a raster plot

    Parameters
    ----------
    field_label : np.ndarray
    The field labels for each position bin a vector
    run_starts, runs_ends : np.ndarray
        The start and stop indices of each run (vectors)
    spikes_in_time : np.ndarray
        The number of spikes in each position bin (vector)
    ttls_in_time : np.ndarray
        TTL occurences in time (vector)

    **kwargs
        separate_plots : bool
            If True then each field will be plotted in a separate figure
        single_axes : bool
            If True will plot all the runs/ spikes in a single axis with fields delimited by horizontal lines

    Returns
    -------
    fig, axes : tuple
        The figure and axes objects
    """
    spikes_in_time = np.ravel(spikes_in_time)
    if ttls_in_time:
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
        if ttls_in_time:
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
        if ttls_in_time:
            ttls = np.array(flatten_list(ttls_per_field))
            ax.imshow(ttls, cmap=matplotlib.colormaps["bone"])
        spiking_arr = np.array(flatten_list(master_raster_arr))
        ax.imshow(spiking_arr, cmap=newcmap, alpha=0.6)
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.hlines(np.cumsum(runs_per_field)[:-1], 0, max_run_len, "regressor")
        ax.set_xlim(0, max_run_len)
        ytick_locs = np.insert(np.cumsum(runs_per_field), 0, 0)
        ytick_locs = np.diff(ytick_locs) // 2 + ytick_locs[:-1]
        ax.set_yticks(ytick_locs, list(map(str, np.unique(run_field_id))))
        ax.set_ylabel("Field ID", rotation=90, labelpad=10)
        ax.set_xlabel("Time (s)")
        ax.set_xticks([0, max_run_len], ["0", f"{(max_run_len) / 50:.2f}"])
        axes2 = ax.twinx()
        axes2.set_yticks(ytick_locs, list(map(str, spikes_per_run)))
        axes2.set_ylim(ax.get_ylim())
        axes2.set_ylabel("Spikes per field", rotation=270, labelpad=10)


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
        number of permutations to use to calculate p-value from randomisation and
        bootstrap estimation of confidence intervals.
        Leave empty to calculate p-value analytically (NB confidence
        intervals will not be calculated).
    alpha : float, default=0.05
        hypothesis test level e.g. 0.05, 0.01 etc.
    hyp : int, default=0
        hypothesis to test; -1/ 0 / 1 (-ve correlated / correlated in either direction / positively correlated).
    conf : bool, default=True
        True or False to calculate confidence intervals via jackknife or bootstrap.

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
            rho_boot - (1 / np.sqrt(n)) * rho_jack_std * norm.ppf(alpha / 2, (0, 1))[0],
            rho_boot + (1 / np.sqrt(n)) * rho_jack_std * norm.ppf(alpha / 2, (0, 1))[0],
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

    rho_sim = 4 * (A * B - C * D) / np.sqrt((n**2 - E**2 - F**2) * (n**2 - G**2 - H**2))

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
        slope = optimize.fminbound(_cost, -1 * max_slope, max_slope, args=(xn, tn))
    except ValueError:
        return np.nan, np.nan
    intercept = np.arctan2(
        np.sum(np.sin(tn - slope * xn)), np.sum(np.cos(tn - slope * xn))
    )
    intercept = intercept + ((0 - slope) * (mnx / mxx))
    slope = slope / mxx
    return slope, intercept


"""
A dictionary containing parameters for the phase precession analysis
"""
phase_precession_config = {
    "pos_sample_rate": 50,
    "lfp_sample_rate": 250,
    "cms_per_bin": 1,  # bin size gets calculated in Ratemap
    "ppm": 445,
    "field_smoothing_kernel_len": 31,
    "field_smoothing_kernel_sigma": 13,
    # minimum firing rate - values below this are discarded (turned to 0)
    "field_threshold": 0.5,
    # field threshold percent - fed into fieldcalcs.local_threshold as prc
    "field_threshold_percent": 20,
    # fractional limit for restricting fields size
    "area_threshold": 0.01,
    # making the bins_per_cm value <1 leads to truncation of xy values
    # on unit circle
    "bins_per_cm": 1,
    "convert_xy_2_cm": True,
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
    "min_spikes": 1,  # min allowed spikes per run
    # instantaneous firing rate (ifr) smoothing constant
    "ifr_smoothing_constant": 1.0 / 3,
    "spatial_lowpass_cutoff": 3,
    "ifr_kernel_len": 1,  # ifr smoothing kernal length
    "ifr_kernel_sigma": 0.5,
    "bins_per_second": 50,  # bins per second for ifr smoothing
}

"""
A list of the regressors that can be used in the phase precession analysis
"""
all_regressors = [
    "spk_numWithinRun",
    "pos_exptdRate_cum",
    "pos_instFR",
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

    Parameters
    ----------
    T : AxonaTrial (or OpenEphysBase eventually)
        The trial object holding position, LFP, spiking and ratemap stuff
    cluster : int
        the cluster to examine
    channel : int
        The channel the cluster was recorded on
    pp_config : dict
        Contains parameters for running the analysis.
        See phase_precession_config dict in ephysiopy.common.eegcalcs
    regressors : list
        A list of the regressors to use in the analysis

    Attributes
    ----------
    orig_xy : np.ndarray
        The original position data
    pos_ts : np.ndarray
        The position timestamps
    spike_ts : np.ndarray
        The spike timestamps
    regressors : dict
        A dictionary containing the regressors and their values
    alpha : float
        The alpha value for hypothesis testing
    hyp : int
        The hypothesis to test
    conf : bool
        Whether to calculate confidence intervals
    eeg : np.ndarray
        The EEG signal
    min_theta : int
        The minimum theta frequency
    max_theta : int
        The maximum theta frequency
    filteredEEG : np.ndarray
        The filtered EEG signal
    phase : np.ndarray
        The phase of the EEG signal
    phaseAdj : np.ma.MaskedArray
        The adjusted phase of the EEG signal as a masked array
    spike_times_in_pos_samples : np.ndarray
        The spike times in position samples (vector with length = npos)
    spk_weights : np.ndarray
        The spike weights (vector with length = npos)
    """

    def __init__(
        self,
        T: AxonaTrial,
        cluster: int,
        channel: int,
        pp_config: dict = phase_precession_config,
        regressors=None,
    ):
        if not T.PosCalcs:
            T.load_pos_data()
        if not T.EEGCalcs:
            T.load_lfp()
        if not T.RateMap:
            T.initialise()

        self.trial = T
        self.cluster = cluster
        self.channel = channel

        # ---------- Set up the parameters ----------
        # this adds, as attributes, the parameters defined in the
        # pp_config dictionary
        self.update_config(pp_config)

        # Positional params...
        self.orig_xy = T.PosCalcs.xy
        self._pos_ts = T.PosCalcs.xyTS

        # ratemap params...
        self.trial.RateMap.smooth_sz = self.field_smoothing_kernel_len

        # regressors...
        self.nshuffles = 1000
        self.update_regressors(regressors)
        self.alpha = 0.05
        self.hyp = 0
        self.conf = True

        # LFP params...
        self.eeg = T.EEGCalcs.sig
        self.lfp_fs = T.EEGCalcs.fs
        L = LFPOscillations(self.eeg, self.lfp_fs)
        self.min_theta = pp_config["min_theta"]
        self.max_theta = pp_config["max_theta"]
        FP = L.getFreqPhase(self.eeg, [self.min_theta, self.max_theta], 2)
        self.filteredEEG = FP.filt_sig
        self.phase = FP.phase
        self.phaseAdj = np.ma.MaskedArray

        # Some spiking params...
        spk_times_in_pos_samples = T.get_binned_spike_times(cluster, channel)
        spk_times_in_pos_samples = np.ravel(spk_times_in_pos_samples).astype(int)
        spk_weights = np.bincount(
            spk_times_in_pos_samples, minlength=len(T.PosCalcs.xyTS)
        )
        self.spike_times_in_pos_samples = spk_times_in_pos_samples
        self.spk_weights = spk_weights

        self.spike_ts = T.get_spike_times(cluster, channel)

    @property
    def pos_ts(self):
        return self._pos_ts

    @pos_ts.setter
    def pos_ts(self, value):
        self._pos_ts = value

    @property
    def spike_eeg_idx(self):
        return (self.spike_ts * self.lfp_fs).astype(int)

    @property
    def spike_pos_idx(self):
        return (self.spike_ts * self.pos_sample_rate).astype(int)

    def update_regressors_from_runs(self, field_props: list[FieldProps]) -> None:
        """
        Once the various metrics have been calculated for each run fill
        out the regressor dict with the relevant values
        """
        if "pos_d_currentdir" in self.regressors.keys():
            d_currentdir = np.ones(shape=self.trial.PosCalcs.npos) * np.nan
            for field in field_props:
                for irun in field.runs:
                    try:
                        d_currentdir[irun._slice] = irun.current_direction
                    except TypeError:
                        print(f"field label: {field.label} - run label: {irun.label}")
                        breakpoint()

            self.update_regressor_values("pos_d_currentdir", d_currentdir)

        # calculate the cumulative distance travelled on each run
        # only goes from 0-1
        if "pos_d_cum" in self.regressors.keys():
            d_cumulative = np.ones(shape=self.trial.PosCalcs.npos) * np.nan
            for f in field_props:
                for r in f.runs:
                    d_cumulative[r._slice] = r.cumulative_distance

            self.update_regressor_values("pos_d_cum", d_cumulative)

        # calculate cumulative sum of the expected normalised firing rate
        # only goes from 0-1
        if "pos_exptdRate_cum" in self.regressors.keys():
            exptd_rate_all = np.ones(shape=self.trial.PosCalcs.npos) * np.nan
            # breakpoint()
            rmap = field_props[0].binned_data.binned_data[0]
            ye, xe = field_props[0].binned_data.bin_edges
            xy = self.trial.PosCalcs.xy
            xBins = np.digitize(xy[0], xe[:-1])
            yBins = np.digitize(xy[1], ye[:-1])
            rmap_infilled = infill_ratemap(rmap)
            exptd_rate = rmap_infilled[yBins - 1, xBins - 1]
            # setting the sample rate to 1 here will result in firing rate
            # being returned and not expected spike count
            for f in field_props:
                for r in f.runs:
                    exptd_rate_all[r._slice] = r.expected_spikes(exptd_rate, 1)

            self.update_regressor_values("pos_exptdRate_cum", exptd_rate_all)

        # direction projected onto the run mean direction is just the x coord
        # good - remembering that xy_new is rho,phi
        # this might be wrong - need to check i'm grabbing the right value
        # from FieldProps... could be rho
        if "pos_d_meanDir" in self.regressors.keys():
            d_meandir = np.ones(shape=self.trial.PosCalcs.npos) * np.nan
            for f in field_props:
                for irun in f.runs:
                    d_meandir[irun._slice] = irun.pos_r

            breakpoint()
            self.update_regressor_values("pos_d_meanDir", d_meandir)

        # smooth binned spikes to get an instantaneous firing rate
        # set up the smoothing kernel
        # all up at 1.0
        if "pos_instFR" in self.regressors.keys():
            kernLenInBins = np.round(self.ifr_kernel_len * self.bins_per_second)
            kernSig = self.ifr_kernel_sigma * self.bins_per_second
            regressor = signal.windows.gaussian(kernLenInBins, kernSig)
            # apply the smoothing kernel over the binned observed spikes
            observed_spikes_in_time = self.spike_times_in_pos_samples
            ifr = signal.convolve(observed_spikes_in_time, regressor, mode="same")
            inst_firing_rate = np.zeros_like(ifr)
            for field in field_props:
                for i_slice in field.run_slices:
                    inst_firing_rate[i_slice] = ifr[i_slice]
            self.update_regressor_values("pos_instFR", inst_firing_rate)

        # find time spent within run
        # only goes from 0-1
        if "pos_timeInRun" in self.regressors.keys():
            time_in_run = np.ones(shape=self.trial.PosCalcs.npos) * np.nan
            for f in field_props:
                for r in f.runs:
                    time_in_run[r._slice] = r.cumulative_time / self.pos_sample_rate

            self.update_regressor_values("pos_timeInRun", time_in_run)

    def update_regressors(self, reg_keys: list | None):
        """
        Create a dict to hold the stats values for
        each regressor
        Default regressors are:
            "spk_numWithinRun",
            "pos_exptdRate_cum",
            "pos_instFR",
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
        assert all([regressor in all_regressors for regressor in reg_keys])

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
        [self.regressors[regressor] for regressor in reg_keys]
        # each of the regressors in regressor_keys is a key with a value
        # of stats_dict

    def update_regressor_values(self, key: str, values):
        """Check whether values is a masked array and if not make it one"""
        self.regressors[key]["values"] = values

    def update_regressor_mask(self, key: str, indices):
        """Mask entries in the 'values' and 'pha' arrays of the relevant regressor"""
        self.regressors[key]["values"].mask[indices] = False

    def get_regressors(self):
        return self.regressors.keys()

    def get_regressor(self, key):
        return self.regressors[key]

    def update_config(self, pp_config):
        """Update the relevant pp_config values"""
        [
            setattr(self, attribute, pp_config[attribute])
            for attribute in pp_config.keys()
        ]

    def performRegression(self, **kwargs):
        """
        Wrapper function for doing the actual regression which has multiple
        stages.

        Specifically here we partition fields into sub-fields, get a bunch of
        information about the position, spiking and theta data and then
        do the actual regression.

        **kwargs
            do_plot : bool
            whether to plot the results of field partitions, the regression(s)

        See Also
        --------
        ephysiopy.common.eegcalcs.phasePrecession.partitionFields()
        ephysiopy.common.eegcalcs.phasePrecession.getPosProps()
        ephysiopy.common.eegcalcs.phasePrecession.getThetaProps()
        ephysiopy.common.eegcalcs.phasePrecession.getSpikeProps()
        ephysiopy.common.eegcalcs.phasePrecession._ppRegress()
        """
        do_plot = kwargs.get("plot", False)

        # Partition fields - comes from ephysiopy.common.fieldca
        binned_data = self.trial.get_rate_map(self.cluster, self.channel)
        _, _, labels, _ = partitionFields(
            binned_data,
            self.field_threshold_percent,
            self.field_threshold,
            self.area_threshold,
        )
        # breakpoint()

        # split into runs
        field_properties = self.getPosProps(labels)

        # get theta cycles, amplitudes, phase etc
        field_properties = self.getThetaProps(field_properties)
        # TODO: next: getSpikeProps(field_properties)
        # the fields that are set within getSpikeProps() I think can be added
        # to the individual runs within the FieldProps instance
        # spkCount is just spikes binned wrt eeg timebase
        #
        # at this point the 'values' and 'pha' arrays in the regressors dict are all
        # npos elements long and are masked arrays. keep as masked arrays and just modify the
        # masks instead of truncating the data
        # Do the regressions
        self.ppPerField(field_properties)

        # Plot the results if asked
        # TODO: make one figure per regressor with subplots for each field
        # see linear_track.py
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

    def get_phase_reg_per_field(self, field_props: list[FieldProps]) -> dict:
        """
        Extracts the phase and all regressors for all runs through each
        field separately

        Parameters
        ----------
        field_props : list
            A list of FieldProps instances

        Returns
        -------
        dict
            two-level dictionary holding regression results per field
            first level keys are field number
            second level are the regressors (current_dir etc)
            items in the second dict are the regression results
        """

        # helper function to extract the relevant values from
        # a variable list (e.g. cumulative distance) at the times
        # when spikes were observed
        def get_spiking_var(var: np.ndarray, observed_spikes: np.ndarray) -> np.ndarray:
            var = np.array(flatten_list(var))
            observed_spikes = np.array(flatten_list(observed_spikes))
            return flatten_list(np.take(var, repeat_ind(observed_spikes)))

        results = dict.fromkeys([f.label for f in field_props])
        for field in field_props:
            results[field.label] = {}
            results[field.label]["phase"] = np.array(
                flatten_list(
                    [run.lfp_data.get_spiking_var().ravel() for run in field.runs]
                )
            )
            for regressor in self.get_regressors():
                match regressor:
                    case "pos_timeInRun":
                        vals = np.array(
                            get_spiking_var(run.cumulative_time, run.observed_spikes)
                            for run in field.runs
                        )
                    case "spk_numWithinRun":
                        vals = np.array(
                            flatten_list([run.spike_num_in_run for run in field.runs])
                        )
                    # case "spk_thetaBatchLabelInRun":
                    #     vals = np.array(
                    #         flatten_list(
                    #             [run.spike_theta_batch_labels for run in field.runs]
                    #         )
                    #     )
                    case "pos_d_currentdir":
                        vals = np.array(
                            get_spiking_var(
                                field.current_direction, field.observed_spikes
                            )
                        )
                    case "pos_d_cum":
                        vals = np.array(
                            get_spiking_var(
                                run.cumulative_distance, run.observed_spikes
                            )
                            for run in field.runs
                        )
                    case "pos_exptdRate_cum":
                        vals = np.array(
                            flatten_list([run.expected_spikes for run in field.runs])
                        )
                    case "pos_instFR":
                        vals = np.array(
                            flatten_list(
                                [run.instantaneous_firing_rate for run in field.runs]
                            )
                        )

                results[field.label][regressor] = vals
        return results
        # self._ppRegress(phase, vals)

    def getPosProps(
        self,
        labels: np.ndarray,
    ) -> list:
        """
        Uses the output of partitionFields and returns vectors the same
        length as pos.

        Parameters
        ----------
        labels : np.ndarray
            The labels of the fields

        Returns
        -------
        list of FieldProps
            A list of FieldProps instances (see ephysiopy.common.fieldcalcs.FieldProps)
        """
        xy = self.trial.PosCalcs.xy

        binned_data = self.trial.get_rate_map(self.cluster, self.channel)
        # The large number of bins combined with the super-smoothed ratemap
        # will lead to fields labelled with lots of small holes in. Fill those
        # gaps in here and calculate the perimeter of the fields based on that
        # labelled image
        labels, _ = ndimage.label(ndimage.binary_fill_holes(labels))

        field_props = fieldprops(
            labels,
            binned_data,
            self.trial.get_spike_times(self.cluster, self.channel),
            xy,
            sample_rate=self.trial.PosCalcs.sample_rate,
        )
        print(
            f"Filtering runs for min duration {self.minimum_allowed_run_duration}, mean speed {self.minimum_allowed_run_speed} and min spikes {self.min_spikes}"
        )
        field_props = filter_runs(
            field_props,
            self.minimum_allowed_run_duration,
            self.minimum_allowed_run_speed,
            min_spikes=1,
        )
        # Smooth the runs before calculating other metrics
        [
            f.smooth_runs(
                self.ifr_smoothing_constant,
                self.spatial_lowpass_cutoff,
                self.pos_sample_rate,
            )
            for f in field_props
        ]

        return field_props

    def getThetaProps(self, field_props: list[FieldProps]):
        """
        Processes the LFP data and inserts into each run within each field
        a segment of LFP data that has had its phase and amplitude extracted
        as well as some other data

        Parameters
        ----------
        field_props : list[FieldProps]
            A list of FieldProps instances

        Returns
        -------
        list of FieldProps
            The amended list with LFP data added to each run for each field

        """
        phase = np.ma.MaskedArray(self.phase, mask=True)
        # get indices of spikes into eeg
        spkEEGIdx = self.spike_eeg_idx
        spkPhase = phase.copy()
        # unmask the valid entries
        spkPhase.mask[spkEEGIdx] = False

        cycleLabel, phaseAdj = get_cycle_labels(spkPhase, self.allowed_min_spike_phase)
        isNegFreq = np.diff(np.unwrap(phaseAdj)) < 0
        isNegFreq = np.append(isNegFreq, isNegFreq[-1])

        isBad = get_bad_cycles(
            self.filteredEEG,
            isNegFreq,
            cycleLabel,
            self.min_power_percent_threshold,
            self.min_theta,
            self.max_theta,
            self.lfp_fs,
        )
        self.bad_cycles = isBad
        cycleLabel = np.ma.MaskedArray(cycleLabel, mask=np.invert(isBad))
        self.cycleLabel = cycleLabel
        lfp_to_pos_ratio = int(self.lfp_fs / self.pos_sample_rate)
        spike_times = self.trial.get_spike_times(self.cluster, self.channel)

        for field in field_props:
            for run in field.runs:
                lfp_slice = slice(
                    run.slice.start * lfp_to_pos_ratio,
                    run.slice.stop * lfp_to_pos_ratio,
                )
                lfp_segment = LFPSegment(
                    run,
                    field.label,
                    run.label,
                    lfp_slice,
                    spike_times=spike_times,
                    mask=np.invert(isBad)[lfp_slice],
                    signal=self.eeg[lfp_slice],
                    filtered_signal=self.filteredEEG[lfp_slice],
                    phase=phaseAdj[lfp_slice],
                    cycle_label=cycleLabel[lfp_slice],
                    sample_rate=self.lfp_fs,
                )
                # lfp_segment.mask(np.invert(isBad)[lfp_slice])
                run.lfp_data = lfp_segment

        return field_props

    def _ppRegress(self, spkDict, whichSpk="first"):
        """
        Perform the regression analysis on the spike data for each field
        in the list of FieldProps

        Parameters
        ----------
        spkDict : dict
            A dictionary containing the spike properties
        whichSpk : str
            Which spike(s) in a cycle to use in the regression analysis

        Returns
        -------
        list
            A list of the updated regressors
        """

        phase = self.phaseAdj
        newSpkRunLabel = spkDict["spkRunLabel"].copy()
        # TODO: need code to deal with splitting the data based on a group of
        # variables
        spkUsed = newSpkRunLabel > 0
        # Calling compressed() method on spkUsed gives a boolean mask with length equal to
        # the number of spikes emitted by the cluster where True is a valid spike (ie it was
        # emitted when in a receptive field detected by the getPosProps() method above)
        # firstInTheta (and presumably lastInTheta) need to be the same length as
        # the number of pos samples - currently it's just the length of some smaller
        # subset of the length of the number of spikes
        if "first" in whichSpk:
            spkUsed[~spkDict["firstInTheta"]] = False
        elif "last" in whichSpk:
            if len(spkDict["lastInTheta"]) < len(spkDict["spkRunLabel"]):
                spkDict["lastInTheta"] = np.insert(spkDict["lastInTheta"], -1, False)
            spkUsed[~spkDict["lastInTheta"]] = False
        spkPosIdxUsed = spkDict["spkPosIdx"].astype(int)
        # copy self.regressors and update with spk/ pos of interest
        regressors = self.regressors.copy()
        # the length of the 'values' of the regressors is dependent on the variable
        # so 'pos_' regressors are vectors as long as the number of position samples
        # and 'spk_' regressors are as long as the number of spikes for the current
        # cluster. These different length vectors need to be dealt with differently..
        for regressor in regressors.keys():
            # breakpoint()
            # self.update_regressor_mask(regressor, spkPosIdxUsed)
            if regressor.startswith("spk_"):
                self.update_regressor_values(
                    regressor, regressors[regressor]["values"][spkUsed]
                )
            elif regressor.startswith("pos_"):
                self.update_regressor_values(
                    regressor, regressors[regressor]["values"][spkPosIdxUsed[spkUsed]]
                )
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
            for regressor in regressors.keys():
                regressors[regressor]["values"] = (
                    np.bincount(
                        cycleLabels[goodPhase],
                        weights=regressors[regressor]["values"][goodPhase],
                        minlength=sz,
                    )
                    / spkCountPerCycle
                )

        # a lot of the logic for determining "good phase" and a "good regeressor"
        # has been taken care of in the various methods that build and modify the
        # list of FieldProps (called field_props throughout this class)
        goodPhase = ~np.isnan(phase)
        for regressor in regressors.keys():
            print(f"Doing regression: {regressor}")
            goodRegressor = ~np.isnan(regressors[regressor]["values"])
            if np.any(goodRegressor):
                # breakpoint()
                reg = regressors[regressor]["values"][
                    np.logical_and(goodRegressor, goodPhase)
                ]
                pha = phase[np.logical_and(goodRegressor, goodPhase)]
                # TODO: all the pha values are masked. make sure only the relevant ones are!
                regressors[regressor]["slope"], regressors[regressor]["intercept"] = (
                    circRegress(reg, pha)
                )
                regressors[regressor]["pha"] = pha
                mnx = np.mean(reg)
                reg = reg - mnx
                mxx = np.max(np.abs(reg)) + np.spacing(1)
                reg = reg / mxx
                # problem regressors = instFR, pos_d_cum
                # breakpoint()
                theta = np.mod(np.abs(regressors[regressor]["slope"]) * reg, 2 * np.pi)
                results = circCircCorrTLinear(
                    theta, pha, self.nshuffles, self.alpha, self.hyp, self.conf
                )
                regressors[regressor]["reg"] = reg
                regressors[regressor]["cor"] = results.rho
                regressors[regressor]["p"] = results.p
                regressors[regressor]["cor_boot"] = results.rho_boot
                regressors[regressor]["p_shuffled"] = results.p_shuffled
                regressors[regressor]["ci"] = results.ci

        self.reg_phase = phase
        return regressors

    def plotRegressor(self, regressor: str, ax=None):
        """
        Plot the regressor against the phase

        Parameters
        ----------
        regressor : str
            The regressor to plot
        ax : matplotlib.axes.Axes
            The axes to plot on

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plot
        """
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

    # def plotPPRegression(self, regressorDict, regressor2plot="pos_d_cum", ax=None):
    #
    #     t = self.getLFPPhaseValsForSpikeTS()
    #     x = self.RateMap.xy[0, self.spike_times_in_pos_samples]
    #     from ephysiopy.common import fieldcalcs
    #
    #     rmap = self.RateMap.get_map(self.spk_weights)
    #     xe, ye = rmap.bin_edges
    #     label = fieldcalcs.field_lims(rmap)
    #     rmap = rmap.binned_data[0].T
    #     xInField = xe[label.nonzero()[1]]
    #     mask = np.logical_and(x > np.min(xInField), x < np.max(xInField))
    #     x = x[mask]
    #     t = t[mask]
    #     # keep x between -1 and +1
    #     mnx = np.mean(x)
    #     xn = x - mnx
    #     mxx = np.max(np.abs(xn))
    #     x = xn / mxx
    #     # keep tn between 0 and 2pi
    #     t = np.remainder(t, 2 * np.pi)
    #     slope, intercept = circRegress(x, t)
    #     rho, p, rho_boot, p_shuff, ci = circCircCorrTLinear(x, t)
    #     plt.figure()
    #     if ax is None:
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111)
    #     else:
    #         ax = ax
    #     ax.plot(x, t, ".", color="regressor")
    #     ax.plot(x, t + 2 * np.pi, ".", color="regressor")
    #     mm = (0, -2 * np.pi, 2 * np.pi, 4 * np.pi)
    #     for m in mm:
    #         ax.plot((-1, 1), (-slope + intercept + m, slope + intercept + m), "r", lw=3)
    #     ax.set_xlim((-1, 1))
    #     ax.set_ylim((-np.pi, 3 * np.pi))
    #     return {
    #         "slope": slope,
    #         "intercept": intercept,
    #         "rho": rho,
    #         "p": p,
    #         "rho_boot": rho_boot,
    #         "p_shuff": p_shuff,
    #         "ci": ci,
    #     }
    #
    # def getLFPPhaseValsForSpikeTS(self):
    #     ts = self.spike_times_in_pos_samples * (
    #         self.lfp_fs / self.pos_sample_rate
    #     )
    #     ts_idx = np.array(np.floor(ts), dtype=int)
    #     return self.phase[ts_idx]


# ----------- WHOLE LFP ANALYSIS -----------
# these methods look at the whole LFP i.e are
# not limited to looking at spiking and its
# relation to the LFP


def theta_filter_lfp(lfp: np.ndarray, fs: float, **kwargs):
    """
    Processes an LFP signal for theta cycles, filtering
    out bad cycles (low power, too long/ short etc) and
    applying labels to each cycle etc
    """
    # get/ set some args
    min_power_percent_threshold = kwargs.pop("min_power_percent_threshold", 0)
    min_theta = kwargs.pop("min_theta", 6)
    max_theta = kwargs.pop("max_theta", 12)

    L = LFPOscillations(lfp, fs)
    freq_phase = L.getFreqPhase(lfp, (6, 12))
    phase = freq_phase.phase

    is_neg_freq = np.diff(np.unwrap(phase)) < 0
    is_neg_freq = np.append(is_neg_freq, is_neg_freq[-1])
    # get start of theta cycles as points where diff > pi
    phase_df = np.diff(phase)
    cycle_starts = phase_df[1::] < -np.pi
    cycle_starts = np.append(cycle_starts, True)
    cycle_starts = np.insert(cycle_starts, 0, True)
    cycle_starts[is_neg_freq] = False
    cycle_label = np.cumsum(cycle_starts)

    # get the "bad" theta cycles and use this to mask
    # the lfp data
    is_bad = get_bad_cycles(
        freq_phase.filt_sig,
        is_neg_freq,
        cycle_label,
        min_power_percent_threshold,
        min_theta,
        max_theta,
        fs,
    )
    phase = np.ma.MaskedArray(phase, mask=is_bad)
    cycle_label = np.ma.MaskedArray(cycle_label, mask=is_bad)
    lfp = np.ma.MaskedArray(freq_phase.filt_sig, mask=is_bad)
    return phase, cycle_label, lfp


def get_cross_wavelet(
    theta_phase: np.ndarray,
    theta_lfp: np.ndarray,
    gamma_lfp: np.ndarray,
    fs: float,
    **kwargs,
):
    """
    Get the cross wavelet transform between the theta and gamma LFP signals
    """
    # get some args
    min_lfp_chunk_secs = kwargs.pop("min_lfp_chunk_secs", 0.1)
    min_lfp_len = int(min_lfp_chunk_secs / (1 / fs))

    # get the indices of the minimum phases
    min_phase_idx = signal.argrelmin(theta_phase)[0]

    _s = [
        slice(min_phase_idx[i - 1], min_phase_idx[i])
        for i in range(1, len(min_phase_idx))
    ]
    slices = [ss for ss in _s if (ss.stop - ss.start > min_lfp_len)]

    all_spectrograms = np.zeros(shape=(len(slices), 100, 360)) * np.nan

    for i, i_slice in enumerate(slices):
        i_phase = theta_phase[i_slice]
        breakpoint()
        xwt = pycwt.xwt(
            theta_lfp[i_slice],
            gamma_lfp[i_slice],
            1 / fs,
            dj=1 / 6,
            s0=20 * (1 / fs),
            # J=-1,
        )
        power = np.abs(xwt[0]) ** 2

        # i_phase ranges from -pi to +pi
        # change to degrees and 0 to 360
        i_phase = np.degrees(i_phase)
        i_phase = np.remainder(i_phase + 180, 360).astype(int)
        i_spectrogram = np.zeros(shape=(np.shape, 360)) * np.nan
        i_spectrogram[:, i_phase] = power
        all_spectrograms[i, :, :] = i_spectrogram

    return all_spectrograms, xwt[-2]


def get_theta_cycle_spectogram(
    phase: np.ndarray,
    cycle_label: np.ndarray,
    filt_lfp: np.ndarray,
    lfp: np.ndarray,
    fs: float,
    **kwargs,
):
    """
    Get a spectrogram of the theta cycles in the LFP
    """
    # get some args
    min_lfp_chunk_secs = kwargs.pop("min_lfp_chunk_secs", 0.1)
    min_lfp_len = int(min_lfp_chunk_secs / (1 / fs))

    # get the indices of the minimum phases
    min_phase_idx = signal.argrelmin(phase)[0]

    _s = [
        slice(min_phase_idx[i - 1], min_phase_idx[i])
        for i in range(1, len(min_phase_idx))
    ]
    slices = [ss for ss in _s if (ss.stop - ss.start > min_lfp_len)]

    all_spectrograms = np.zeros(shape=(len(slices), 100, 360)) * np.nan

    wavelet = "cmor1.0-1.0"
    scales = np.geomspace(2, 140, num=100)

    for i, i_slice in enumerate(slices):
        i_phase = phase[i_slice]
        cwtmatr, freqs = pywt.cwt(
            lfp[i_slice], scales, wavelet, sampling_period=1 / fs, method="fft"
        )
        power = np.abs(cwtmatr) ** 2

        # i_phase ranges from -pi to +pi
        # change to degrees and 0 to 360
        i_phase = np.degrees(i_phase)
        i_phase = np.remainder(i_phase + 180, 360).astype(int)
        i_spectrogram = np.zeros(shape=(len(scales), 360)) * np.nan
        i_spectrogram[:, i_phase] = power
        all_spectrograms[i, :, :] = i_spectrogram

    return all_spectrograms, freqs


def detect_oscillation_episodes(lfp: np.ndarray, fs: float):
    scales = np.geomspace(2, 140, num=100)
    cwtmatr, freqs = pywt.cwt(
        lfp, scales, wavelet="cmor1.0-1.0", sampling_period=1 / fs
    )
    power = np.abs(cwtmatr) ** 2

    freq_band = (20, 40)
    freq_idx = np.where(np.logical_and(freqs >= freq_band[0], freqs <= freq_band[1]))[0]
    mean_power = np.mean(power[freq_idx, :], axis=0)
    power_threshold = np.percentile(mean_power, 97.72)

    is_high_power = mean_power >= power_threshold

    values, starts, lengths = find_runs(is_high_power)

    # cut 160ms windows around the centres of the high power episodes
    half_win = int(0.08 * fs)
    episode_slices = []
    for v, s, l in zip(values, starts, lengths):
        if v:
            c = s + l // 2
            episode_slices.append(
                slice(max(0, c - half_win), min(len(lfp), c + half_win))
            )

    # make sure each episode is separated by at least 100ms
    min_separation = int(0.1 * fs)
    filtered_slices = []
    if len(episode_slices) > 0:
        filtered_slices.append(episode_slices[0])
        for es in episode_slices[1:]:
            if es.start - filtered_slices[-1].stop >= min_separation:
                filtered_slices.append(es)

    # get the indices of the max amplitude within each episode
    peak_indices = []
    for es in filtered_slices:
        ep = lfp[es]
        peak_idx = np.argmax(np.abs(ep))
        peak_indices.append(es.start + peak_idx)

    # extract 400ms windows around each peak
    half_win = int(0.2 * fs)
    final_slices = []
    for pi in peak_indices:
        final_slices.append(slice(max(0, pi - half_win), min(len(lfp), pi + half_win)))

    return final_slices, freqs
