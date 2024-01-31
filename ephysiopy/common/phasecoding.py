import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
from ephysiopy.common.binning import RateMap
from ephysiopy.common.ephys_generic import PosCalcsGeneric
from ephysiopy.common.rhythmicity import LFPOscillations
from ephysiopy.common.utils import bwperim
from ephysiopy.common.utils import count_runs_and_unique_numbers
from scipy import ndimage, optimize, signal
from scipy.stats import norm
from collections import defaultdict


def labelledCumSum(X, L):
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
    S[L == 0] = np.nan
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

    k = signal.gaussian(kernelLen, kernelSig)
    bins = np.arange(-179.5, 180, 1)
    phaseDist, _ = np.histogram(spkPhase / np.pi * 180, bins=bins)
    phaseDist = ndimage.convolve(phaseDist, k)
    phaseMin = bins[
        int(np.ceil(np.nanmean(np.nonzero(
            phaseDist == np.min(phaseDist))[0])))
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
    rho = (
        4
        * (A * B - C * D)
        / np.sqrt((n**2 - E**2 - F**2) * (n**2 - G**2 - H**2))
    )
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
    rho = (
        4
        * (A * B - C * D)
        / np.sqrt((n**2 - E**2 - F**2) * (n**2 - G**2 - H**2))
    )
    return rho


def circCircCorrTLinear(theta,
                        phi,
                        k=1000,
                        alpha=0.05,
                        hyp=0,
                        conf=True):
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
            rho_boot
            - (1 / np.sqrt(n)) * rho_jack_std *
            norm.ppf(alpha / 2, (0, 1))[0],
            rho_boot
            + (1 / np.sqrt(n)) * rho_jack_std *
            norm.ppf(alpha / 2, (0, 1))[0],
        )
    elif conf and k and n < 25 and n > 4:
        from sklearn.utils import resample

        # set up the bootstrapping parameters
        boot_samples = []
        for i in range(k):
            theta_sample = resample(theta, replace=True)
            phi_sample = resample(phi, replace=True)
            boot_samples.append(ccc(theta_sample, phi_sample))
        rho_boot = np.mean(boot_samples)
        # confidence intervals
        p = ((1.0 - alpha) / 2.0) * 100
        lower = max(0.0, np.percentile(boot_samples, p))
        p = (alpha + ((1.0 - alpha) / 2.0)) * 100
        upper = min(1.0, np.percentile(boot_samples, p))

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

    rho_sim = (
        4
        * (A * B - C * D)
        / np.sqrt((n**2 - E**2 - F**2) * (n**2 - G**2 - H**2))
    )

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

    slope = optimize.fminbound(
        _cost, -1 * max_slope, max_slope, args=(xn, tn))
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
    "spike_sample_rate": 96000,
    "cms_per_bin": 1,  # bin size gets calculated in Ratemap
    "ppm": 400,
    "field_smoothing_kernel_len": 51,
    "field_smoothing_kernel_sigma": 5,
    # fractional limit of field peak to restrict fields with
    "field_threshold": 0.35,
    # field threshold percent - fed into fieldcalcs.local_threshold as prc
    "field_threshold_percent": 65,
    # fractional limit for restricting fields at environment edges
    "area_threshold": np.nan,
    "bins_per_cm": 2,
    "convert_xy_2_cm": False,
    # defines start/ end of theta cycle
    "allowed_min_spike_phase": np.pi,
    # percentile power below which theta cycles are rejected
    "min_power_percent_threshold": 0,
    # bandwidth of theta in bins - NOT SURE ABOUT THIS FOR DIFFERENT SAMPLING
    # FREQS
    # IE NOT AXONA
    "allowed_theta_len": [20, 42],
    # AND THIS
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
        lfp_sig: np.array,
        lfp_fs: int,
        xy: np.array,
        spike_ts: np.array,
        pos_ts: np.array,
        pp_config: dict = phase_precession_config,
        regressors=None
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
        self.phaseAdj = None

        self.update_position(xy, self.ppm, cm=self.convert_xy_2_cm)
        self.update_rate_map()

        spk_times_in_pos_samples = self.getSpikePosIndices(spike_ts)
        spk_weights = np.bincount(
            spk_times_in_pos_samples, minlength=len(self.pos_ts))
        self.spk_times_in_pos_samples = spk_times_in_pos_samples
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

    def update_regressors(self, reg_keys: list = None):
        '''
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
        '''
        if reg_keys is None:
            reg_keys = all_regressors
        else:
            assert all([k in all_regressors for k in reg_keys])

        # Create a dict to hold the stats values for
        # each regressor
        stats_dict = {
            "values": None,
            "pha": None,
            "slope": None,
            "intercept": None,
            "cor": None,
            "p": None,
            "cor_boot": None,
            "p_shuffled": None,
            "ci": None,
            "reg": None,
        }
        self.regressors = {}
        self.regressors = defaultdict(
            lambda: stats_dict.copy(), self.regressors)
        [self.regressors[k] for k in reg_keys]
        # each of the regressors in regressor_keys is a key with a value
        # of stats_dict

    def update_regressor_values(self, key, values):
        self.regressors[key]["values"] = values

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
        R = RateMap(self.PosData.xy, self.PosData.dir,
                    self.PosData.speed, xyInCms=self.convert_xy_2_cm)
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

        See Also:
            ephysiopy.common.eegcalcs.phasePrecession.partitionFields()
            ephysiopy.common.eegcalcs.phasePrecession.getPosProps()
            ephysiopy.common.eegcalcs.phasePrecession.getThetaProps()
            ephysiopy.common.eegcalcs.phasePrecession.getSpikeProps()
            ephysiopy.common.eegcalcs.phasePrecession._ppRegress()
        """

        # Partition fields
        peaksXY, _, labels, _ = self.partitionFields(plot=True)

        # split into runs
        posD, runD = self.getPosProps(
            labels, peaksXY, laserEvents=laserEvents, plot=True
        )

        # get theta cycles, amplitudes, phase etc
        self.getThetaProps()

        # get the indices of spikes for various metrics such as
        # theta cycle, run etc
        spkD = self.getSpikeProps(
            posD["runLabel"], runD["meanDir"], runD["runDurationInPosBins"]
        )

        # Do the regressions
        self._ppRegress(spkD)

        # Plot the results
        n_regressors = len(self.get_regressors())
        fig, ax = plt.subplots(1, n_regressors, figsize=(10, 10))
        if n_regressors == 1:
            ax = [ax]
        for ra in zip(self.get_regressors(), ax):
            self.plotRegressor(ra[0], ra[1])

    def partitionFields(self, ftype="g", plot=False, **kwargs):
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
        rmap, (ye, xe) = self.RateMap.getMap(self.spk_weights)
        rmap = rmap.T
        nan_idx = np.isnan(rmap)
        rmap[nan_idx] = 0
        # start image processing:
        # get some markers
        from ephysiopy.common import fieldcalcs

        markers = fieldcalcs.local_threshold(
            rmap, prc=self.field_threshold_percent)
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
            ndimage.maximum_position(
                rmap, labels=labels, index=fieldId)
        ).astype(int)
        # COMCoords = np.array(
        #     ndimage.center_of_mass(
        #         rmap, labels=labels, index=fieldId)
        # ).astype(int)
        peaksXY = np.vstack((xe[peakCoords[:, 0]], ye[peakCoords[:, 1]])).T
        # find the peak rate at each of the centre of the detected fields to
        # subsequently threshold the field at some fraction of the peak value
        peakRates = rmap[peakCoords[:, 0], peakCoords[:, 1]]
        fieldThresh = peakRates * self.field_threshold
        rmFieldMask = np.zeros_like(rmap)
        for fid in fieldId:
            f = labels[peakCoords[fid - 1, 0], peakCoords[fid - 1, 1]]
            rmFieldMask[labels == f] = rmap[labels == f] > fieldThresh[f - 1]
        labels[~rmFieldMask.astype(bool)] = 0
        # peakBinInds = np.ceil(peakCoords)
        # re-order some vars to get into same format as fieldLabels
        peakLabels = labels[peakCoords[:, 0], peakCoords[:, 1]]
        peaksXY = peaksXY[peakLabels - 1, :]
        peaksRate = peakRates[peakLabels - 1]
        # peakBinInds = peakBinInds[peakLabels-1, :]
        # peaksXY = peakCoords - np.min(xy, 1)

        # if ~np.isnan(self.area_threshold):
        #     # TODO: this needs fixing so sensible values are used and the
        #     # modified bool array is propagated correctly ie makes
        #     # sense to have a function that applies a bool array to whatever
        #     # arrays are used as output and call it in a couple of places
        #     # areaInBins = self.area_threshold * self.binsPerCm
        #     lb = ndimage.label(markers)[0]
        #     rp = skimage.measure.regionprops(lb)
        #     for reg in rp:
        #         print(reg.filled_area)
        #     markers = skimage.morphology.remove_small_objects(
        #         lb, min_size=4000, connectivity=4, in_place=True)
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(211)
            ax.pcolormesh(ye, xe, rmap, cmap=matplotlib.cm.get_cmap("jet"),
                          edgecolors="face")
            ax.set_title("Smoothed ratemap + peaks")
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set_aspect("equal")
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.plot(peaksXY[:, 1], peaksXY[:, 0], "ko")
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)

            ax = fig.add_subplot(212)
            ax.imshow(labels, interpolation="nearest", origin="lower")
            ax.set_title("Labelled restricted fields")
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set_aspect("equal")

        return peaksXY, peaksRate, labels, rmap

    def getPosProps(
            self,
            labels,
            peaksXY,
            laserEvents=None,
            plot=False,
            **kwargs):
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
        spkPosInd[spkPosInd > len(xy.T)] = len(xy.T) - 1
        nPos = xy.shape[1]
        xy_old = xy.copy()
        xydir = np.squeeze(xydir)
        xydir_old = xydir.copy()

        rmap, (ye, xe) = self.RateMap.getMap(
            self.spk_weights
        )
        rmap = rmap.T

        # The large number of bins combined with the super-smoothed ratemap
        # will lead to fields labelled with lots of small holes in. Fill those
        # gaps in here and calculate the perimeter of the fields based on that
        # labelled image
        labels, n_labels = ndimage.label(ndimage.binary_fill_holes(labels))

        rmap[np.isnan(rmap)] = 0
        xBins = np.digitize(xy[0], xe[:-1])
        yBins = np.digitize(xy[1], ye[:-1])
        fieldLabel = labels[xBins - 1, yBins - 1]
        fl_counts, fl_bins = np.histogram(fieldLabel, bins=np.unique(labels))
        for i, fl in enumerate(fl_bins[1::]):
            print("Field {} has {} samples".format(i, fl_counts[i]))

        fieldPerimMask = bwperim(labels)
        fieldPerimYBins, fieldPerimXBins = np.nonzero(fieldPerimMask)
        fieldPerimX = ye[fieldPerimXBins]
        fieldPerimY = xe[fieldPerimYBins]
        fieldPerimXY = np.vstack((fieldPerimX, fieldPerimY))
        peaksXYBins = np.array(
            ndimage.maximum_position(
                rmap, labels=labels, index=np.unique(labels)[1::]
            )
        ).astype(int)
        peakY = xe[peaksXYBins[:, 0]]
        peakX = ye[peaksXYBins[:, 1]]
        peaksXY = np.vstack((peakX, peakY)).T

        posRUnsmthd = np.zeros((nPos)) * np.nan
        posAngleFromPeak = np.zeros_like(posRUnsmthd) * np.nan
        perimAngleFromPeak = np.zeros_like(fieldPerimMask) * np.nan
        for i, peak in enumerate(peaksXY):
            i = i + 1
            # grab each fields perim coords and the pos samples within it
            y_ind, x_ind = np.nonzero(fieldPerimMask == i)
            thisFieldPerim = np.array([xe[x_ind], ye[y_ind]])
            if thisFieldPerim.any():
                this_xy = xy[:, fieldLabel == i]
                # calculate angle from the field peak for each point on the
                # perim and each pos sample that lies within the field
                thisPerimAngle = np.arctan2(
                    thisFieldPerim[1, :] -
                    peak[1], thisFieldPerim[0, :] - peak[0]
                )
                thisPosAngle = np.arctan2(
                    this_xy[1, :] - peak[1], this_xy[0, :] - peak[0]
                )
                posAngleFromPeak[fieldLabel == i] = thisPosAngle

                perimAngleFromPeak[fieldPerimMask
                                   == i] = thisPerimAngle
                # for each pos sample calculate which point on the perim is
                # most colinear with the field centre - see _circ_abs for more
                thisAngleDf = circ_abs(
                    thisPerimAngle[:, np.newaxis] - thisPosAngle[np.newaxis, :]
                )
                thisPerimInd = np.argmin(thisAngleDf, 0)
                # calculate the distance to the peak from pos and the min perim
                # point and calculate the ratio (r - see OUtputs for method)
                tmp = this_xy.T - peak.T
                distFromPos2Peak = np.hypot(tmp[:, 0], tmp[:, 1])
                tmp = thisFieldPerim[:, thisPerimInd].T - peak.T
                distFromPerim2Peak = np.hypot(tmp[:, 0], tmp[:, 1])
                posRUnsmthd[fieldLabel == i] = distFromPos2Peak / \
                    distFromPerim2Peak
        # the skimage find_boundaries method combined with the labelled mask
        # strive to make some of the values in thisDistFromPos2Peak larger than
        # those in thisDistFromPerim2Peak which means that some of the vals in
        # posRUnsmthd larger than 1 which means the values in xy_new later are
        # wrong - so lets cap any value > 1 to 1. The same cap is applied later
        # to rho when calculating the angular values. Print out a warning
        # message letting the user know how many values have been capped
        print(
            "\n\n{:.2%} posRUnsmthd values have been capped to 1\n\n".format(
                np.sum(posRUnsmthd >= 1) / posRUnsmthd.size
            )
        )
        runs_count, _ = count_runs_and_unique_numbers(fieldLabel)
        for k in runs_count.keys():
            print(f"Field {k} has {runs_count[k]} runs through it")
        posRUnsmthd[posRUnsmthd >= 1] = 1
        # label non-zero contiguous runs with a unique id
        runLabel = labelContigNonZeroRuns(fieldLabel)
        isRun = runLabel > 0
        runStartIdx = getLabelStarts(runLabel)
        runEndIdx = getLabelEnds(runLabel)
        # find runs that are too short, have low speed or too few spikes
        runsSansSpikes = np.ones(len(runStartIdx), dtype=bool)
        spkRunLabels = runLabel[spkPosInd] - 1
        runsSansSpikes[spkRunLabels[spkRunLabels > 0]] = False
        k = signal.boxcar(self.speed_smoothing_window_len) / float(
            self.speed_smoothing_window_len
        )
        spdSmthd = signal.convolve(np.squeeze(spd), k, mode="same")
        runDurationInPosBins = runEndIdx - runStartIdx + 1
        runsMinSpeed = []
        runId = np.unique(runLabel)[1::]
        for run in runId:
            runsMinSpeed.append(np.min(spdSmthd[runLabel == run]))
        runsMinSpeed = np.array(runsMinSpeed)
        badRuns = np.logical_or(
            np.logical_or(
                runsMinSpeed < self.minimum_allowed_run_speed,
                runDurationInPosBins < self.minimum_allowed_run_duration,
            ),
            runsSansSpikes,
        )
        badRuns = np.squeeze(badRuns)
        runLabel = applyFilter2Labels(~badRuns, runLabel)
        runStartIdx = runStartIdx[~badRuns]
        runEndIdx = runEndIdx[~badRuns]  # + 1
        runsMinSpeed = runsMinSpeed[~badRuns]
        runDurationInPosBins = runDurationInPosBins[~badRuns]
        isRun = runLabel > 0

        # calculate mean and std direction for each run
        runComplexMnDir = np.squeeze(np.zeros_like(runStartIdx))
        np.add.at(
            runComplexMnDir,
            runLabel[isRun] - 1,
            np.exp(1j * (xydir[isRun] * (np.pi / 180))),
        )
        meanDir = np.angle(runComplexMnDir)  # circ mean
        tortuosity = 1 - np.abs(runComplexMnDir) / runDurationInPosBins

        # caculate angular distance between the runs main direction and the
        # pos's direction to the peak centre
        posPhiUnSmthd = np.ones_like(fieldLabel) * np.nan
        posPhiUnSmthd[isRun] = posAngleFromPeak[isRun] - \
            meanDir[runLabel[isRun] - 1]

        # smooth r and phi in cartesian space
        # convert to cartesian coords first
        posXUnSmthd, posYUnSmthd = pol2cart(posRUnsmthd, posPhiUnSmthd)
        posXYUnSmthd = np.vstack((posXUnSmthd, posYUnSmthd))

        # filter each run with filter of appropriate length
        filtLen = np.squeeze(
            np.floor((runEndIdx - runStartIdx + 1)
                     * self.ifr_smoothing_constant)
        )
        xy_new = np.zeros_like(xy_old) * np.nan
        for i in range(len(runStartIdx)):
            if filtLen[i] > 2:
                filt = signal.firwin(
                    int(filtLen[i] - 1),
                    cutoff=self.spatial_lowpass_cutoff /
                    self.pos_sample_rate * 2,
                    window="blackman",
                )
                xy_new[:, runStartIdx[i]: runEndIdx[i]] = signal.filtfilt(
                    filt, [1], posXYUnSmthd[:,
                                            runStartIdx[i]: runEndIdx[i]],
                    axis=1
                )

        r, phi = cart2pol(xy_new[0], xy_new[1])
        r[r > 1] = 1

        # calculate the direction of the smoothed data
        xydir_new = np.arctan2(np.diff(xy_new[1]), np.diff(xy_new[0]))
        xydir_new = np.append(xydir_new, xydir_new[-1])
        xydir_new[runEndIdx] = xydir_new[runEndIdx - 1]

        # project the distance value onto the current direction
        if "pos_d_currentdir" in self.regressors.keys():
            d_currentdir = r * np.cos(xydir_new - phi)
            self.update_regressor_values("pos_d_currentdir", d_currentdir)

        # calculate the cumulative distance travelled on each run
        if "pos_d_cum" in self.regressors.keys():
            dr = np.sqrt(np.diff(np.power(r, 2), 1))
            d_cumulative = labelledCumSum(np.insert(dr, 0, 0), runLabel)
            self.update_regressor_values("pos_d_cum", d_cumulative)

        # calculate cumulative sum of the expected normalised firing rate
        if "pos_exptdRate_cum" in self.regressors.keys():
            exptdRate_cumulative = labelledCumSum(1 - r, runLabel)
            self.update_regressor_values(
                "pos_exptdRate_cum", exptdRate_cumulative)

        # direction projected onto the run mean direction is just the x coord
        if "pos_d_meanDir" in self.regressors.keys():
            d_meandir = xy_new[0]
            self.update_regressor_values("pos_d_meanDir", d_meandir)

        # smooth binned spikes to get an instantaneous firing rate
        # set up the smoothing kernel
        if "pos_instFR" in self.regressors.keys():
            kernLenInBins = np.round(
                self.ifr_kernel_len * self.bins_per_second)
            kernSig = self.ifr_kernel_sigma * self.bins_per_second
            k = signal.gaussian(kernLenInBins, kernSig)
            # get a count of spikes to smooth over
            spkCount = np.bincount(spkPosInd, minlength=nPos)
            # apply the smoothing kernel
            instFiringRate = signal.convolve(spkCount, k, mode="same")
            instFiringRate[~isRun] = np.nan
            self.update_regressor_values("pos_instFR", instFiringRate)

        # find time spent within run
        if "pos_timeInRun" in self.regressors.keys():
            time = np.ones(nPos)
            time = labelledCumSum(time, runLabel)
            timeInRun = time / self.pos_sample_rate
            self.update_regressor_values("pos_timeInRun", timeInRun)

        fieldNum = fieldLabel[runStartIdx]
        mnSpd = np.squeeze(np.zeros_like(fieldNum))
        np.add.at(mnSpd, runLabel[isRun] - 1, spd[isRun])
        nPts = np.bincount(runLabel[isRun] - 1, minlength=len(mnSpd))
        mnSpd = mnSpd / nPts
        centralPeripheral = np.squeeze(np.zeros_like(fieldNum))
        np.add.at(centralPeripheral, runLabel[isRun] - 1, xy_new[1, isRun])
        centralPeripheral = centralPeripheral / nPts
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(221)
            ax.plot(xy_new[0], xy_new[1])
            ax.set_title("Unit circle x-y")
            ax.set_aspect("equal")
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])

            ax = fig.add_subplot(222)
            ax.plot(fieldPerimX, fieldPerimY, "k.")
            ax.set_title("Field perim and\n laser on events")
            ax.plot(xy[1, fieldLabel > 0], xy[0, fieldLabel > 0], "y.")
            if laserEvents is not None:
                validOns = np.setdiff1d(
                    laserEvents, np.nonzero(~np.isnan(r))[0])
                ax.plot(xy[0, validOns], xy[1, validOns], "rx")
            ax.set_aspect("equal")

            angleCMInd = np.round(perimAngleFromPeak / np.pi * 180) + 180
            angleCMInd[angleCMInd == 0] = 360
            im = np.zeros_like(fieldPerimMask)
            for fl in fl_bins[1::]:
                xi, yi = np.nonzero(fieldPerimMask == fl)
                im[xi, yi] = angleCMInd[xi, yi]
            imM = np.ma.MaskedArray(im, mask=~fieldPerimMask, copy=True)
            #############################################
            # create custom colormap
            cmap = plt.cm.get_cmap("jet_r")
            cmaplist = [cmap(i) for i in range(cmap.N)]
            cmaplist[0] = (1, 1, 1, 1)
            cmap = cmap.from_list("Runvals cmap", cmaplist, cmap.N)
            bounds = np.linspace(0, 1.0, 100)
            norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
            # add the runs through the fields
            runVals = np.zeros_like(im)
            runVals[xBins[isRun] - 1, yBins[isRun] - 1] = r[isRun]
            runVals = runVals
            ax = fig.add_subplot(223)
            imm = ax.imshow(
                runVals,
                cmap=cmap,
                norm=norm,
                origin="lower",
                interpolation="nearest"
            )
            plt.colorbar(imm, orientation="horizontal")
            ax.set_aspect("equal")
            # add a custom colorbar for colors in runVals

            # create a custom colormap for the plot
            cmap = matplotlib.cm.get_cmap("hsv")
            cmaplist = [cmap(i) for i in range(cmap.N)]
            cmaplist[0] = (1, 1, 1, 1)
            cmap = cmap.from_list("Perim cmap", cmaplist, cmap.N)
            bounds = np.linspace(0, 360, cmap.N)
            norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

            imm = ax.imshow(
                imM,
                cmap=cmap,
                norm=norm,
                origin="lower",
                interpolation="nearest"
            )
            plt.colorbar(imm)
            ax.set_title("Runs by distance and angle")
            ax.plot(peaksXYBins[:, 1], peaksXYBins[:, 0], "ko")
            ax.set_xlim(0, im.shape[1])
            ax.set_ylim(0, im.shape[0])
            #############################################
            ax = fig.add_subplot(224)
            ax.imshow(rmap, origin="lower", interpolation="nearest")
            ax.set_aspect("equal")
            ax.set_title("Smoothed ratemap")

        posKeys = (
            "xy",
            "xydir",
            "r",
            "phi",
            "xy_old",
            "xydir_old",
            "fieldLabel",
            "runLabel",
            # "d_currentdir",
            # "d_cumulative",
            # "exptdRate_cumulative",
            # "d_meandir",
            # "instFiringRate",
            # "timeInRun",
            "fieldPerimMask",
            "perimAngleFromPeak",
            "posAngleFromPeak",
        )
        runsKeys = (
            "runStartIdx",
            "runEndIdx",
            "runDurationInPosBins",
            "runsMinSpeed",
            "meanDir",
            "tortuosity",
            "mnSpd",
            "centralPeripheral",
        )
        posDict = dict.fromkeys(posKeys, np.nan)
        # neat trick: locals is a dict that holds all locally scoped variables
        for thiskey in posDict.keys():
            posDict[thiskey] = locals()[thiskey]
        runsDict = dict.fromkeys(runsKeys, np.nan)
        for thiskey in runsDict.keys():
            runsDict[thiskey] = locals()[thiskey]
        return posDict, runsDict

    def getThetaProps(self, **kwargs):
        spikeTS = self.spike_ts
        phase = self.phase
        filteredEEG = self.filteredEEG
        oldAmplt = filteredEEG.copy()
        # get indices of spikes into eeg
        spkEEGIdx = np.ceil(
            spikeTS * (self.lfp_sample_rate / self.pos_sample_rate)
        ).astype(int)
        spkEEGIdx[spkEEGIdx > len(phase)] = len(phase) - 1
        spkCount = np.bincount(spkEEGIdx, minlength=len(phase))
        spkPhase = phase[spkEEGIdx]
        minSpikingPhase = getPhaseOfMinSpiking(spkPhase)
        phaseAdj = fixAngle(
            phase - minSpikingPhase *
            (np.pi / 180) + self.allowed_min_spike_phase
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
        cycleTotBinCount = np.bincount(cycleLabel)
        cycleHasBadLen = np.logical_or(
            cycleTotBinCount > self.allowed_theta_len[1],
            cycleTotBinCount < self.allowed_theta_len[0],
        )

        # remove data calculated as 'bad'
        isBadCycle = np.logical_or(cycleHasBadLen, cycleHasBadPow)
        isInBadCycle = isBadCycle[cycleLabel]
        isBad = np.logical_or(isInBadCycle, isNegFreq)
        phaseAdj[isBad] = np.nan
        self.phaseAdj = phaseAdj
        ampAdj = filteredEEG.copy()
        ampAdj[isBad] = np.nan
        cycleLabel[isBad] = 0
        self.cycleLabel = cycleLabel
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

        spikeTS = self.spike_ts
        xy = self.RateMap.xy
        phase = self.phaseAdj
        cycleLabel = self.cycleLabel
        spkEEGIdx = np.ceil(
            spikeTS * self.lfp_sample_rate)
        spkEEGIdx[spkEEGIdx > len(phase)] = len(phase) - 1
        spkEEGIdx = spkEEGIdx.astype(int)
        spkPosIdx = np.ceil(spikeTS * self.pos_sample_rate)
        spkPosIdx[spkPosIdx > xy.shape[1]] = xy.shape[1] - 1
        spkRunLabel = runLabel[spkPosIdx.astype(int)]
        thetaCycleLabel = cycleLabel[spkEEGIdx.astype(int)]

        # build mask true for spikes in 1st half of cycle
        firstInTheta = thetaCycleLabel[:-1] != thetaCycleLabel[1::]
        firstInTheta = np.insert(firstInTheta, 0, True)
        lastInTheta = firstInTheta[1::]
        # calculate two kinds of numbering for spikes in a run
        numWithinRun = labelledCumSum(
            np.ones_like(spkPosIdx), spkRunLabel)
        thetaBatchLabelInRun = labelledCumSum(
            firstInTheta.astype(float), spkRunLabel
        )

        spkCount = np.bincount(
            spkRunLabel[spkRunLabel > 0], minlength=len(meanDir))
        rateInPosBins = spkCount[1::] / durationInPosBins.astype(float)
        # update the regressor dict from __init__ with relevant values
        if "spk_numWithinRun" in self.regressors.keys():
            self.update_regressor_values("spk_numWithinRun", numWithinRun)
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
        return spkDict

    def _ppRegress(self, spkDict, whichSpk="first", **kwargs):

        phase = self.phaseAdj
        newSpkRunLabel = spkDict["spkRunLabel"].copy()
        # TODO: need code to deal with splitting the data based on a group of
        # variables
        spkUsed = newSpkRunLabel > 0
        if "first" in whichSpk:
            spkUsed[~spkDict["firstInTheta"]] = False
        elif "last" in whichSpk:
            if len(spkDict["lastInTheta"]) < len(spkDict["spkRunLabel"]):
                spkDict["lastInTheta"] = np.insert(
                    spkDict["lastInTheta"], -1, False)
            spkUsed[~spkDict["lastInTheta"]] = False
        spkPosIdxUsed = spkDict["spkPosIdx"].astype(int)
        # copy self.regressors and update with spk/ pos of interest
        regressors = self.regressors.copy()
        for k in regressors.keys():
            if k.startswith("spk_"):
                self.update_regressor_values(
                    k, regressors[k]["values"][spkUsed])
            elif k.startswith("pos_"):
                self.update_regressor_values(
                    k, regressors[k]["values"][spkPosIdxUsed[spkUsed]]
                )
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
            spkCountPerCycle = np.bincount(
                cycleLabels[goodPhase], minlength=sz)
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
            reg = regressors[k]["values"][np.logical_and(
                goodRegressor, goodPhase)]
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
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = ax
        vals = self.regressors[regressor]["values"]
        pha = self.reg_phase
        ax.plot(vals, pha, 'k.')
        ax.plot(vals, pha + 2 * np.pi, 'k.')
        slope = self.regressors[regressor]["slope"]
        intercept = self.regressors[regressor]["intercept"]
        mm = (0, -2 * np.pi, 2 * np.pi, 4 * np.pi)
        for m in mm:
            ax.plot(
                (-1, 1), (-slope + intercept + m, slope + intercept + m),
                "r", lw=3
            )
        ax.set_xlim(-1, 1)
        ax.set_ylim(-np.pi, 3 * np.pi)
        ax.set_title(regressor)
        ax.set_ylabel("Phase")
        ax.set_xlabel("Normalised position")
        return ax

    def plotPPRegression(self,
                         regressorDict,
                         regressor2plot="pos_d_cum",
                         ax=None):

        t = self.getLFPPhaseValsForSpikeTS()
        x = self.RateMap.xy[0, self.spk_times_in_pos_samples]
        from ephysiopy.common import fieldcalcs

        rmap, (ye, xe) = self.RateMap.getMap(self.spk_weights)
        rmap = rmap.T
        label = fieldcalcs.field_lims(rmap)
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
            ax.plot((-1, 1), (-slope + intercept + m,
                    slope + intercept + m), "r", lw=3)
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
        ts = self.spk_times_in_pos_samples * (
            self.lfp_sample_rate / self.pos_sample_rate
        )
        ts_idx = np.array(np.floor(ts), dtype=int)
        return self.phase[ts_idx]

# Define a group of static methods for doing various operations on circular
# and labelled data
