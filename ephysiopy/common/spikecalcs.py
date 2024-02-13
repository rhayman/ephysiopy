import os
import warnings
from pathlib import Path
from collections import namedtuple
from astropy.convolution import convolve, Box1DKernel, Gaussian1DKernel
from collections.abc import Sequence
import h5py
import matplotlib.pylab as plt
import numpy as np
from scipy.special import erf
from phylib.io.model import TemplateModel
from scipy import signal, stats
from ephysiopy.common.utils import min_max_norm

from ephysiopy.openephys2py.KiloSort import KiloSortSession


def get_param(waveforms, param="Amp", t=200, fet=1):
    """
    Returns the requested parameter from a spike train as a numpy array

    Args:
        waveforms (numpy array): Shape of array can be nSpikes x nSamples
            OR
            a nSpikes x nElectrodes x nSamples
        param (str): Valid values are:
            'Amp' - peak-to-trough amplitude (default)
            'P' - height of peak
            'T' - depth of trough
            'Vt' height at time t
            'tP' - time of peak (in seconds)
            'tT' - time of trough (in seconds)
            'PCA' - first n fet principal components (defaults to 1)
        t (int): The time used for Vt
        fet (int): The number of principal components
            (use with param 'PCA')
    """
    from scipy import interpolate
    from sklearn.decomposition import PCA

    if param == "Amp":
        return np.ptp(waveforms, axis=-1)
    elif param == "P":
        return np.max(waveforms, axis=-1)
    elif param == "T":
        return np.min(waveforms, axis=-1)
    elif param == "Vt":
        times = np.arange(0, 1000, 20)
        f = interpolate.interp1d(times, range(50), "nearest")
        if waveforms.ndim == 2:
            return waveforms[:, int(f(t))]
        elif waveforms.ndim == 3:
            return waveforms[:, :, int(f(t))]
    elif param == "tP":
        idx = np.argmax(waveforms, axis=-1)
        m = interpolate.interp1d(
            [0, waveforms.shape[-1] - 1], [0, 1 / 1000.0])
        return m(idx)
    elif param == "tT":
        idx = np.argmin(waveforms, axis=-1)
        m = interpolate.interp1d(
            [0, waveforms.shape[-1] - 1], [0, 1 / 1000.0])
        return m(idx)
    elif param == "PCA":
        pca = PCA(n_components=fet)
        if waveforms.ndim == 2:
            return pca.fit(waveforms).transform(waveforms).squeeze()
        elif waveforms.ndim == 3:
            out = np.zeros((waveforms.shape[0], waveforms.shape[1] * fet))
            st = np.arange(0, waveforms.shape[1] * fet, fet)
            en = np.arange(fet, fet + (waveforms.shape[1] * fet), fet)
            rng = np.vstack((st, en))
            for i in range(waveforms.shape[1]):
                if ~np.any(np.isnan(waveforms[:, i, :])):
                    A = np.squeeze(
                        pca.fit(waveforms[:, i, :].squeeze()).transform(
                            waveforms[:, i, :].squeeze()
                        )
                    )
                    if A.ndim < 2:
                        out[:, rng[0, i]:rng[1, i]] = np.atleast_2d(A).T
                    else:
                        out[:, rng[0, i]:rng[1, i]] = A
            return out


def mahal(u, v):
    """
    Returns the L-ratio and Isolation Distance measures calculated on the
    principal components of the energy in a spike matrix.

    Args:
        waveforms (np.ndarray, optional): The waveforms to be processed. If
            None, the function will return None.
        spike_clusters (np.ndarray, optional): The spike clusters to be
            processed.
        cluster_id (int, optional): The ID of the cluster to be processed.
        fet (int, default=1): The feature to be used in the PCA calculation.

    Returns:
        tuple: A tuple containing the L-ratio and Isolation Distance of the
            cluster.

    Raises:
        Exception: If an error occurs during the calculation of the L-ratio or
            Isolation Distance.
    """
    u_sz = u.shape
    v_sz = v.shape
    if u_sz[1] != v_sz[1]:
        warnings.warn("Input size mismatch: \
                        matrices must have same num of columns")
    if v_sz[0] < v_sz[1]:
        warnings.warn("Too few rows: v must have more rows than columns")
    if np.any(np.imag(u)) or np.any(np.imag(v)):
        warnings.warn("No complex inputs are allowed")
    m = np.nanmean(v, axis=0)
    M = np.tile(m, reps=(u_sz[0], 1))
    C = v - np.tile(m, reps=(v_sz[0], 1))
    _, R = np.linalg.qr(C)
    ri = np.linalg.solve(R.T, (u - M).T)
    d = np.sum(ri * ri, 0).T * (v_sz[0] - 1)
    return d


def cluster_quality(waveforms: np.ndarray = None,
                    spike_clusters: np.ndarray = None,
                    cluster_id: int = None,
                    fet: int = 1):
    """
    Returns the L-ratio and Isolation Distance measures calculated
    on the principal components of the energy in a spike matrix.

    Args:
        waveforms (np.ndarray, optional): The waveforms to be processed.
            If None, the function will return None.
        spike_clusters (np.ndarray, optional): The spike clusters to be
            processed.
        cluster_id (int, optional): The ID of the cluster to be processed.
        fet (int, default=1): The feature to be used in the PCA calculation.

    Returns:
        tuple: A tuple containing the L-ratio and Isolation Distance of the
            cluster.

    Raises:
        Exception: If an error occurs during the calculation of the L-ratio or
            Isolation Distance.
    """
    if waveforms is None:
        return None
    nSpikes, nElectrodes, _ = waveforms.shape
    wvs = waveforms.copy()
    E = np.sqrt(np.nansum(waveforms**2, axis=2))
    zeroIdx = np.sum(E, 0) == [0, 0, 0, 0]
    E = E[:, ~zeroIdx]
    wvs = wvs[:, ~zeroIdx, :]
    normdWaves = (wvs.T / E.T).T
    PCA_m = get_param(normdWaves, "PCA", fet=fet)
    badIdx = np.sum(PCA_m, axis=0) == 0
    PCA_m = PCA_m[:, ~badIdx]
    # get mahalanobis distance
    idx = spike_clusters == cluster_id
    nClustSpikes = np.count_nonzero(idx)
    try:
        d = mahal(PCA_m, PCA_m[idx, :])
        # get the indices of the spikes not in the cluster
        M_noise = d[~idx]
        df = np.prod((fet, nElectrodes))
        from scipy import stats

        L = np.sum(1 - stats.chi2.cdf(M_noise, df))
        L_ratio = L / nClustSpikes
        # calculate isolation distance
        if nClustSpikes < nSpikes / 2:
            M_noise.sort()
            isolation_dist = M_noise[nClustSpikes]
        else:
            isolation_dist = np.nan
    except Exception:
        isolation_dist = L_ratio = np.nan
    return L_ratio, isolation_dist


def xcorr(x1: np.ndarray,
          x2=None,
          Trange=None,
          binsize=0.001,
          **kwargs) -> tuple:
    """
    Calculates the ISIs in x1 or x1 vs x2 within a given range

    Args:
        x1, x2 (array_like): The times of the spikes emitted by the
                            cluster(s) in seconds
        Trange (array_like): Range of times to bin up in seconds
                                Defaults to [-0.5, +0.5]
        binsize (float): The size of the bins in seconds

    Returns:
        counts (np.ndarray): The cross-correlogram of the spike trains
            x1 and x2
        bins (np.ndarray): The bins used to calculate the cross-correlogram
    """
    if x2 is None:
        x2 = x1.copy()
    if Trange is None:
        Trange = np.array([-0.5, 0.5])
    if isinstance(Trange, list):
        Trange = np.array(Trange)
    y = []
    irange = x1[:, np.newaxis] + Trange[np.newaxis, :]
    dts = np.searchsorted(x2, irange)
    for i, t in enumerate(dts):
        y.extend((x2[t[0]: t[1]] - x1[i]))
    y = np.array(y, dtype=float)
    counts, bins = np.histogram(y[y != 0],
                                bins=int(np.ptp(Trange)/binsize)+1,
                                range=Trange)
    return counts, bins


def contamination_percent(
        x1: np.ndarray,
        x2: np.ndarray = None,
        **kwargs) -> tuple:
    '''
    Computes the cross-correlogram between two sets of spikes and
    estimates how refractory the cross-correlogram is.

    Args:
        st1 (np.array): The first set of spikes.
        st2 (np.array): The second set of spikes.

    kwargs:
        Anything that can be fed into xcorr above

    Returns:
        Q (float): a measure of refractoriness
        R (float): a second measure of refractoriness
                (kicks in for very low firing rates)

    Notes:
        Taken from KiloSorts ccg.m

        The contamination metrics are calculated based on
        an analysis of the 'shoulders' of the cross-correlogram.
        Specifically, the spike counts in the ranges +/-5-25ms and
        +/-250-500ms are compared for refractoriness
    '''
    if x2 is None:
        x2 = x1.copy()
    c, b = xcorr(x1, x2, **kwargs)
    left = [[-0.05, -0.01]]
    right = [[0.01, 0.051]]
    far = [[-0.5, -0.249], [0.25, 0.501]]

    def get_shoulder(bins, vals):
        all = np.array([np.logical_and(bins >= i[0], bins < i[1])
                        for i in vals])
        return np.any(all, 0)

    inner_left = get_shoulder(b, left)
    inner_right = get_shoulder(b, right)
    outer = get_shoulder(b, far)

    tbin = 1000
    Tr = max(np.concatenate([x1, x2])) - min(np.concatenate([x1, x2]))

    def get_normd_shoulder(idx):
        return np.sum(c[idx[:-1]]) / (len(np.nonzero(idx)[0]) *
                                      tbin * len(x1) * len(x2) / Tr)

    Q00 = get_normd_shoulder(outer)
    Q01 = max(get_normd_shoulder(inner_left),
              get_normd_shoulder(inner_right))

    R00 = max(np.mean(c[outer[:-1]]),
              np.mean(c[inner_left[:-1]]),
              np.mean(c[inner_right[1:]]))

    middle_idx = np.nonzero(b == 0)[0]
    a = c[middle_idx]
    c[middle_idx] = 0
    Qi = np.zeros(10)
    Ri = np.zeros(10)
    # enumerate through the central range of the xcorr
    # saving the same calculation as done above
    for i, t in enumerate(np.linspace(0.001, 0.01, 10)):
        irange = [[-t, t]]
        chunk = get_shoulder(b, irange)
        # compute the same normalized ratio as above;
        # this should be 1 if there is no refractoriness
        Qi[i] = get_normd_shoulder(chunk)  # save the normd prob
        n = np.sum(c[chunk[:-1]])/2
        lam = R00 * i
        # this is tricky: we approximate the Poisson likelihood with a
        # gaussian of equal mean and variance
        # that allows us to integrate the probability that we would see <N
        # spikes in the center of the
        # cross-correlogram from a distribution with mean R00*i spikes
        p = 1/2 * (1 + erf((n - lam)/np.sqrt(2*lam)))

        Ri[i] = p  # keep track of p for each bin size i

    c[middle_idx] = a  # restore the center value of the cross-correlogram
    return c, Qi, Q00, Q01, Ri


# a namedtuple to hold some metrics from the KS run
KSMetaTuple = namedtuple(
    'KSMeta', 'Amplitude group KSLabel ContamPct ')


class SpikeCalcsGeneric(object):
    """
    Deals with the processing and analysis of spike data.
    There should be one instance of this class per cluster in the
    recording session. NB this differs from previous versions of this
    class where there was one instance per recording session and clusters
    were selected by passing in the cluster id to the methods.

    Args:
        spike_times (array_like): The times of spikes in the trial in seconds
        waveforms (np.array, optional): An nSpikes x nChannels x nSamples array

    """

    def __init__(self, spike_times: np.ndarray,
                 cluster: int,
                 waveforms: np.ndarray = None,
                 **kwargs):
        self.spike_times = spike_times  # IN SECONDS
        self._waves = waveforms
        self.cluster = cluster
        self._event_ts = None  # the times that events occured IN SECONDS
        # window, in seconds, either side of the stimulus, to examine
        self._event_window = np.array((-0.050, 0.100))
        self._stim_width = None  # the width, in ms, of the stimulus
        # used to increase / decrease size of bins in psth
        self._secs_per_bin = 0.001
        self._sample_rate = 30000
        self._pos_sample_rate = 50
        self._duration = None
        # these values should be specific to OE data
        self._pre_spike_samples = 16
        self._post_spike_samples = 34
        # values from running KS
        self._ksmeta = KSMetaTuple(None, None, None, None)
        # update the __dict__ attribute with the kwargs
        self.__dict__.update(kwargs)

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        self._sample_rate = value

    @property
    def pos_sample_rate(self):
        return self._pos_sample_rate

    @pos_sample_rate.setter
    def pos_sample_rate(self, value):
        self._pos_sample_rate = value

    @property
    def pre_spike_samples(self):
        return self._pre_spike_samples

    @pre_spike_samples.setter
    def pre_spike_samples(self, value):
        self._pre_spike_samples = int(self._pre_spike_samples)

    @property
    def post_spike_samples(self):
        return self._post_spike_samples

    @post_spike_samples.setter
    def post_spike_samples(self, value):
        self._post_spike_samples = int(self._post_spike_samples)

    def waveforms(self, channel_id: Sequence = None):
        if self._waves is not None:
            if channel_id is None:
                return self._waves[:, :, :]
            else:
                if isinstance(channel_id, int):
                    channel_id = [channel_id]
                return self._waves[:, channel_id, :]
        else:
            return None

    @property
    def n_spikes(self):
        """
        Returns the number of spikes in the cluster

        Returns:
            int: The number of spikes in the cluster
        """
        return len(self.spike_times)

    @property
    def event_ts(self):
        return self._event_ts

    @event_ts.setter
    def event_ts(self, value):
        self._event_ts = value

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, value):
        self._duration = value

    @property
    def KSMeta(self):
        return self._ksmeta

    def update_KSMeta(self, value: dict):
        """
        Takes in a TemplateModel instance from a phy session and
        parses out the relevant metrics for the cluster and places
        into the namedtuple KSMeta
        """
        metavals = []
        for f in KSMetaTuple._fields:
            if f in value.keys():
                metavals.append(value[f][self.cluster])
            else:
                metavals.append(None)
        self._ksmeta = KSMetaTuple(*metavals)

    @property
    def event_window(self):
        return self._event_window

    @event_window.setter
    def event_window(self, value):
        self._event_window = value

    @property
    def stim_width(self):
        return self._stim_width

    @stim_width.setter
    def stim_width(self, value):
        self._stim_width = value

    @property
    def secs_per_bin(self):
        return self._secs_per_bin

    @secs_per_bin.setter
    def secs_per_bin(self, value):
        self._secs_per_bin = value

    def acorr(self, Trange: np.ndarray = None) -> tuple:
        """
        Calculates the autocorrelogram of a spike train

        Args:
            ts (np.ndarray): The spike times
            Trange (np.ndarray): The range of times to calculate the
                autocorrelogram over

        Returns:
            counts (np.ndarray): The autocorrelogram
            bins (np.ndarray): The bins used to calculate the
                autocorrelogram
        """
        return xcorr(self.spike_times, Trange=Trange)

    def trial_mean_fr(self) -> float:
        # Returns the trial mean firing rate for the cluster
        if self.duration is None:
            raise IndexError("No duration provided, give me one!")
        return self.n_spikes / self.duration

    def mean_isi_range(self, isi_range: int) -> float:
        """
        Calculates the mean of the autocorrelation from 0 to n milliseconds
        Used to help classify a neurons type (principal, interneuron etc)

        Args:
            isi_range (int): The range in ms to calculate the mean over

        Returns:
            float: The mean of the autocorrelogram between 0 and n milliseconds
        """
        bins = 201
        trange = np.array((-500, 500))
        counts, bins = self.acorr(Trange=trange)
        mask = np.logical_and(bins > 0, bins < isi_range)
        return np.mean(counts[mask[1:]])

    def mean_waveform(self, channel_id: Sequence = None):
        """
        Returns the mean waveform and sem for a given spike train on a
        particular channel

        Args:
            cluster_id (int): The cluster to get the mean waveform for

        Returns:
            mn_wvs (ndarray): The mean waveforms, usually 4x50 for tetrode
                                recordings
            std_wvs (ndarray): The standard deviations of the waveforms,
                                usually 4x50 for tetrode recordings
        """
        x = self.waveforms(channel_id)
        if x is not None:
            return np.mean(x, axis=0), np.std(x, axis=0)
        else:
            return None

    def psth(self, **kwargs):
        """
        Calculate the PSTH of event_ts against the spiking of a cell

        Args:
            cluster_id (int): The cluster for which to calculate the psth

        Returns:
            x, y (list): The list of time differences between the spikes of
                            the cluster and the events (x) and the trials (y)
        """
        if self._event_ts is None:
            raise Exception("Need some event timestamps! Aborting")
        event_ts = self.event_ts
        event_ts.sort()
        if isinstance(event_ts, list):
            event_ts = np.array(event_ts)

        irange = event_ts[:, np.newaxis] + self.event_window[np.newaxis, :]
        dts = np.searchsorted(self.spike_times, irange)
        x = []
        y = []
        for i, t in enumerate(dts):
            tmp = self.spike_times[t[0]:t[1]] - event_ts[i]
            x.extend(tmp)
            y.extend(np.repeat(i, len(tmp)))
        return x, y

    def psch(
            self, bin_width_secs: float) -> np.ndarray:
        """
        Calculate the peri-stimulus *count* histogram of a cell's spiking
        against event times.

        Args:
            cluster_id (int): The cluster for which to calculate the psth.
            bin_width_secs (float): The width of each bin in seconds.

        Returns:
            result (np.ndarray): Rows are counts of spikes per bin_width_secs.
            Size of columns ranges from self.event_window[0] to
            self.event_window[1] with bin_width_secs steps;
            so x is count, y is "event".
        """
        if self._event_ts is None:
            raise Exception("Need some event timestamps! Aborting")
        event_ts = self.event_ts
        event_ts.sort()
        if isinstance(event_ts, list):
            event_ts = np.array(event_ts)

        irange = event_ts[:, np.newaxis] + self.event_window[np.newaxis, :]
        dts = np.searchsorted(self.spike_times, irange)
        bins = np.arange(self.event_window[0],
                         self.event_window[1], bin_width_secs)
        result = np.zeros(shape=(len(bins)-1, len(event_ts)))
        for i, t in enumerate(dts):
            tmp = self.spike_times[t[0]:t[1]] - event_ts[i]
            indices = np.digitize(tmp, bins=bins)
            counts = np.bincount(indices, minlength=len(bins))
            result[:, i] = counts[1:]
        return result

    def ifr_sp_corr(
        self,
        ts,
        speed,
        minSpeed=2.0,
        maxSpeed=40.0,
        sigma=3,
        shuffle=False,
        nShuffles=100,
        minTime=30,
        plot=False,
    ):
        """
        Calculates the correlation between the instantaneous firing rate and
        speed.

        Args:
            ts (np.array): The times in seconds at which the cluster fired.
            speed (np.array): Instantaneous speed (1 x nSamples).
            minSpeed (float, optional): Speeds below this value are ignored.
                Defaults to 2.0 cm/s as with Kropff et al., 2015.
            maxSpeed (float, optional): Speeds above this value are ignored.
                Defaults to 40.0 cm/s.
            sigma (int, optional): The standard deviation of the gaussian used
                to smooth the spike train. Defaults to 3.
            shuffle (bool, optional): Whether to shuffle the spike train.
                Defaults to False.
            nShuffles (int, optional): The number of resamples to feed into
                the permutation test. Defaults to 9999.
                See scipy.stats.PermutationMethod.
            minTime (int, optional): The minimum time for which the spike
                train should be considered. Defaults to 30.
            plot (bool, optional): Whether to plot the result.
                Defaults to False.
        """
        speed = speed.ravel()
        posSampRate = self.pos_sample_rate
        nSamples = len(speed)
        x1 = np.round(ts * posSampRate).astype(int)
        spk_hist = np.bincount(x1, minlength=nSamples)
        # smooth the spk_hist (which is a temporal histogram) with a 250ms
        # gaussian as with Kropff et al., 2015
        h = signal.windows.gaussian(13, sigma)
        h = h / float(np.sum(h))
        # filter for low speeds
        lowSpeedIdx = speed < minSpeed
        highSpeedIdx = speed > maxSpeed
        speed_filt = speed[~np.logical_or(lowSpeedIdx, highSpeedIdx)]
        spk_hist_filt = spk_hist[~np.logical_or(lowSpeedIdx, highSpeedIdx)]
        spk_sm = signal.filtfilt(h.ravel(), 1, spk_hist_filt)
        sm_spk_rate = spk_sm * posSampRate
        # the permutation test for significance
        rng = np.random.default_rng()
        method = stats.PermutationMethod(
            n_resamples=nShuffles, random_state=rng)
        res = stats.pearsonr(sm_spk_rate, speed_filt, method=method)
        if plot:
            # do some fancy plotting stuff
            _, sp_bin_edges = np.histogram(speed_filt, bins=50)
            sp_dig = np.digitize(speed_filt, sp_bin_edges, right=True)
            spks_per_sp_bin = [
                spk_hist_filt[sp_dig == i] for i in range(len(sp_bin_edges))
            ]
            rate_per_sp_bin = []
            variance_per_sp_bin = []
            for x in spks_per_sp_bin:
                rate_per_sp_bin.append(np.mean(x) * posSampRate)
                variance_per_sp_bin.append((np.var(x) * posSampRate)/len(x))
            rate_filter = signal.gaussian(5, 1.0)
            rate_filter = rate_filter / np.sum(rate_filter)
            binned_spk_rate = signal.filtfilt(rate_filter, 1, rate_per_sp_bin)
            # instead of plotting a scatter plot of the firing rate at each
            # speed bin, plot a log normalised heatmap and overlay results

            spk_binning_edges = np.linspace(
                np.min(sm_spk_rate), np.max(sm_spk_rate), len(sp_bin_edges)
            )
            speed_mesh, spk_mesh = np.meshgrid(sp_bin_edges, spk_binning_edges)
            binned_rate, _, _ = np.histogram2d(
                speed_filt, sm_spk_rate, bins=[sp_bin_edges, spk_binning_edges]
            )
            # blur the binned rate a bit to make it look nicer
            from ephysiopy.common.utils import blurImage

            sm_binned_rate = blurImage(binned_rate, 5)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            from matplotlib.colors import LogNorm

            speed_mesh = speed_mesh[:-1, :-1]
            spk_mesh = spk_mesh[:-1, :-1]
            ax.pcolormesh(
                speed_mesh,
                spk_mesh,
                sm_binned_rate,
                norm=LogNorm(),
                alpha=0.5,
                shading="nearest",
                edgecolors="None",
            )
            # overlay the smoothed binned rate against speed
            ax.plot(sp_bin_edges, binned_spk_rate, "r")
            ax.plot(sp_bin_edges, binned_spk_rate +
                    np.sqrt(variance_per_sp_bin), "r--")
            ax.plot(sp_bin_edges, binned_spk_rate -
                    np.sqrt(variance_per_sp_bin), "r--")
            # do the linear regression and plot the fit too
            # TODO: linear regression is broken ie not regressing the correct
            # variables
            lr = stats.linregress(speed_filt, sm_spk_rate)
            end_point = lr.intercept + \
                ((sp_bin_edges[-1]-sp_bin_edges[0]) * lr.slope)
            ax.plot(
                [np.min(sp_bin_edges), np.max(sp_bin_edges)],
                [lr.intercept, end_point],
                "r--",
            )
            ax.set_xlim(np.min(sp_bin_edges), np.max(sp_bin_edges[-2]))
            ax.set_ylim(0, np.nanmax(binned_spk_rate) * 1.1)
            ax.set_ylabel("Firing rate(Hz)")
            ax.set_xlabel("Running speed(cm/s)")
            ax.set_title(
                "Intercept: {0:.3f}   Slope: {1:.5f}\nPearson: {2:.5f}".format(
                    lr.intercept, lr.slope, lr.rvalue
                )
            )
        # do some shuffling of the data to see if the result is signficant
        if shuffle:
            # shift spikes by at least 30 seconds after trial start and
            # 30 seconds before trial end
            timeSteps = np.random.randint(
                30 * posSampRate, nSamples - (30 * posSampRate), nShuffles
            )
            shuffled_results = []
            for t in timeSteps:
                spk_count = np.roll(spk_hist, t)
                spk_count_filt = spk_count[~np.logical_or(
                    lowSpeedIdx, highSpeedIdx)]
                spk_count_sm = signal.filtfilt(h.ravel(), 1, spk_count_filt)
                shuffled_results.append(stats.pearsonr(
                    spk_count_sm, speed_filt)[0])
            if plot:
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.hist(np.abs(shuffled_results), 20)
                ylims = ax.get_ylim()
                ax.vlines(res, ylims[0], ylims[1], "r")
        return res

    def responds_to_stimulus(self,
                             threshold: float,
                             min_contiguous: int,
                             return_activity: bool = False,
                             return_magnitude: bool = False,
                             **kwargs) -> tuple:
        """
        Checks whether a cluster responds to a laser stimulus.

        Args:
            cluster (int): The cluster to check.
            threshold (float): The amount of activity the cluster needs to go
                beyond to be classified as a responder (1.5 = 50% more or less
                than the baseline activity).
            min_contiguous (int): The number of contiguous samples in the
                post-stimulus period for which the cluster needs to be active
                beyond the threshold value to be classed as a responder.
            return_activity (bool): Whether to return the mean reponse curve.
            return_magnitude (int): Whether to return the magnitude of the
                response. NB this is either +1 for excited or -1 for inhibited.

        Returns:
            responds (bool): Whether the cell responds or not.
            OR
            tuple: responds (bool), normed_response_curve (np.ndarray).
            OR
            tuple: responds (bool), normed_response_curve (np.ndarray),
                response_magnitude (np.ndarray).
        """
        spk_count_by_trial = self.psch(self._secs_per_bin)
        firing_rate_by_trial = spk_count_by_trial / self.secs_per_bin
        mean_firing_rate = np.mean(firing_rate_by_trial, 1)
        # smooth with a moving average
        # check nothing in kwargs first
        if "window_len" in kwargs.keys():
            window_len = kwargs["window_len"]
        else:
            window_len = 5
        if "window" in kwargs.keys():
            window = kwargs["window"]
        else:
            window = "flat"
        if 'flat' in window:
            kernel = Box1DKernel(window_len)
        if 'gauss' in window:
            kernel = Gaussian1DKernel(1, window_len)
        if 'do_smooth' in kwargs.keys():
            do_smooth = kwargs.get('do_smooth')
        else:
            do_smooth = True

        if do_smooth:
            smoothed_binned_spikes = convolve(mean_firing_rate,
                                              kernel,
                                              boundary='wrap')
        else:
            smoothed_binned_spikes = mean_firing_rate
        nbins = np.floor(np.sum(np.abs(self.event_window)) / self.secs_per_bin)
        bins = np.linspace(self.event_window[0],
                           self.event_window[1],
                           int(nbins))
        # normalize all activity by activity in the time before
        # the laser onset
        idx = bins < 0
        normd = min_max_norm(smoothed_binned_spikes,
                             np.min(smoothed_binned_spikes[idx]),
                             np.max(smoothed_binned_spikes[idx]))
        # mask the array outside of a threshold value so that
        # only True values in the masked array are those that
        # exceed the threshold (positively or negatively)
        # the threshold provided to this function is expressed
        # as a % above / below unit normality so adjust that now
        # so it is expressed as a pre-stimulus firing rate mean
        # pre_stim_mean = np.mean(smoothed_binned_spikes[idx])
        # pre_stim_max = pre_stim_mean * threshold
        # pre_stim_min = pre_stim_mean * (threshold-1.0)
        # updated so threshold is double (+ or -) the pre-stim
        # norm (lies between )
        normd_masked = np.ma.masked_inside(normd, -threshold, 1+threshold)
        # find the contiguous runs in the masked array
        # that are at least as long as the min_contiguous value
        # and classify this as a True response
        slices = np.ma.notmasked_contiguous(normd_masked)
        if slices and np.any(np.isfinite(normd)):
            # make sure that slices are within the first 25ms post-stim
            if ~np.any([s.start > 50 and s.start < 75 for s in slices]):
                if not return_activity:
                    return False
                else:
                    if return_magnitude:
                        return False, normd, 0
                    return False, normd
            max_runlength = max([len(normd_masked[s]) for s in slices])
            if max_runlength >= min_contiguous:
                if not return_activity:
                    return True
                else:
                    if return_magnitude:
                        sl = [slc for slc in slices if
                              (slc.stop-slc.start) == max_runlength]
                        mag = [-1 if np.mean(normd[sl[0]]) < 0 else 1][0]
                        return True, normd, mag
                    else:
                        return True, normd
        if not return_activity:
            return False
        else:
            if return_magnitude:
                return False, normd, 0
            return False, normd

    def theta_mod_idx(self):
        """
        Calculates a theta modulation index of a spike train based on the cells
        autocorrelogram.

        Args:
            x1 (np.array): The spike time-series.

        Returns:
            thetaMod (float): The difference of the values at the first peak
            and trough of the autocorrelogram.
        """
        corr, _ = self.acorr()
        # Take the fft of the spike train autocorr (from -500 to +500ms)
        from scipy.signal import periodogram

        freqs, power = periodogram(corr, fs=200, return_onesided=True)
        # Smooth the power over +/- 1Hz
        b = signal.windows.boxcar(3)
        h = signal.filtfilt(b, 3, power)

        # Square the amplitude first
        sqd_amp = h**2
        # Then find the mean power in the +/-1Hz band either side of that
        theta_band_max_idx = np.nonzero(
            sqd_amp == np.max(sqd_amp[np.logical_and(freqs > 6, freqs < 11)])
        )[0][0]
        # Get the mean theta band power - mtbp
        mtbp = np.mean(sqd_amp[theta_band_max_idx-1: theta_band_max_idx+1])
        # Find the mean amplitude in the 2-50Hz range
        other_band_idx = np.logical_and(freqs > 2, freqs < 50)
        # Get the mean in the other band - mobp
        mobp = np.mean(sqd_amp[other_band_idx])
        # Find the ratio of these two - this is the theta modulation index
        return (mtbp - mobp) / (mtbp + mobp)

    def theta_mod_idxV2(self):
        """
        This is a simpler alternative to the theta_mod_idx method in that it
        calculates the difference between the normalized temporal
        autocorrelogram at the trough between 50-70ms and the
        peak between 100-140ms over their sum (data is binned into 5ms bins)

        Measure used in Cacucci et al., 2004 and Kropff et al 2015
        """
        corr, bins = self.acorr()
        # 'close' the right-hand bin
        bins = bins[0:-1]
        # normalise corr so max is 1.0
        corr = corr / float(np.max(corr))
        thetaAntiPhase = np.min(
            corr[np.logical_and(bins > 50/1000., bins < 70/1000.)])
        thetaPhase = np.max(
            corr[np.logical_and(bins > 100/1000., bins < 140/1000.)])
        return (thetaPhase - thetaAntiPhase) / (thetaPhase + thetaAntiPhase)

    def theta_band_max_freq(self):
        """
    Calculates the frequency with the maximum power in the theta band (6-12Hz)
    of a spike train's autocorrelogram.

    This function is used to look for differences in theta frequency in
    different running directions as per Blair.
    See Welday paper - https://doi.org/10.1523/jneurosci.0712-11.2011

    Args:
        x1 (np.ndarray): The spike train for which the autocorrelogram will be
            calculated.

    Returns:
        float: The frequency with the maximum power in the theta band.

    Raises:
        ValueError: If the input spike train is not valid.
    """
        corr, _ = self.acorr()
        # Take the fft of the spike train autocorr (from -500 to +500ms)
        from scipy.signal import periodogram

        freqs, power = periodogram(corr, fs=200, return_onesided=True)
        power_masked = np.ma.MaskedArray(power,
                                         np.logical_or(freqs < 6, freqs > 12))
        return freqs[np.argmax(power_masked)]

    def smooth_spike_train(self, npos, sigma=3.0, shuffle=None):
        """
        Returns a spike train the same length as num pos samples that has been
        smoothed in time with a gaussian kernel M in width and standard
        deviation equal to sigma.

        Args:
            x1 (np.array): The pos indices the spikes occurred at.
            npos (int): The number of position samples captured.
            sigma (float): The standard deviation of the gaussian used to
                smooth the spike train.
            shuffle (int, optional): The number of seconds to shift the spike
                train by. Default is None.

        Returns:
            smoothed_spikes (np.array): The smoothed spike train.
        """
        spk_hist = np.bincount(self.spike_times, minlength=npos)
        if shuffle is not None:
            spk_hist = np.roll(spk_hist, int(shuffle * 50))
        # smooth the spk_hist (which is a temporal histogram) with a 250ms
        # gaussian as with Kropff et al., 2015
        h = signal.windows.gaussian(13, sigma)
        h = h / float(np.sum(h))
        return signal.filtfilt(h.ravel(), 1, spk_hist)

    def contamination_percent(self,
                              **kwargs) -> tuple:

        c, Qi, Q00, Q01, Ri = contamination_percent(self.spike_times, **kwargs)
        Q = min(Qi/(max(Q00, Q01)))  # this is a measure of refractoriness
        # this is a second measure of refractoriness (kicks in for very low
        # firing rates)
        R = min(Ri)
        return Q, R


class SpikeCalcsAxona(SpikeCalcsGeneric):
    """
    Replaces SpikeCalcs from ephysiopy.axona.spikecalcs
    """

    def half_amp_dur(self, waveforms):
        """
        Calculates the half amplitude duration of a spike.

        Args:
            A (ndarray): An nSpikes x nElectrodes x nSamples array.

        Returns:
            had (float): The half-amplitude duration for the channel
                (electrode) that has the strongest (highest amplitude)
                signal. Units are ms.
        """
        from scipy import optimize

        best_chan = np.argmax(np.max(np.mean(waveforms, 0), 1))
        mn_wvs = np.mean(waveforms, 0)
        wvs = mn_wvs[best_chan, :]
        half_amp = np.max(wvs) / 2
        half_amp = np.zeros_like(wvs) + half_amp
        t = np.linspace(0, 1 / 1000.0, 50)
        # create functions from the data using PiecewisePolynomial
        from scipy.interpolate import BPoly

        p1 = BPoly.from_derivatives(t, wvs[:, np.newaxis])
        p2 = BPoly.from_derivatives(t, half_amp[:, np.newaxis])
        xs = np.r_[t, t]
        xs.sort()
        x_min = xs.min()
        x_max = xs.max()
        x_mid = xs[:-1] + np.diff(xs) / 2
        roots = set()
        for val in x_mid:
            root, infodict, ier, mesg = optimize.fsolve(
                lambda x: p1(x) - p2(x), val, full_output=True
            )
            if ier == 1 and x_min < root < x_max:
                roots.add(root[0])
        roots = list(roots)
        if len(roots) > 1:
            r = np.abs(np.diff(roots[0:2]))[0]
        else:
            r = np.nan
        return r

    def p2t_time(self, waveforms):
        """
        The peak to trough time of a spike in ms

        Args:
            cluster (int): The cluster whose waveforms are to be analysed

        Returns:
            p2t (float): The mean peak-to-trough time for the channel
                (electrode) that has the strongest (highest amplitude) signal.
                Units are ms.
        """
        best_chan = np.argmax(np.max(np.mean(waveforms, 0), 1))
        tP = get_param(waveforms, param="tP")
        tT = get_param(waveforms, param="tT")
        mn_tP = np.mean(tP, 0)
        mn_tT = np.mean(tT, 0)
        p2t = np.abs(mn_tP[best_chan] - mn_tT[best_chan])
        return p2t * 1000

    def plotClusterSpace(self,
                         waveforms,
                         param="Amp",
                         clusts=None,
                         bins=256,
                         **kwargs):
        """
        Assumes the waveform data is signed 8-bit ints
        TODO: aspect of plot boxes in ImageGrid not right as scaled by range of
        values now
        """
        from itertools import combinations

        import matplotlib.colors as colors
        from mpl_toolkits.axes_grid1 import ImageGrid

        from ephysiopy.axona.tintcolours import colours as tcols

        self.scaling = np.full(4, 15)

        amps = get_param(waveforms, param=param)
        cmap = np.tile(tcols[0], (bins, 1))
        cmap[0] = (1, 1, 1)
        cmap = colors.ListedColormap(cmap)
        cmap._init()
        alpha_vals = np.ones(cmap.N + 3)
        alpha_vals[0] = 0
        cmap._lut[:, -1] = alpha_vals
        cmb = combinations(range(4), 2)
        if "fig" in kwargs:
            fig = kwargs["fig"]
        else:
            fig = plt.figure(figsize=(8, 6))
        grid = ImageGrid(fig, 111, nrows_ncols=(2, 3),
                         axes_pad=0.1, aspect=False)
        clustCMap0 = np.tile(tcols[0], (bins, 1))
        clustCMap0[0] = (1, 1, 1)
        clustCMap0 = colors.ListedColormap(clustCMap0)
        clustCMap0._init()
        clustCMap0._lut[:, -1] = alpha_vals
        for i, c in enumerate(cmb):
            h, ye, xe = np.histogram2d(
                amps[:, c[0]], amps[:, c[1]],
                range=((-128, 127), (-128, 127)), bins=bins
            )
            x, y = np.meshgrid(xe[0:-1], ye[0:-1])
            grid[i].pcolormesh(
                x, y, h, cmap=clustCMap0, shading="nearest", edgecolors="face"
            )
            h, ye, xe = np.histogram2d(
                amps[:, c[0]], amps[:, c[1]],
                range=((-128, 127), (-128, 127)), bins=bins
            )
            clustCMap = np.tile(tcols[1], (bins, 1))
            clustCMap[0] = (1, 1, 1)
            clustCMap = colors.ListedColormap(clustCMap)
            clustCMap._init()
            clustCMap._lut[:, -1] = alpha_vals
            grid[i].pcolormesh(
                x, y, h, cmap=clustCMap, shading="nearest", edgecolors="face"
            )
            s = str(c[0] + 1) + " v " + str(c[1] + 1)
            grid[i].text(
                0.05,
                0.95,
                s,
                va="top",
                ha="left",
                size="small",
                color="k",
                transform=grid[i].transAxes,
            )
            grid[i].set_xlim(xe.min(), xe.max())
            grid[i].set_ylim(ye.min(), ye.max())
        plt.setp([a.get_xticklabels() for a in grid], visible=False)
        plt.setp([a.get_yticklabels() for a in grid], visible=False)
        return fig


class SpikeCalcsOpenEphys(SpikeCalcsGeneric):
    def __init__(self, spike_times, cluster, waveforms=None, **kwargs):
        super().__init__(spike_times, cluster, waveforms, **kwargs)
        self.n_samples = [-40, 41]
        self.TemplateModel = None

    def get_waveforms(
        self,
        cluster: int,
        cluster_data: KiloSortSession,
        n_waveforms: int = 2000,
        n_channels: int = 64,
        channel_range=None,
        **kwargs
    ) -> np.ndarray:
        """
        Returns waveforms for a cluster.

        Args:
            cluster (int): The cluster to return the waveforms for.
            cluster_data (KiloSortSession): The KiloSortSession object for the
                session that contains the cluster.
            n_waveforms (int, optional): The number of waveforms to return.
                Defaults to 2000.
            n_channels (int, optional): The number of channels in the
                recording. Defaults to 64.
        """
        # instantiate the TemplateModel - this is used to get the waveforms
        # for the cluster. TemplateModel encapsulates the results of KiloSort
        if self.TemplateModel is None:
            self.TemplateModel = TemplateModel(
                dir_path=os.path.join(cluster_data.fname_root),
                sample_rate=3e4,
                dat_path=os.path.join(cluster_data.fname_root,
                                      "continuous.dat"),
                n_channels_dat=n_channels,
            )
        # get the waveforms for the given cluster on the best channel only
        waveforms = self.TemplateModel.get_cluster_spike_waveforms(cluster)
        # get a random subset of the waveforms
        rng = np.random.default_rng()
        total_waves = waveforms.shape[0]
        n_waveforms = n_waveforms if n_waveforms < total_waves else total_waves
        waveforms_subset = rng.choice(waveforms, n_waveforms)
        # return the waveforms
        if channel_range is None:
            return np.squeeze(waveforms_subset[:, :, 0])
        else:
            if isinstance(channel_range, Sequence):
                return np.squeeze(waveforms_subset[:, :, channel_range])
            else:
                warnings.warn("Invalid channel_range sequence")

    def get_channel_depth_from_templates(self, pname: Path):
        """
        Determine depth of template as well as closest channel. Adopted from
        'templatePositionsAmplitudes' by N. Steinmetz
        (https://github.com/cortex-lab/spikes)
        """
        # Load inverse whitening matrix
        Winv = np.load(os.path.join(pname, "whitening_mat_inv.npy"))
        # Load templates
        templates = np.load(os.path.join(pname, "templates.npy"))
        # Load channel_map and positions
        channel_map = np.load(os.path.join(pname, "channel_map.npy"))
        channel_positions = np.load(os.path.join(pname,
                                                 "channel_positions.npy"))
        map_and_pos = np.array([np.squeeze(channel_map),
                                channel_positions[:, 1]])
        # unwhiten all the templates
        tempsUnW = np.zeros(np.shape(templates))
        for i in np.shape(templates)[0]:
            tempsUnW[i, :, :] = np.squeeze(templates[i, :, :]) @ Winv

        tempAmp = np.squeeze(np.max(tempsUnW, 1)) - \
            np.squeeze(np.min(tempsUnW, 1))
        tempAmpsUnscaled = np.max(tempAmp, 1)
        # need to zero-out the potentially-many low values on distant channels
        threshVals = tempAmpsUnscaled * 0.3
        tempAmp[tempAmp < threshVals[:, None]] = 0
        # Compute the depth as a centre of mass
        templateDepths = np.sum(tempAmp * map_and_pos[1, :], -1) / \
            np.sum(tempAmp, 1)
        maxChanIdx = np.argmin(
            np.abs((templateDepths[:, None] - map_and_pos[1, :].T)), 1
        )
        return templateDepths, maxChanIdx

    def get_template_id_for_cluster(self, pname: Path, cluster: int):
        """
        Determine the best channel (one with highest amplitude spikes)
        for a given cluster.
        """
        spike_templates = np.load(os.path.join(pname, "spike_templates.npy"))
        spike_times = np.load(os.path.join(pname, "spike_times.npy"))
        spike_clusters = np.load(os.path.join(pname, "spike_clusters.npy"))
        cluster_times = spike_times[spike_clusters == cluster]
        rez_mat = h5py.File(os.path.join(pname, "rez.mat"), "r")
        st3 = rez_mat["rez"]["st3"]
        st_spike_times = st3[0, :]
        idx = np.searchsorted(st_spike_times, cluster_times)
        template_idx, counts = np.unique(spike_templates[idx],
                                         return_counts=True)
        ind = np.argmax(counts)
        return template_idx[ind]


class SpikeCalcsProbe(SpikeCalcsGeneric):
    """
    Encapsulates methods specific to probe-based recordings
    """

    def __init__(self):
        pass
