import os
import warnings
from pathlib import Path
from collections import namedtuple
from astropy.convolution import convolve, Box1DKernel, Gaussian1DKernel
from collections.abc import Sequence
import h5py
import seaborn as sns
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.transforms as transforms
import numpy as np
from scipy.special import erf
from scipy import interpolate
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from phylib.io.model import TemplateModel
from scipy import signal, stats
from ephysiopy.common.utils import (
    min_max_norm,
    shift_vector,
    BinnedData,
    VariableToBin,
    MapType,
    TrialFilter,
)

# from ephysiopy.common.rhythmicity import LFPOscillations
from ephysiopy.openephys2py.KiloSort import KiloSortSession

# from ephysiopy.common.statscalcs import mean_resultant_vector


def get_param(waveforms, param="Amp", t=200, fet=1) -> np.ndarray:
    """
    Returns the requested parameter from a spike train as a numpy array.

    Parameters
    ----------
    waveforms : np.ndarray
        Shape of array can be nSpikes x nSamples OR nSpikes x nElectrodes x nSamples.
    param : str, default='Amp'
        Valid values are:
        - 'Amp': peak-to-trough amplitude
        - 'P': height of peak
        - 'T': depth of trough
        - 'Vt': height at time t
        - 'tP': time of peak (in seconds)
        - 'tT': time of trough (in seconds)
        - 'PCA': first n fet principal components (defaults to 1)
    t : int, default=200
        The time used for Vt
    fet : int, default=1
        The number of principal components (use with param 'PCA').

    Returns
    -------
    np.ndarray
        The requested parameter as a numpy array.

    """
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
        m = interpolate.interp1d([0, waveforms.shape[-1] - 1], [0, 1 / 1000.0])
        return m(idx)
    elif param == "tT":
        idx = np.argmin(waveforms, axis=-1)
        m = interpolate.interp1d([0, waveforms.shape[-1] - 1], [0, 1 / 1000.0])
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
                        out[:, rng[0, i] : rng[1, i]] = np.atleast_2d(A).T
                    else:
                        out[:, rng[0, i] : rng[1, i]] = A
            return out


def get_peak_to_trough_time(waveforms: np.ndarray) -> np.ndarray:
    """
    Returns the time in seconds of the peak to trough in a waveform.

    Parameters
    ----------
    waveforms : np.ndarray
        The waveforms to calculate the peak to trough time for.

    Returns
    -------
    np.ndarray
        The time of the peak to trough in seconds.
    """
    peak_times = get_param(waveforms, "tP")
    trough_times = get_param(waveforms, "tT")
    return np.mean(trough_times - peak_times)


def get_burstiness(
    isi_matrix: np.ndarray, whiten: bool = False, plot_pcs: bool = False
) -> np.ndarray:
    """
    Returns the burstiness of a waveform.

    Parameters
    ----------
    isi_matrix : np.ndarray
        A matrix of normalized interspike intervals (ISIs) for the neurons.
        Rows are neurons, columns are ISI time bins.

    Returns
    -------
    np.ndarray

    Notes
    -----
    Algorithm:

    1) The interspike intervals between 0 and 60ms were binned into 2ms bins,
    and the area of the histogram was normalised to 1 to produce a
    probability distribution histogram for each neuron

    2) A principal components analysis (PCA) is performed on the matrix of
    the ISI probability distributions of all neurons

    3) Neurons were then assigned to two clusters using a k-means clustering
    algorithm on the first three principal components

    4) a linear discriminant analysis performed in MATLAB (‘classify’) was
    undertaken to determine the optimal linear discriminant (Fishers Linear
    Discriminant) i.e., the plane which best separated the two clusters in a
    three-dimensional scatter plot of the principal components.

    Training on 80% of the data and testing on the remaining 20% resulted in a
    good separation of the two clusters.

    5) A burstiness score was assigned to each neuron which was calculated by
    computing the shortest distance between the plotted point for each neuron
    in the three-dimensional cluster space (principal components 1,2 and 3),
    and the plane separating the two clusters (i.e., the optimal linear
    discriminant).

    6) To ensure the distribution of these burstiness scores was bimodal,
    reflecting the presence of two classes of neuron (‘bursty’ versus
    ‘non-bursty’), probability density functions for Gaussian mixture models
    with between one and four underlying Gaussian curves were fitted and the
    fit of each compared using the Akaike information criterion (AIC)

    7) Optionally plot the principal components and the centres of the
    kmeans results

    """
    isi_matrix = np.asarray(isi_matrix)
    # A) check for NaN values in the isi_matrix and remove
    remove_idx = np.isnan(np.sum(isi_matrix, -1))
    if np.any(remove_idx):
        warnings.warn("NaN values detected in isi_matrix, removing those rows")
        isi_matrix = isi_matrix[~remove_idx, :]

    # B) do the PCA
    from sklearn.decomposition import PCA

    pca = PCA(n_components=3, whiten=whiten)
    pca.fit(isi_matrix)
    pca_matrix = pca.transform(isi_matrix)  # columns are principal components

    # C) do the k-means clustering
    from scipy.cluster.vq import kmeans2

    km = kmeans2(
        pca_matrix,
        2,
        minit="points",
        iter=20,
        missing="raise",
        seed=np.random.default_rng(21),
    )

    # D) do the linear discriminant analysis
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    lda = LDA()
    lda.fit(pca_matrix, km[1])

    # E) calculate the distance from the plane
    from scipy.spatial.distance import cdist

    # get the coefficients of the plane
    coeffs = lda.coef_[0]
    # intercept = lda.intercept_[0]
    # calculate the distance from the plane
    distances = cdist(pca_matrix, [coeffs], metric="mahalanobis")
    # normalize the distances
    distances = (distances - np.min(distances)) / (
        np.max(distances) - np.min(distances)
    )
    # project the ISI distributions onto the optimal linear discriminant
    # of the two clusters
    pca_matrix_normed = pca_matrix - np.mean(pca_matrix, axis=0)
    pca_matrix_normed = pca_matrix_normed / np.std(pca_matrix_normed, axis=0)
    # project onto the plane
    pca_distances = np.dot(pca_matrix_normed, coeffs.T)

    if plot_pcs:
        # F) plot the principal components and the centres
        # of the kmeans results
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(projection="3d")
        k1 = km[1] == 0
        k2 = km[1] == 1
        ax.scatter(
            pca_matrix[k1, 0],
            pca_matrix[k1, 1],
            pca_matrix[k1, 2],
            c="blue",
            label="Cluster 1",
            alpha=0.5,
        )

        ax.scatter(
            pca_matrix[k2, 0],
            pca_matrix[k2, 1],
            pca_matrix[k2, 2],
            c="red",
            label="Cluster 2",
            alpha=0.5,
        )
        plt.scatter(
            km[0][:, 0],
            km[0][:, 1],
            marker="x",
            s=100,
            color="green",
            label="K-means centroids",
        )
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_zlabel("PC 3")
        plt.title("PCA of ISI Matrix with K-means Clustering")
        plt.legend()

        # Plot the ISI data sorted by the distances to the discriminant
        # boundary
        _, ax2 = plt.subplots(figsize=(5, 12))
        sorted_indices = np.argsort(pca_distances.squeeze())
        # normalise the ISIS matrix for better visualisation
        isi_matrix = isi_matrix / np.max(isi_matrix, axis=-1, keepdims=True)
        xi = np.linspace(0, 60, isi_matrix.shape[1] + 1)
        yi = np.arange(isi_matrix.shape[0] + 1)
        vmax = np.max(isi_matrix[sorted_indices, :])

        ax2.pcolormesh(
            xi,
            yi,
            isi_matrix[sorted_indices, :],
            cmap="viridis",
            edgecolors="face",
            vmax=vmax,
        )
        ax2.set_aspect(0.1)
        ax2.set_xlabel("ISI bins(ms)")
        ax2.set_xticks([0, 60])
        ax2.set_ylabel("Cells")
        ax2.set_yticks([])
        ax2.annotate(
            "Less bursty",
            xy=(-0.1, 0.95),
            xytext=(-0.1, 0.7),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="black"),
            rotation=90,
            fontsize=10,
            color="black",
            ha="center",
        )
        ax2.annotate(
            "More bursty",
            xy=(-0.1, 0.05),
            xytext=(-0.1, 0.2),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="black"),
            rotation=90,
            fontsize=10,
            color="black",
            ha="center",
        )

        # Plot the distances as a histogram
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        N, bins, patches = ax3.hist(pca_distances, bins=150, density=True, color="blue")
        for bin_patch in zip(bins[:-1], patches):
            if bin_patch[0] < lda.intercept_:
                bin_patch[1].set_facecolor("red")
        axtrans = transforms.blended_transform_factory(ax3.transData, ax3.transAxes)
        ax3.vlines(
            lda.intercept_, ymin=0, ymax=1, colors="black", transform=axtrans, zorder=1
        )
        ax3.set_xlabel("Burstiness")
        ax3.set_ylabel("Probability")

        # Try a plot using seaborn
        _, ax4 = plt.subplots(figsize=(7, 5))
        df = pd.DataFrame({"Burstiness": pca_distances, "Cluster": km[1]})
        sns.histplot(
            df,
            x="Burstiness",
            stat="density",
            bins=150,
            kde=True,
            ax=ax4,
        )
        axtrans = transforms.blended_transform_factory(ax4.transData, ax4.transAxes)
        ax4.vlines(
            lda.intercept_, ymin=0, ymax=1, colors="black", transform=axtrans, zorder=1
        )
        plt.show()
    # return the distances as the burstiness score
    return pca_distances.squeeze(), pca_matrix, isi_matrix


def mahal(u, v):
    """
    Returns the L-ratio and Isolation Distance measures calculated on the
    principal components of the energy in a spike matrix.

    Parameters
    ----------
    u : np.ndarray
        The first set of waveforms.
    v : np.ndarray
        The second set of waveforms.

    Returns
    -------
    np.ndarray
        The Mahalanobis distances.

    Raises
    ------
    Warning
        If input size mismatch, too few rows, or complex inputs are detected.

    """
    u_sz = u.shape
    v_sz = v.shape
    if u_sz[1] != v_sz[1]:
        warnings.warn(
            "Input size mismatch: \
                        matrices must have same num of columns"
        )
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


def cluster_quality(
    waveforms: np.ndarray = None,
    spike_clusters: np.ndarray = None,
    cluster_id: int = None,
    fet: int = 1,
):
    """
    Returns the L-ratio and Isolation Distance measures calculated
    on the principal components of the energy in a spike matrix.

    Parameters
    ----------
    waveforms : np.ndarray, optional
        The waveforms to be processed. If None, the function will return None.
    spike_clusters : np.ndarray, optional
        The spike clusters to be processed.
    cluster_id : int, optional
        The ID of the cluster to be processed.
    fet : int, optional
        The feature to be used in the PCA calculation (default is 1).

    Returns
    -------
    tuple
        A tuple containing the L-ratio and Isolation Distance of the cluster.

    Raises
    ------
    Exception
        If an error occurs during the calculation of the L-ratio or Isolation Distance.

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


def xcorr(
    x1: np.ndarray,
    x2: np.ndarray | None = None,
    Trange: np.ndarray | list = np.array([-0.5, 0.5]),
    binsize: float = 0.001,
    normed=False,
    **kwargs,
) -> BinnedData:
    """
    Calculates the ISIs in x1 or x1 vs x2 within a given range.

    Parameters
    ----------
    x1 : np.ndarray
        The times of the spikes emitted by the first cluster in seconds.
    x2 : np.ndarray, optional
        The times of the spikes emitted by the second cluster in seconds. If None, x1 is used.
    Trange : np.ndarray or list, optional
        Range of times to bin up in seconds (default is [-0.5, 0.5]).
    binsize : float, optional
        The size of the bins in seconds (default is 0.001).
    normed : bool, optional
        Whether to divide the counts by the total number of spikes to give a probability (default is False).
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    BinnedData
        A BinnedData object containing the binned data and the bin edges.

    """
    if x2 is None:
        x2 = x1.copy()

    if isinstance(Trange, list):
        Trange = np.array(Trange)

    y = []
    irange = x2[:, np.newaxis] + Trange[np.newaxis, :]
    dts = np.searchsorted(x1, irange)

    for i, t in enumerate(dts):
        y.extend((x1[t[0] : t[1]] - x2[i]))
    y = np.array(y, dtype=float)

    counts, bins = np.histogram(
        y[y != 0], bins=int(np.ptp(Trange) / binsize) + 1, range=(Trange[0], Trange[1])
    )

    if normed:
        counts = counts / len(x1)
    ids = kwargs.pop("cluster_id", [])

    return BinnedData(
        variable=VariableToBin.TIME,
        map_type=MapType.SPK,
        binned_data=[counts],
        bin_edges=[bins],
        cluster_id=ids,
    )


def fit_smoothed_curve_to_xcorr(xc: BinnedData, **kwargs) -> BinnedData:
    """
    Idea is to smooth out the result of an auto- or cross-correlogram with
    a view to correlating the result with another auto- or cross-correlogram
    to see how similar two of these things are.

    Check Brandon et al., 2011?2012?
    """
    pass


def contamination_percent(
    x1: np.ndarray, x2: np.ndarray | None = None, **kwargs
) -> tuple:
    """
    Computes the cross-correlogram between two sets of spikes and
    estimates how refractory the cross-correlogram is.

    Parameters
    ----------
    x1 : np.ndarray
        The first set of spikes.
    x2 : np.ndarray, optional
        The second set of spikes. If None, x1 is used.
    **kwargs : dict
        Additional keyword arguments that can be fed into xcorr.

    Returns
    -------
    tuple
        A tuple containing:
        - Q (float): A measure of refractoriness.
        - R (float): A second measure of refractoriness (kicks in for very low firing rates).

    Notes
    -----
    Taken from KiloSorts ccg.m

    The contamination metrics are calculated based on
    an analysis of the 'shoulders' of the cross-correlogram.
    Specifically, the spike counts in the ranges +/-5-25ms and

    """
    if x2 is None:
        x2 = x1.copy()
    xc = xcorr(x1, x2, **kwargs)
    b = xc.bin_edges[0]
    c = xc.binned_data[0]
    left = [[-0.05, -0.01]]
    right = [[0.01, 0.051]]
    far = [[-0.5, -0.249], [0.25, 0.501]]

    def get_shoulder(bins, vals):
        all = np.array([np.logical_and(bins >= i[0], bins < i[1]) for i in vals])
        return np.any(all, 0)

    inner_left = get_shoulder(b, left)
    inner_right = get_shoulder(b, right)
    outer = get_shoulder(b, far)

    tbin = 1000
    Tr = max(np.concatenate([x1, x2])) - min(np.concatenate([x1, x2]))

    def get_normd_shoulder(idx):
        return np.nansum(c[idx[:-1]]) / (
            len(np.nonzero(idx)[0]) * tbin * len(x1) * len(x2) / Tr
        )

    Q00 = get_normd_shoulder(outer)
    Q01 = np.nanmax([get_normd_shoulder(inner_left), get_normd_shoulder(inner_right)])

    R00 = np.nanmax(
        [
            np.nanmean(c[outer[:-1]]),
            np.nanmean(c[inner_left[:-1]]),
            np.nanmean(c[inner_right[1:]]),
        ]
    )

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
        n = np.nansum(c[chunk[:-1]]) / 2
        lam = R00 * i
        # this is tricky: we approximate the Poisson likelihood with a
        # gaussian of equal mean and variance
        # that allows us to integrate the probability that we would see <N
        # spikes in the center of the
        # cross-correlogram from a distribution with mean R00*i spikes
        p = 1 / 2 * (1 + erf((n - lam) / np.sqrt(2 * lam)))

        Ri[i] = p  # keep track of p for each bin size i

    c[middle_idx] = a  # restore the center value of the cross-correlogram
    return c, Qi, Q00, Q01, Ri


# a namedtuple to hold some metrics from the KiloSort run
KSMetaTuple = namedtuple("KSMeta", "Amplitude group KSLabel ContamPct ")


class SpikeCalcsGeneric(object):
    """
    Deals with the processing and analysis of spike data.
    There should be one instance of this class per cluster in the
    recording session. NB this differs from previous versions of this
    class where there was one instance per recording session and clusters
    were selected by passing in the cluster id to the methods.

    NB Axona waveforms are nSpikes x nChannels x nSamples - this boils
    down to nSpikes x 4 x 50
    NB KiloSort waveforms are nSpikes x nSamples x nChannels - these are ordered
    by 'best' channel first and then the rest of the channels. This boils
    down to nSpikes x 61 x 12 SO THIS NEEDS TO BE CHANGED to
    nSpikes x nChannels x nSamples

    Parameters
    ----------
    spike_times : np.ndarray
        The times of spikes in the trial in seconds.
    cluster : int
        The cluster ID.
    waveforms : np.ndarray, optional
        An nSpikes x nChannels x nSamples array.
    **kwargs : dict
        Additional keyword arguments.

    Attributes
    ----------
    spike_times : np.ma.MaskedArray
        The times of spikes in the trial in seconds.
    _waves : np.ma.MaskedArray or None
        The waveforms of the spikes.
    cluster : int
        The cluster ID.
    n_spikes : int
        the total number of spikes for the current cluster
    duration : float, int
        total duration of the trial in seconds
    event_ts : np.ndarray or None
        The times that events occurred in seconds.
    event_window : np.ndarray
        The window, in seconds, either side of the stimulus, to examine.
    stim_width : float or None
        The width, in ms, of the stimulus.
    secs_per_bin : float
        The size of bins in PSTH.
    sample_rate : int
        The sample rate of the recording.
    pos_sample_rate : int
        The sample rate of the position data.
    pre_spike_samples : int
        The number of samples before the spike.
    post_spike_samples : int
        The number of samples after the spike.
    KSMeta : KSMetaTuple
        The metadata from KiloSort.
    """

    def __init__(
        self,
        spike_times: np.ndarray,
        cluster: int,
        waveforms: np.ndarray = None,
        **kwargs,
    ):
        self.spike_times = np.ma.MaskedArray(spike_times)  # IN SECONDS
        if waveforms is not None:
            # if waveforms.shape[-1] > 50:
            # this is a hack to deal with the fact that KiloSort waveforms are 82 samples long
            # and I want them comparable with Axona which is 50
            # waveforms = waveforms[:, :, 16:66]
            n_spikes, n_channels, n_samples = waveforms.shape
            assert self.n_spikes == n_spikes, (
                "Number of spike times does not match number of waveforms"
            )
            self._waves = np.ma.MaskedArray(waveforms)
        else:
            self._waves = None
        self.cluster = cluster
        self._event_ts = None  # the times that events occured IN SECONDS
        # window, in seconds, either side of the stimulus, to examine
        self._event_window = np.array((-0.050, 0.100))
        self._stim_width = None  # the width, in ms, of the stimulus
        # used to increase / decrease size of bins in psth
        self._secs_per_bin = 0.001
        self._sample_rate = 50000
        self._pos_sample_rate = 50
        self._duration = None
        self._invert_waveforms = False
        # these values should be specific to Axona data
        # this I think is wrong as the pre capture buffer
        # should be 200ms and 800ms post trigger
        # whereas these values are equal to 330 microseconds
        # and 708 microseconds if sampling is at 48kHz
        self._pre_spike_samples = 10
        self._post_spike_samples = 40
        # if waveforms is not None:
        #     if self.n_samples > 50:
        # self.pre_spike_samples = 41
        # self.post_spike_samples = 41
        # self.sample_rate = 30000
        # these values are specific to OE data
        # if sample rate is 3kHz and the number of samples
        # captured per waveform is 82 then it looks from
        # plotting the waveforms that the pre spike samples is
        # 40 so 1.33 milliseconds and the pos spike samples is
        # 1.40 milliseconds
        # values from running KS
        self._ksmeta = KSMetaTuple(None, None, None, None)
        # update the __dict__ attribute with the kwargs
        self.__dict__.update(kwargs)

    @property
    def sample_rate(self) -> int | float:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int | float) -> None:
        self._sample_rate = value

    @property
    def pos_sample_rate(self) -> int | float:
        return self._pos_sample_rate

    @pos_sample_rate.setter
    def pos_sample_rate(self, value: int | float) -> None:
        self._pos_sample_rate = value

    @property
    def pre_spike_samples(self) -> int:
        return self._pre_spike_samples

    @pre_spike_samples.setter
    def pre_spike_samples(self, value: int) -> None:
        self._pre_spike_samples = int(value)

    @property
    def post_spike_samples(self) -> int:
        return self._post_spike_samples

    @post_spike_samples.setter
    def post_spike_samples(self, value: int) -> None:
        self._post_spike_samples = int(value)

    # its possible due to referencing that waveforms are inverted
    # so add the option to correct that here
    @property
    def invert_waveforms(self) -> bool:
        return self._invert_waveforms

    @invert_waveforms.setter
    def invert_waveforms(self, val: bool) -> None:
        self._invert_waveforms = val

    def waveforms(self, channel_id: Sequence = None) -> np.ndarray | None:
        """
        Returns the waveforms of the cluster.

        Parameters
        ----------
        channel_id : Sequence, optional
            The channel IDs to return the waveforms for.
            If None, returns waveforms for all channels.

        Returns
        -------
        np.ndarray | None
            The waveforms of the cluster,
            or None if no waveforms are available.


        """
        if self._waves is not None:
            scaling = 1
            if self.invert_waveforms:
                scaling = -1
            if channel_id is None:
                return self._waves[:, :, :] * scaling
            else:
                if isinstance(channel_id, int):
                    channel_id = [channel_id]
                return self._waves[:, channel_id, :] * scaling
        else:
            return None

    @property
    def n_spikes(self):
        """
        Returns the number of spikes in the cluster

        Returns
        -------
        int
            The number of spikes in the cluster
        """
        return np.ma.count(self.spike_times)

    @property
    def n_channels(self) -> int | None:
        """
        Returns the number of channels in the waveforms.

        Returns
        -------
        int | None
            The number of channels in the waveforms,
            or None if no waveforms are available.
        """
        if self._waves is not None:
            return self._waves.shape[1]
        else:
            return None

    @property
    def n_samples(self) -> int | None:
        """
        Returns the number of samples in the waveforms.

        Returns
        -------
        int | None
            The number of samples in the waveforms,
            or None if no waveforms are available.
        """
        if self._waves is not None:
            return self._waves.shape[2]
        else:
            return None

    @property
    def event_ts(self) -> np.ndarray:
        return self._event_ts

    @event_ts.setter
    def event_ts(self, value: np.ndarray) -> None:
        self._event_ts = value

    @property
    def duration(self) -> float | int | None:
        return self._duration

    @duration.setter
    def duration(self, value: float | int | None):
        self._duration = value

    @property
    def KSMeta(self) -> KSMetaTuple:
        return self._ksmeta

    def update_KSMeta(self, value: dict) -> None:
        """
        Takes in a TemplateModel instance from a phy session and
        parses out the relevant metrics for the cluster and places
        into the namedtuple KSMeta.

        Parameters
        ----------
        value : dict
            A dictionary containing the relevant metrics for the cluster.

        """
        metavals = []
        for f in KSMetaTuple._fields:
            if f in value.keys():
                if self.cluster in value[f].keys():
                    metavals.append(value[f][self.cluster])
                else:
                    metavals.append(None)
            else:
                metavals.append(None)
        self._ksmeta = KSMetaTuple(*metavals)

    @property
    def event_window(self) -> np.ndarray:
        return self._event_window

    @event_window.setter
    def event_window(self, value: np.ndarray):
        self._event_window = value

    @property
    def stim_width(self) -> int | float | None:
        return self._stim_width

    @stim_width.setter
    def stim_width(self, value: int | float | None):
        self._stim_width = value

    @property
    def secs_per_bin(self) -> float | int:
        return self._secs_per_bin

    @secs_per_bin.setter
    def secs_per_bin(self, value: float | int):
        self._secs_per_bin = value

    def apply_filter(self, *trial_filter: TrialFilter) -> None:
        """
        Applies a mask to the spike times.

        Parameters
        ----------
        trial_filter : TrialFilter
            The filter
        """
        if trial_filter:
            for i_filter in trial_filter:
                assert isinstance(i_filter, TrialFilter), "Filter must be a TrialFilter"
                assert i_filter.name == "time", "Only time filters are supported"
        self.spike_times.mask = False
        if self._waves and self._waves is not None:
            self._waves.mask = False
        if not trial_filter or len(trial_filter) == 0:
            if self._waves and self._waves is not None:
                self._waves.mask = False
            self.spike_times.mask = False
        else:
            mask = np.zeros_like(self.spike_times, dtype=bool)
            for i_filter in trial_filter:
                i_mask = np.logical_and(
                    self.spike_times > i_filter.start, self.spike_times < i_filter.end
                )
                mask = np.logical_or(mask, i_mask)
            self.spike_times.mask = mask
            if self._waves is not None:
                self._waves.mask = mask

    def acorr(self, Trange: np.ndarray = np.array([-0.5, 0.5]), **kwargs) -> BinnedData:
        """
        Calculates the autocorrelogram of a spike train.

        Parameters
        ----------
        Trange : np.ndarray, optional
            The range of times to calculate the autocorrelogram over (default is [-0.5, 0.5]).
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        BinnedData
            Container for the binned data.
        """
        return xcorr(self.spike_times, Trange=Trange, **kwargs)

    def trial_mean_fr(self) -> float:
        # Returns the trial mean firing rate for the cluster
        if self.duration is None:
            raise IndexError("No duration provided, give me one!")
        return self.n_spikes / self.duration

    def mean_isi_range(self, isi_range: float) -> float:
        """
        Calculates the mean of the autocorrelation from 0 to n seconds.
        Used to help classify a neuron's type (principal, interneuron, etc).

        Parameters
        ----------
        isi_range : int
            The range in seconds to calculate the mean over.

        Returns
        -------
        float
            The mean of the autocorrelogram between 0 and n seconds.
        """
        trange = np.array((-0.5, 0.5))
        ac = self.acorr(Trange=trange)
        bins = ac.bin_edges[0]
        counts = ac.binned_data[0]
        mask = np.logical_and(bins > 0, bins < isi_range)
        return np.mean(counts[mask[1:]])

    def mean_waveform(self, channel_id: Sequence = None):
        """
        Returns the mean waveform and standard error of the mean (SEM) for a
        given spike train on a particular channel.

        Parameters
        ----------
        channel_id : Sequence, optional
            The channel IDs to return the mean waveform for. If None, returns
            mean waveforms for all channels.

        Returns
        -------
        tuple
            A tuple containing:
            - mn_wvs (np.ndarray): The mean waveforms, usually 4x50 for tetrode recordings.
            - std_wvs (np.ndarray): The standard deviations of the waveforms, usually 4x50 for tetrode recordings.
        """
        x = self.waveforms(channel_id)
        if x is not None:
            return np.mean(x, axis=0), np.std(x, axis=0)
        else:
            return None

    def get_best_channel(self) -> int | None:
        """
        Returns the channel with the highest mean amplitude of the waveforms.

        Returns
        -------
        int | None
            The index of the channel with the highest mean amplitude,
            or None if no waveforms are available.
        """
        wvs = self.waveforms()
        if wvs is not None:
            amps = np.mean(np.ptp(wvs, axis=-1), axis=0)
            return np.argmax(amps)
        else:
            return None

    def estimate_AHP(self) -> float | None:
        """
        Estimate the decay time for the AHP of the waveform of the
        best channel for the current cluster.

        Returns
        -------
        float | None
            The estimated AHP decay time in microseconds,
            or None if no waveforms are available.
        """
        best_chan = self.get_best_channel()
        waveform, _ = self.mean_waveform(best_chan)

        if waveform is None:
            return None

        # get the times
        times = np.linspace(0, 1000, self.n_samples)  # in microseconds
        pre_spike = int(self.pre_spike_samples * (1000000 / self.sample_rate))
        post_spike = int(self.post_spike_samples * (1000000 / self.sample_rate))

        f = interpolate.interp1d(
            times, np.linspace(-(pre_spike), post_spike, self.n_samples), "nearest"
        )
        f_t = f(times)

        # get the baseline voltage (for Axona this should be the
        # 200 microseconds before the spike)
        baseline_voltage = np.mean(waveform[f_t < 0])
        # make sure this min measure isafter the spike would
        # have triggered the capture of the buffer i.e. after t=0
        min_voltage = np.min(waveform[f_t >= 0])
        half_voltage = (baseline_voltage + min_voltage) / 2

        # get the index of the minima of the waveform
        t_thresh = np.nonzero(f_t > 0)[0][0]
        idx = np.argmin(waveform[f_t > 0]) + t_thresh

        # find the time it takes to return to baseline voltage
        X = np.atleast_2d(f_t[idx:]).T
        y = np.atleast_2d(waveform[idx:]).T

        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)

        poly.fit(X_poly, y)
        lin2 = LinearRegression()
        lin2.fit(X_poly, y)

        tn = np.linspace(f_t[idx], 1500, 1000).reshape(-1, 1)
        yn = lin2.predict(poly.fit_transform(tn))

        # predict the time it takes to return to half voltage
        ahp_time = None
        for i, v in enumerate(yn):
            if v >= half_voltage:
                ahp_time = tn[i][0]
                break

        try:
            return ahp_time - f_t[idx]
        except Exception:
            return None

    def psth(self) -> tuple[list, ...]:
        """
        Calculate the PSTH of event_ts against the spiking of a cell


        Returns
        -------
        x, y : list
        The list of time differences between the spikes of the cluster
        and the events (x) and the trials (y)
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
            tmp = self.spike_times[t[0] : t[1]] - event_ts[i]
            x.extend(tmp)
            y.extend(np.repeat(i, len(tmp)))
        return x, y

    def psch(self, bin_width_secs: float) -> np.ndarray:
        """
        Calculate the peri-stimulus *count* histogram of a cell's spiking
        against event times.

        Parameters
        ----------
        bin_width_secs : float
            The width of each bin in seconds.

        Returns
        -------
        result : np.ndarray
            Rows are counts of spikes per bin_width_secs.
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
        bins = np.arange(self.event_window[0], self.event_window[1], bin_width_secs)
        result = np.empty(shape=(len(bins) - 1, len(event_ts)), dtype=np.int64)
        for i, t in enumerate(dts):
            tmp = self.spike_times[t[0] : t[1]] - event_ts[i]
            indices = np.digitize(tmp, bins=bins)
            counts = np.bincount(indices, minlength=len(bins))
            result[:, i] = counts[1:]
        return result

    def get_shuffled_ifr_sp_corr(
        self, ts: np.array, speed: np.array, nShuffles: int = 100, **kwargs
    ):
        """
        Returns an nShuffles x nSamples sized array of shuffled
        instantaneous firing rate x speed correlations

        Parameters
        ----------
        ts : np.ndarray
            the times in seconds at which the cluster fired
        speed : np.ndarray
            the speed vector
        nShuffles : int
            the number of times to shuffle the timestamp vector 'ts'
        **kwargs
            Passed into ifr_sp_corr

        Returns
        -------
        np.ndarray
            A nShuffles x nSamples sized array of the shuffled firing rate vs
            speed correlations.
        """
        # shift spikes by at least 30 seconds after trial start and
        # 30 seconds before trial end
        nSamples = len(speed)
        random_seed = kwargs.get("random_seed", None)
        r = np.random.default_rng(random_seed)
        timeSteps = r.integers(low=30, high=ts[-1] - 30, size=nShuffles)
        shuffled_ifr_sp_corrs = []
        for t in timeSteps:
            shift_ts = shift_vector(ts, t, maxlen=nSamples)
            res = self.ifr_sp_corr(shift_ts, speed, **kwargs)
            shuffled_ifr_sp_corrs.append(res.statistic)
        return np.array(shuffled_ifr_sp_corrs)

    def ifr_sp_corr(
        self,
        ts,
        speed,
        minSpeed=2.0,
        maxSpeed=40.0,
        sigma=3,
        nShuffles=100,
        **kwargs,
    ):
        """
        Calculates the correlation between the instantaneous firing rate and
        speed.

        Parameters
        ----------
        ts  : np.ndarray
            The times in seconds at which the cluster fired.
        speed : np.ndarray
            Instantaneous speed (nSamples lenght vector).
        minSpeed : float, default=2.0
            Speeds below this value are ignored.
        maxSpeed : float, default=40.0
            Speeds above this value are ignored.
        sigma : int, default=3
            The standard deviation of the gaussian used
            to smooth the spike train.
        nShuffles : int, default=100
            The number of resamples to feed into
            the permutation test.
        **kwargs:
            method: how the significance of the speed vs firing rate correlation
                    is calculated

        Examples
        --------
        An example of how I was calculating this is:

        >> rng = np.random.default_rng()
        >> method = stats.PermutationMethod(n_resamples=nShuffles, random_state=rng)

        See Also
        --------
        See scipy.stats.PermutationMethod.

        """
        speed = np.ma.masked_invalid(speed)
        speed = speed.ravel()
        orig_speed_mask = speed.mask
        posSampRate = self.pos_sample_rate
        nSamples = len(speed)
        x1 = np.floor(ts * posSampRate).astype(int)
        # crop the end of the timestamps if longer than the pos data
        x1 = np.delete(x1, np.nonzero(x1 >= nSamples))
        spk_hist = np.bincount(x1, minlength=nSamples)
        # smooth the spk_hist (which is a temporal histogram) with a 250ms
        # gaussian as with Kropff et al., 2015
        h = signal.windows.gaussian(13, sigma)
        h = h / float(np.sum(h))
        # filter for low and high speeds
        speed_mask = np.logical_or(speed < minSpeed, speed > maxSpeed)
        # make sure the original mask is preserved
        speed_mask = np.logical_or(speed_mask, orig_speed_mask)
        speed_filt = np.ma.MaskedArray(speed, speed_mask)
        # speed might contain nans so mask these too
        speed_filt = np.ma.fix_invalid(speed_filt)
        speed_mask = speed_filt.mask
        spk_hist_filt = np.ma.MaskedArray(spk_hist, speed_mask)
        spk_sm = signal.filtfilt(h.ravel(), 1, spk_hist_filt)
        sm_spk_rate = np.ma.MaskedArray(spk_sm * posSampRate, speed_mask)
        # the permutation test for significance, only perform
        # on the non-masked data
        rng = np.random.default_rng()
        method = stats.PermutationMethod(n_resamples=nShuffles, random_state=rng)
        method = kwargs.get("method", method)
        res = stats.pearsonr(
            sm_spk_rate.compressed(), speed_filt.compressed(), method=method
        )
        return res

    def get_ifr(self, spike_times: np.array, n_samples: int, **kwargs) -> np.ndarray:
        """
        Returns the instantaneous firing rate of the cluster

        Parameters
        ----------
        spike_times : np.ndarray
            The times in seconds at which the cluster fired.
        n_samples : int
            The number of samples to use in the calculation.
            Practically this should be the number of position
            samples in the recording.

        Returns
        -------
        np.ndarray
            The instantaneous firing rate of the cluster
        """
        posSampRate = self.pos_sample_rate
        x1 = np.floor(spike_times * posSampRate).astype(int)
        spk_hist = np.bincount(x1, minlength=n_samples)
        sigma = kwargs.get("sigma", 3)
        h = signal.windows.gaussian(13, sigma)
        h = h / float(np.sum(h))
        spk_sm = signal.filtfilt(h.ravel(), 1, spk_hist)
        ifr = spk_sm * posSampRate
        return ifr

    def responds_to_stimulus(
        self,
        threshold: float,
        min_contiguous: int,
        return_activity: bool = False,
        return_magnitude: bool = False,
        **kwargs,
    ):
        """
        Checks whether a cluster responds to a laser stimulus.

        Parameters
        ----------
        threshold : float
            The amount of activity the cluster needs to go
            beyond to be classified as a responder (1.5 = 50% more or less
            than the baseline activity).
        min_contiguous : int
            The number of contiguous samples in the
            post-stimulus period for which the cluster needs to be active
            beyond the threshold value to be classed as a responder.
        return_activity : bool
            Whether to return the mean reponse curve.
        return_magnitude : int
            Whether to return the magnitude of the
            response. NB this is either +1 for excited or -1 for inhibited.

        Returns
        -------
        namedtuple
            With named fields "responds" (bool), "normed_response_curve" (np.ndarray),
            "response_magnitude" (np.ndarray)
        """
        spk_count_by_trial = self.psch(self._secs_per_bin)
        firing_rate_by_trial = spk_count_by_trial / self.secs_per_bin
        mean_firing_rate = np.mean(firing_rate_by_trial, 1)
        # smooth with a moving average
        # check nothing in kwargs first
        window_len = kwargs.get("window_len", 5)
        window = kwargs.get("window", "flat")

        kernel = Box1DKernel(window_len)
        if "gauss" in window:
            kernel = Gaussian1DKernel(1, x_size=window_len)
        do_smooth = kwargs.get("do_smooth", True)

        if do_smooth:
            smoothed_binned_spikes = convolve(mean_firing_rate, kernel, boundary="wrap")
        else:
            smoothed_binned_spikes = mean_firing_rate
        nbins = np.floor(np.sum(np.abs(self.event_window)) / self.secs_per_bin)
        bins = np.linspace(self.event_window[0], self.event_window[1], int(nbins))
        # normalize all activity by activity in the time before
        # the laser onset
        idx = bins < 0
        normd = min_max_norm(
            smoothed_binned_spikes,
            np.min(smoothed_binned_spikes[idx]),
            np.max(smoothed_binned_spikes[idx]),
        )
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
        normd_masked = np.ma.masked_inside(normd, -threshold, 1 + threshold)
        # find the contiguous runs in the masked array
        # that are at least as long as the min_contiguous value
        # and classify this as a True response
        # set up the return variables here and set /return appropriately
        # in the conditional code below
        responds_to_stim = False
        mag = 0
        Response = namedtuple(
            "Response", ["responds", "normed_response_curve", "response_magnitude"]
        )
        this_response = Response(responds_to_stim, normd, mag)
        slices = np.ma.notmasked_contiguous(normd_masked)
        if slices and np.any(np.isfinite(normd)):
            # make sure that slices are within the first 25ms post-stim
            if ~np.any([s.start >= 50 and s.start <= 75 for s in slices]):
                return this_response
            max_runlength = max([len(normd_masked[s]) for s in slices])
            if max_runlength >= min_contiguous:
                if not return_activity:
                    this_response = Response(True, normd, mag)
                    return this_response
                else:
                    if return_magnitude:
                        sl = [
                            slc
                            for slc in slices
                            if (slc.stop - slc.start) == max_runlength
                        ]
                        mag = [-1 if np.mean(normd[sl[0]]) < 0 else 1][0]
                        this_response = Response(True, normd, mag)
                        return this_response
                    else:
                        this_response = Response(True, normd, mag)
                        return this_response
        return this_response

    def theta_mod_idx(self, **kwargs) -> float:
        """
        Calculates a theta modulation index of a spike train based on the cells
        autocorrelogram.

        The difference of the mean power in the theta band (6-11 Hz) and
        the mean power in the 1-50 Hz band is divided by their sum to give
        a metric that lives between 0 and 1

        Returns
        -------
        float
            The difference of the values at the first peak
            and trough of the autocorrelogram.

        Notes
        -----
        This is a fairly skewed metric with a distribution strongly biased
        to -1 (although more evenly distributed than theta_mod_idxV2 below)
        """
        ac = self.acorr(**kwargs)
        # Take the fft of the spike train autocorr (from -500 to +500ms)
        from scipy.signal import periodogram

        fs = 1.0 / kwargs.get("binsize", 0.0001)
        freqs, power = periodogram(ac.binned_data[0], fs=fs, return_onesided=True)
        # Smooth the power over +/- 1Hz when fs=200
        b = signal.windows.boxcar(3)  # another filter type - blackman?
        h = signal.filtfilt(b, 3, power)

        # Square the amplitude first to get power
        sqd_amp = h**2
        # Get the mean theta band power - mtbp
        mtbp = np.mean(sqd_amp[np.logical_and(freqs >= 6, freqs <= 11)])
        # Find the mean amplitude in the 1-50Hz range
        # Get the mean in the other band - mobp
        mobp = np.mean(sqd_amp[np.logical_and(freqs > 2, freqs < 50)])
        # Find the ratio of these two - this is the theta modulation index
        return float((mtbp - mobp) / (mtbp + mobp))

    def theta_mod_idxV2(self) -> float:
        """
        This is a simpler alternative to the theta_mod_idx method in that it
        calculates the difference between the normalized temporal
        autocorrelogram at the trough between 50-70ms and the
        peak between 100-140ms over their sum (data is binned into 5ms bins)

        Returns
        -------
        float
            The difference of the values at the first peak
            and trough of the autocorrelogram.

        Notes
        -----
        Measure used in Cacucci et al., 2004 and Kropff et al 2015
        """
        ac = self.acorr()
        bins = ac.bin_edges[0]
        corr = ac.binned_data[0]
        # 'close' the right-hand bin
        bins = bins[0:-1]
        # normalise corr so max is 1.0
        corr = corr / float(np.max(corr))
        thetaAntiPhase = np.min(
            corr[np.logical_and(bins > 50 / 1000.0, bins < 70 / 1000.0)]
        )
        thetaPhase = np.max(
            corr[np.logical_and(bins > 100 / 1000.0, bins < 140 / 1000.0)]
        )
        return float((thetaPhase - thetaAntiPhase) / (thetaPhase + thetaAntiPhase))

    def theta_mod_idxV3(self, **kwargs) -> float:
        """
        Another theta modulation index score this time based on the method used
        by Kornienko et al., (2024) (Kevin Allens lab)
        see https://doi.org/10.7554/eLife.35949.001

        Uses the binned spike train instead of the autocorrelogram as
        the input to the periodogram function (they use pwelch in R;
        periodogram is a simplified call to welch in scipy.signal)

        The resulting metric is similar to that in theta_mod_idx above except
        that the frequency bands compared to the theta band are narrower and
        exclusive of the theta band

        Produces a fairly normally distributed score with a mean and median
        pretty close to 0

        Parameters
        ----------
        **kwargs
            Passed into get_ifr_power_spectrum

        Returns
        -------
        float
            The difference of the values at the first peak
            and trough of the autocorrelogram.

        """
        freqs, power = self.get_ifr_power_spectrum(**kwargs)
        # smooth with a boxcar filter with a 0.5Hz window
        win_len = np.count_nonzero(np.logical_and(freqs >= 0, freqs <= 0.5))
        w = signal.windows.boxcar(win_len)
        b = signal.filtfilt(w, 1, power)
        sqd_amp = b**2
        mtbp = np.mean(sqd_amp[np.logical_and(freqs >= 6, freqs <= 10)])
        mobp = np.mean(
            sqd_amp[
                np.logical_or(
                    np.logical_and(freqs > 3, freqs < 5),
                    np.logical_and(freqs > 11, freqs < 13),
                )
            ]
        )
        return float((mtbp - mobp) / (mtbp + mobp))

    def get_ifr_power_spectrum(self) -> tuple[np.ndarray, ...]:
        """
        Returns the power spectrum of the instantaneous firing rate of a cell

        Used to calculate the theta_mod_idxV3 score above

        Returns
        -------
        tuple of np.ndarray
            The frequency and power of the instantaneous firing rate
        """
        binned_spikes = np.bincount(
            np.array(self.spike_times * self.pos_sample_rate, dtype=int).ravel(),
            minlength=int(self.pos_sample_rate * self.duration),
        )
        # possibly smooth the spike train...
        freqs, power = signal.periodogram(binned_spikes, fs=self.pos_sample_rate)
        freqs = freqs.ravel()
        power = power.ravel()
        return freqs, power

    def theta_band_max_freq(self):
        """
        Calculates the frequency with the maximum power in the theta band (6-12Hz)
        of a spike train's autocorrelogram.

        This function is used to look for differences in theta frequency in
        different running directions as per Blair.
        See Welday paper - https://doi.org/10.1523/jneurosci.0712-11.2011

        Returns
        -------
        float
            The frequency with the maximum power in the theta band.

        Raises
        ------
        ValueError
            If the input spike train is not valid.
        """
        ac = self.acorr()
        # Take the fft of the spike train autocorr (from -500 to +500ms)
        from scipy.signal import periodogram

        freqs, power = periodogram(ac.binned_data[0], fs=200, return_onesided=True)
        power_masked = np.ma.MaskedArray(power, np.logical_or(freqs < 6, freqs > 12))
        return freqs[np.argmax(power_masked)]

    def smooth_spike_train(self, npos, sigma=3.0, shuffle=None):
        """
        Returns a spike train the same length as num pos samples that has been
        smoothed in time with a gaussian kernel M in width and standard
        deviation equal to sigma.

        Parameters
        ----------
        npos : int
            The number of position samples captured.
        sigma : float, default=3.0
            The standard deviation of the gaussian used to
            smooth the spike train.
        shuffle : int, default=None
            The number of seconds to shift the spike
            train by. Default is None.

        Returns
        -------
        np.ndarray
            The smoothed spike train.
        """
        spk_hist = np.bincount(self.spike_times, minlength=npos)
        if shuffle is not None:
            spk_hist = np.roll(spk_hist, int(shuffle * 50))
        # smooth the spk_hist (which is a temporal histogram) with a 250ms
        # gaussian as with Kropff et al., 2015
        h = signal.windows.gaussian(13, sigma)
        h = h / float(np.sum(h))
        return signal.filtfilt(h.ravel(), 1, spk_hist)

    def contamination_percent(self, **kwargs) -> tuple:
        """
        Returns the contamination percentage of a spike train.

        Parameters
        ----------
        **kwargs
            Passed into the contamination_percent function.

        Returns
        -------
        tuple of float
            Q - A measure of refractoriness.
            R - A second measure of refractoriness (kicks in for very low firing rates).

        """

        _, Qi, Q00, Q01, Ri = contamination_percent(self.spike_times, **kwargs)
        # this is a measure of refractoriness
        Q = np.nanmin([Qi / (np.nanmax([Q00, Q01]))])
        # this is a second measure of refractoriness (kicks in for very low
        # firing rates)
        R = np.nanmin(Ri)
        return Q, R

    def plot_waveforms(self, n_waveforms: int = 2000, n_channels: int = 4):
        """
        Plots the waveforms of the cluster.

        Parameters
        ----------
        n_waveforms : int, optional
            The number of waveforms to plot.
        n_channels : int, optional
            The number of channels to plot.

        Returns
        -------
        None
        """
        if self._waves is None:
            raise ValueError("No waveforms available for this cluster.")

        from matplotlib import pyplot as plt

        fig, axes = plt.subplots(n_channels, 1, figsize=(5, 10))
        for i in range(n_channels):
            axes[i].plot(self.waveforms()[:n_waveforms, i, :].T, c="gray")
            # plot mean waveform on top
            axes[i].plot(np.mean(self.waveforms()[:n_waveforms, i, :], axis=0), c="red")
            axes[i].set_title(f"Channel {i}")
            axes[i].set_xlabel("Time (ms)")
            axes[i].set_ylabel("Amplitude (uV)")
        plt.tight_layout()
        plt.show()


class SpikeCalcsAxona(SpikeCalcsGeneric):
    """
    Replaces SpikeCalcs from ephysiopy.axona.spikecalcs
    """

    def __init__(
        self,
        spike_times: np.ndarray,
        cluster: int,
        waveforms: np.ndarray = None,
        *args,
        **kwargs,
    ):
        super().__init__(spike_times, cluster, waveforms, *args, **kwargs)

    def half_amp_dur(self, waveforms) -> float:
        """
        Calculates the half amplitude duration of a spike.

        Parameters
        ----------
        A : np.ndarray
            An nSpikes x nElectrodes x nSamples array.

        Returns
        -------
        float
            The half-amplitude duration for the channel
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

    def p2t_time(self, waveforms) -> float:
        """
        The peak to trough time of a spike in ms

        Parameters
        ----------
        cluster : int
            The cluster whose waveforms are to be analysed

        Returns
        -------
        float
            The mean peak-to-trough time for the channel
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

    def plotClusterSpace(
        self,
        waveforms,
        param="Amp",
        clusts: None | int | list = None,
        cluster_vec: None | np.ndarray | list = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Assumes the waveform data is signed 8-bit ints

        NB THe above assumption is mostly broken as waveforms by default are now
        in volts so you need to construct the trial object (AxonaTrial, OpenEphysBase
        etc) with volts=False (works for Axona, less sure about OE)
        TODO: aspect of plot boxes in ImageGrid not right as scaled by range of
        values now

        Parameters
        ----------
        waveforms : np.ndarray
            the array of waveform data. For Axona recordings this
            is nSpikes x nChannels x nSamplesPerWaveform
        param : str
            the parameter to plot. See get_param at the top of this file
            for valid args
        clusts : int, list or None, default None
            which clusters to colour in
        cluster_vec : np.ndarray, list or None, default None
            the cluster identity of each spike in waveforms must be nSpikes long
        **kwargs
            passed into ImageGrid
        """
        if cluster_vec is not None:
            assert np.shape(waveforms)[0] == len(cluster_vec)

        from itertools import combinations

        from mpl_toolkits.axes_grid1 import ImageGrid
        from matplotlib.collections import RegularPolyCollection
        from ephysiopy.axona.tintcolours import colours as tcols

        try:
            from numpy.lib.arraysetops import isin
        except ImportError:
            from numpy import isin as isin

        c_vec = np.zeros(shape=(np.shape(waveforms)[0]))
        if clusts is not None:
            idx = isin(cluster_vec, clusts)
            c_vec[idx] = cluster_vec[idx]
        c_vec = [[np.floor(t * 255) for t in tcols[i]] for i in c_vec.astype(int)]

        self.scaling = np.full(4, 15)

        amps = get_param(waveforms, param=param)
        cmb = combinations(range(4), 2)
        if "fig" in kwargs:
            fig = kwargs["fig"]
        else:
            fig = plt.figure(figsize=(8, 6))
        grid = ImageGrid(fig, 111, nrows_ncols=(2, 3), axes_pad=0.1, aspect=False)
        for i, c in enumerate(cmb):
            if np.sum(amps[:, c[0]]) > 0 and np.sum(amps[:, c[1]]) > 0:
                xy = np.array([amps[:, c[0]], amps[:, c[1]]]).T
                rects = RegularPolyCollection(
                    numsides=4,
                    rotation=0,
                    facecolors=c_vec,
                    edgecolors=c_vec,
                    offsets=xy,
                    offset_transform=grid[i].transData,
                )
                grid[i].add_collection(rects)
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
            grid[i].set_xlim(0, 256)
            grid[i].set_ylim(0, 256)
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
    ) -> np.ndarray:
        """
        Returns waveforms for a cluster.

        Parameters
        ----------
        cluster : int
            The cluster to return the waveforms for.
        cluster_data : KiloSortSession
            The KiloSortSession object for the
            session that contains the cluster.
        n_waveforms : int, default=2000
            The number of waveforms to return.
        n_channels : int, default=64
            The number of channels in the recording.

        Returns
        -------
        np.ndarray
            The waveforms for the cluster.
        """
        # instantiate the TemplateModel - this is used to get the waveforms
        # for the cluster. TemplateModel encapsulates the results of KiloSort
        if self.TemplateModel is None:
            self.TemplateModel = TemplateModel(
                dir_path=os.path.join(cluster_data.fname_root),
                sample_rate=3e4,
                dat_path=os.path.join(cluster_data.fname_root, "continuous.dat"),
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
        return np.empty(0)

    def get_channel_depth_from_templates(self, pname: Path):
        """
        Determine depth of template as well as closest channel.

        Parameters
        ----------
        pname : Path
            The path to the directory containing the KiloSort results.

        Returns
        -------
        tuple of np.ndarray
            The depth of the template and the index of the closest channel.

        Notes
        -----
        Adopted from
        'templatePositionsAmplitudes' by N. Steinmetz
        (https://github.com/cortex-lab/spikes)
        """
        # Load inverse whitening matrix
        Winv = np.load(os.path.join(pname, "whitening_mat_inv.npy"))
        # Load templates
        templates = np.load(os.path.join(pname, "templates.npy"))
        # Load channel_map and positions
        channel_map = np.load(os.path.join(pname, "channel_map.npy"))
        channel_positions = np.load(os.path.join(pname, "channel_positions.npy"))
        map_and_pos = np.array([np.squeeze(channel_map), channel_positions[:, 1]])
        # unwhiten all the templates
        tempsUnW = np.zeros(np.shape(templates))
        for i in range(np.shape(templates)[0]):
            tempsUnW[i, :, :] = np.squeeze(templates[i, :, :]) @ Winv

        tempAmp = np.squeeze(np.max(tempsUnW, 1)) - np.squeeze(np.min(tempsUnW, 1))
        tempAmpsUnscaled = np.max(tempAmp, 1)
        # need to zero-out the potentially-many low values on distant channels
        threshVals = tempAmpsUnscaled * 0.3
        tempAmp[tempAmp < threshVals[:, None]] = 0
        # Compute the depth as a centre of mass
        templateDepths = np.sum(tempAmp * map_and_pos[1, :], -1) / np.sum(tempAmp, 1)
        maxChanIdx = np.argmin(
            np.abs((templateDepths[:, None] - map_and_pos[1, :].T)), 1
        )
        return templateDepths, maxChanIdx

    def get_template_id_for_cluster(self, pname: Path, cluster: int) -> int:
        """
        Determine the best channel (one with highest amplitude spikes)
        for a given cluster.

        Parameters
        ----------
        pname : Path
            The path to the directory containing the KiloSort results.
        cluster : int
            The cluster to get the template ID for.

        Returns
        -------
        int
            The template ID for the cluster.
        """
        spike_templates = np.load(os.path.join(pname, "spike_templates.npy"))
        spike_times = np.load(os.path.join(pname, "spike_times.npy"))
        spike_clusters = np.load(os.path.join(pname, "spike_clusters.npy"))
        cluster_times = spike_times[spike_clusters == cluster]
        rez_mat = h5py.File(os.path.join(pname, "rez.mat"), "r")
        st3 = rez_mat["rez"]["st3"]
        st_spike_times = st3[0, :]
        idx = np.searchsorted(st_spike_times, cluster_times)
        template_idx, counts = np.unique(spike_templates[idx], return_counts=True)
        ind = np.argmax(counts)
        return template_idx[ind]


class SpikeCalcsProbe(SpikeCalcsGeneric):
    """
    Encapsulates methods specific to probe-based recordings
    """

    def __init__(self):
        pass
