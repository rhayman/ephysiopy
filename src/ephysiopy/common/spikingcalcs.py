import warnings
from collections import namedtuple
from astropy.convolution import convolve, Box1DKernel, Gaussian1DKernel
import seaborn as sns
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.transforms as transforms
import numpy as np
from scipy.special import erf
from scipy import signal, stats
from ephysiopy.common.utils import (
    min_max_norm,
    shift_vector,
    BinnedData,
    VariableToBin,
    MapType,
    clean_kwargs,
)


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

    if isi_matrix.ndim == 1:
        isi_matrix = isi_matrix.reshape(1, -1)

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
        _, bins, patches = ax3.hist(
            pca_distances, bins=150, density=True, color="blue")
        for bin_patch in zip(bins[:-1], patches):
            if bin_patch[0] < lda.intercept_:
                bin_patch[1].set_facecolor("red")
        axtrans = transforms.blended_transform_factory(
            ax3.transData, ax3.transAxes)
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
        axtrans = transforms.blended_transform_factory(
            ax4.transData, ax4.transAxes)
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

    from ephysiopy.common.waveformcalcs import get_param

    if waveforms is None:
        return None
    nSpikes, nElectrodes, _ = waveforms.shape
    wvs = waveforms.copy()
    E = np.sqrt(np.nansum(waveforms**2, axis=2))
    zeroIdx = np.sum(E, 0) == [0, 0, 0, 0]
    E = E[:, ~zeroIdx]
    if not np.any(E):
        return None
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
        y.extend((x1[t[0]: t[1]] - x2[i]))
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
        all = np.array([np.logical_and(bins >= i[0], bins < i[1])
                       for i in vals])
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
    Q01 = np.nanmax([get_normd_shoulder(inner_left),
                    get_normd_shoulder(inner_right)])

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


class SpikeCalcsGeneric:
    def __init__(
        self,
        spike_times: np.ndarray,
        cluster: int | None = None,
        event_ts: np.ndarray | None = None,
        event_window: np.ndarray = np.array([-0.5, 0.5]),
        pos_sample_rate: float = 50.0,
        sample_rate: float = 30000.0,
    ):
        self.spike_times = spike_times
        self._event_ts = event_ts
        self.cluster = cluster
        self.event_window = event_window
        self.pos_sample_rate = pos_sample_rate
        self.sample_rate = sample_rate
        self.pos_sample_rate = pos_sample_rate
        self._secs_per_bin = None
        self._duration = None

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
    def duration(self) -> float | int | None:
        return self._duration

    @property
    def n_pos_samples(self) -> int | None:
        if self.duration is None:
            raise IndexError("No duration provided, give me one!")
        return int(self.duration * self.pos_sample_rate)

    @property
    def event_ts(self) -> np.ndarray | None:
        return self._event_ts

    @event_ts.setter
    def event_ts(self, value: np.ndarray | None):
        self._event_ts = value

    @duration.setter
    def duration(self, value: float | int | None):
        self._duration = value

    @property
    def secs_per_bin(self) -> float | int:
        return self._secs_per_bin

    @secs_per_bin.setter
    def secs_per_bin(self, value: float | int):
        self._secs_per_bin = value

    def trial_mean_fr(self) -> float:
        # Returns the trial mean firing rate for the cluster
        if self.duration is None:
            raise IndexError("No duration provided, give me one!")
        return self.n_spikes / self.duration

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

    def shuffle_isis(self) -> np.ndarray:
        """
        Shuffle the spike train ISIs and return a new spike train with
        the same number of spikes but a different ISI distributedon.
        Useful for creating null distributions of spike metrics.
        """
        ts = self.spike_times
        isis = np.diff(ts)
        new_ts = np.cumsum(np.random.permutation(isis))
        # TODO: make sure the start and stop time range is within
        # the bounds of the original
        return new_ts

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
            tmp = self.spike_times[t[0]: t[1]] - event_ts[i]
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
        if self.event_ts is None:
            raise Exception("Need some event timestamps! Aborting")

        event_ts = self.event_ts
        event_ts.sort()
        if isinstance(event_ts, list):
            event_ts = np.array(event_ts)

        irange = event_ts[:, np.newaxis] + self.event_window[np.newaxis, :]
        dts = np.searchsorted(self.spike_times, irange)
        bins = np.arange(
            self.event_window[0],
            self.event_window[1] - self.secs_per_bin,
            bin_width_secs,
        )
        result = np.empty(shape=(len(bins), len(event_ts)), dtype=np.int64)
        for i, t in enumerate(dts):
            tmp = self.spike_times[t[0]: t[1]] - event_ts[i]
            indices = np.digitize(tmp, bins=bins) - 1
            counts = np.bincount(indices, minlength=len(bins))
            result[:, i] = counts
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
        method = stats.PermutationMethod(
            n_resamples=nShuffles, random_state=rng)
        method = kwargs.get("method", method)
        res = stats.pearsonr(
            sm_spk_rate.compressed(), speed_filt.compressed(), method=method
        )
        return res

    def get_ifr(self, spike_times: np.array, **kwargs) -> np.ndarray:
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
        spk_hist = np.bincount(x1, minlength=self.n_pos_samples)
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
    ) -> namedtuple:
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
            smoothed_binned_spikes = convolve(
                mean_firing_rate, kernel, boundary="wrap")
        else:
            smoothed_binned_spikes = mean_firing_rate
        nbins = np.floor(np.sum(np.abs(self.event_window)) / self.secs_per_bin)
        bins = np.linspace(
            self.event_window[0], self.event_window[1], int(nbins))
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
            "Response", ["responds", "normed_response_curve",
                         "response_magnitude"]
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

        fs = 1.0 / kwargs.get("binsize", 0.001)
        freqs, power = periodogram(
            ac.binned_data[0], fs=fs, return_onesided=True)
        # Smooth the power over +/- 1Hz
        win_size = np.count_nonzero(freqs <= 1)
        if win_size % 2 == 1:
            win_size += 1
        b = signal.windows.boxcar(win_size)  # another filter type - blackman?
        h = signal.filtfilt(b, win_size, power)

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
        low_band = kwargs.get("low_band", (0, 0.5))
        theta_band = kwargs.get("theta_band", (6, 10))
        kws = clean_kwargs(self.get_ifr_power_spectrum, kwargs)

        freqs, power = self.get_ifr_power_spectrum(**kws)
        # smooth with a boxcar filter with a 0.5Hz window
        win_len = np.count_nonzero(
            np.logical_and(freqs >= low_band[0], freqs <= low_band[1])
        )
        w = signal.windows.boxcar(win_len)
        b = signal.filtfilt(w, 1, power)
        sqd_amp = b**2
        mtbp = np.mean(
            sqd_amp[np.logical_and(
                freqs >= theta_band[0], freqs <= theta_band[1])]
        )
        mobp = np.mean(
            sqd_amp[
                np.logical_or(
                    np.logical_and(
                        freqs > theta_band[0] - 3, freqs < theta_band[0]),
                    np.logical_and(
                        freqs > theta_band[1], freqs < theta_band[1] + 3),
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
            np.array(self.spike_times * self.pos_sample_rate,
                     dtype=int).ravel(),
            minlength=int(self.pos_sample_rate * self.duration),
        )
        # possibly smooth the spike train...
        freqs, power = signal.periodogram(
            binned_spikes, fs=self.pos_sample_rate)
        freqs = freqs.ravel()
        power = power.ravel()
        return freqs, power

    def theta_band_max_freq(self, **kws):
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

        theta_band = kws.get("theta_band", (6, 12))

        freqs, power = periodogram(
            ac.binned_data[0], fs=200, return_onesided=True)
        power_masked = np.ma.MaskedArray(
            power, np.logical_or(freqs < theta_band[0], freqs > theta_band[1])
        )
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
