from dataclasses import dataclass, field
import numpy as np
import astropy.convolution as cnv
import skimage
from collections import defaultdict, namedtuple
import inspect
from enum import Enum
import copy
from pathlib import Path
import os


class VariableToBin(Enum):
    """
    Holds a human readable representation of the variable being binned
    """

    XY = 1
    DIR = 2
    SPEED = 3
    XY_TIME = 4
    SPEED_DIR = 5
    EGO_BOUNDARY = 6
    TIME = 7
    X = 8  # linear track
    Y = 9  # linear track
    PHI = 10  # linear track sqrt(x^2 + y^2)


class MapType(Enum):
    """
    A human readable representation of the map type
    """

    RATE = 1
    POS = 2
    SPK = 3
    ADAPTIVE = 4
    AUTO_CORR = 5
    CROSS_CORR = 6


# A named tuple used in BinnedData for uniquely identifying a cluster/ channel
ClusterID = namedtuple("ClusterID", ["Cluster", "Channel"])


@dataclass
class BinnedData:
    """
    A dataclass to store binned data. The binned data is stored in a list of
    numpy arrays. The bin edges are stored in a list of numpy arrays. The
    variable to bin is stored as an instance of the VariableToBin enum.
    The map type is stored as an instance of the MapType enum.
    The binned data and bin edges are initialized as
    empty lists. bin_units is how to conver the binned data
    to "real" units e.g. for XY it might be how to convert to cms,
    for time to seconds etc. You multiply the binned data by that
    number to get the real values. Note that this might not make sense
    / be obvious for some binning (i.e. SPEED_DIR)

    The BinnedData class is the output of the main binning function in the
    ephysiopy.common.binning.RateMap class. It is used to store the binned data
    as a convenience mostly for easily iterating over the binned data and
    using the bin_edges to plot the data.
    As such, it is used as a convenience for plotting as the bin edges
    are used when calling pcolormesh in the plotting functions.
    """

    variable: VariableToBin = VariableToBin.XY
    map_type: MapType = MapType.RATE
    binned_data: list[np.ma.MaskedArray] = field(default_factory=list)
    bin_edges: list[np.ndarray] = field(default_factory=list)
    cluster_id: list[ClusterID] = field(default_factory=list)

    def __init__(
        self,
        variable: VariableToBin,
        map_type: MapType,
        binned_data: list[np.ma.MaskedArray],
        bin_edges: list[np.ndarray],
        cluster_id: list[ClusterID] = ClusterID(0, 0),
    ):
        if isinstance(binned_data, np.ndarray) or isinstance(
            binned_data, np.ma.MaskedArray
        ):
            binned_data = [binned_data]

        if isinstance(bin_edges, np.ndarray):
            bin_edges = [bin_edges]

        if isinstance(cluster_id, int):
            cluster_id = [cluster_id]

        self.variable = variable
        self.map_type = map_type
        self.binned_data = [np.ma.masked_invalid(bd) for bd in binned_data]
        self.bin_edges = bin_edges
        if cluster_id is not None:
            self.cluster_id = cluster_id

    def __iter__(self):
        current = 0
        while current < len(self.binned_data):
            yield self.__getitem__(current)
            current += 1

    def __assert_equal_bin_edges__(self, other):
        assert np.all(
            [np.all(s == o) for s, o in zip(self.bin_edges, other.bin_edges)]
        ), "Bin edges do not match"

    def __len__(self):
        return len(self.binned_data)

    def __getitem__(self, i):
        """
        Returns a specified index of the binned_data as a BinnedData instance.
        The data in binned_data is a deep copy of the original so can be
        modified without affecting the original.

        Parameters
        ----------
        i : int
            The index of binned_data to return
        """
        # this try/except fixes an error when generating
        # shuffled data
        try:
            cluster_id = self.cluster_id[i]
        except IndexError:
            cluster_id = 0

        return BinnedData(
            variable=self.variable,
            map_type=self.map_type,
            binned_data=[copy.deepcopy(self.binned_data[i])],
            bin_edges=self.bin_edges,
            cluster_id=cluster_id,
        )

    def __truediv__(self, other):
        """
        Divides the binned data by the binned data of
        another BinnedData instance i.e. spike data / pos data to get
        a rate map.

        Parameters
        ----------
        other : BinnedData
            the denominator
        """
        if isinstance(other, BinnedData):
            self.__assert_equal_bin_edges__(other)
            if len(self.binned_data) > len(other.binned_data):
                if (other.map_type.value == MapType.POS.value) and (
                    self.map_type.value == MapType.SPK.value
                ):
                    if len(other.binned_data) == 1:
                        return BinnedData(
                            variable=self.variable,
                            map_type=MapType.RATE,
                            binned_data=[
                                a / b
                                for a in self.binned_data
                                for b in other.binned_data
                            ],
                            bin_edges=self.bin_edges,
                            cluster_id=self.cluster_id,
                        )

            return BinnedData(
                variable=self.variable,
                map_type=MapType.RATE,
                binned_data=[
                    a / b for a, b in zip(self.binned_data, other.binned_data)
                ],
                bin_edges=self.bin_edges,
                cluster_id=self.cluster_id,
            )

    def __add__(self, other):
        """
        Adds the binned_data of another BinnedData instance
        to the binned_data of this instance.

        Parameters
        ----------
        other : BinnedData
            The instance to add to the current one

        """
        if isinstance(other, BinnedData):
            self.__assert_equal_bin_edges__(other)
            return BinnedData(
                variable=self.variable,
                map_type=self.map_type,
                binned_data=self.binned_data + other.binned_data,
                bin_edges=self.bin_edges,
                cluster_id=self.cluster_id + other.cluster_id,
            )

    def __eq__(self, other) -> bool:
        """
        Checks for equality of two instances of BinnedData
        """
        assert isinstance(other, BinnedData)
        self.__assert_equal_bin_edges__(other)
        if np.all(
            [
                np.all(np.isfinite(sbd) == np.isfinite(obd))
                for sbd, obd in zip(self.binned_data, other.binned_data)
            ]
        ):
            if np.all(
                [
                    np.all(sbd[np.isfinite(sbd)] == obd[np.isfinite(obd)])
                    for sbd, obd in zip(self.binned_data, other.binned_data)
                ]
            ):
                return True
            else:
                return False
        else:
            return False

    def get_cluster(self, id: ClusterID):
        """
        Returns the binned data for the specified cluster id

        Parameters
        ----------
        id : ClusterID
            The cluster id to return

        Returns
        -------
        BinnedData
            A new BinnedData instance with the binned data for
            the specified cluster id

        """
        if id in self.cluster_id:
            idx = self.cluster_id.index(id)
            return BinnedData(
                variable=self.variable,
                map_type=self.map_type,
                binned_data=[self.binned_data[idx]],
                bin_edges=self.bin_edges,
                cluster_id=id,
            )

    def set_nan_indices(self, indices):
        """
        Sets the values of the binned data at the specified indices to NaN.

        Parameters
        ----------
        indices : np.ndarray
            The indices to convert to NaN
        """
        for i in range(len(self.binned_data)):
            self.binned_data[i].mask[indices] = True

    def T(self):
        return BinnedData(
            variable=self.variable,
            map_type=self.map_type,
            binned_data=[a.T for a in self.binned_data],
            bin_edges=self.bin_edges[::-1],
            cluster_id=self.cluster_id,
        )

    def correlate(self, other=None, as_matrix=False) -> list[float] | np.ndarray:
        """
        This method is used to correlate the binned data of this BinnedData
        instance with the binned data of another BinnedData instance.

        Parameters
        ----------
        other : BinnedData
            The other BinnedData instance to correlate with.
            If None, then correlations are performed between all the data held
            in the list self.binned_data
        as_matrix : bool
            If True will return the full correlation matrix for
            all of the correlations in the list of data in self.binned_data. If
            False, a list of the unique correlations for the comparisons in
            self.binned_data are returned.

        Returns
        -------
        BinnedData
            A new BinnedData instance with the correlation of the
            binned data of this instance and the other instance.
        """
        if other is not None:
            assert isinstance(other, BinnedData)
            self.__assert_equal_bin_edges__(other)
        if other is not None:
            result = np.reshape(
                [corr_maps(a, b) for a in self.binned_data for b in other.binned_data],
                newshape=(len(self.binned_data), len(other.binned_data)),
            )
        else:
            result = np.reshape(
                [corr_maps(a, b) for a in self.binned_data for b in self.binned_data],
                newshape=(len(self.binned_data), len(self.binned_data)),
            )
        if as_matrix:
            return result
        else:
            # pick out the relevant diagonal
            k = -1
            if len(self.binned_data) == 1:
                k = 0
            if other is not None:
                if len(other.binned_data) == 1:
                    k = 0
                idx = np.tril_indices(
                    n=len(self.binned_data), m=len(other.binned_data), k=k
                )
            else:
                idx = np.tril_indices(n=len(self.binned_data), k=k)
            return result[idx]


def cluster_intersection(A: BinnedData, B: BinnedData):
    """
    Gets the intersection of clusters between two instances
    of BinnedData.

    Parameters
    ----------
    A, B : BinnedData
        The two instances

    Returns
    -------
    A, B : BinnedData
        The modified instances with only the overlapping clusters
        present in both
    """
    A_ids = [str(c.Cluster) + "_" + str(c.Channel) for c in A.cluster_id]
    B_ids = [str(c.Cluster) + "_" + str(c.Channel) for c in B.cluster_id]

    common_ids = list(set(A_ids).intersection(B_ids))
    # sort numerically
    common_ids = sorted(common_ids, key=int)

    A_out = BinnedData(A.variable, A.map_type, [], A.bin_edges, [])
    B_out = BinnedData(B.variable, B.map_type, [], B.bin_edges, [])

    for id in common_ids:
        if id in A_ids:
            idx = A_ids.index(id)
            A_out.binned_data.append(A.binned_data[idx])
            A_out.cluster_id.append(A.cluster_id[idx])
        if id in B_ids:
            idx = B_ids.index(id)
            B_out.binned_data.append(B.binned_data[idx])
            B_out.cluster_id.append(B.cluster_id[idx])

    return A_out, B_out


@dataclass(eq=True)
class TrialFilter:
    """
    A basic dataclass for holding filter values

    Units:
        time: seconds
        dir: degrees
        speed: cm/s
        xrange/ yrange: cm

    """

    name: str
    start: float | str
    end: float | str

    def __init__(self, name: str, start: float | str, end: float | str = None):
        """
        Parameters
        ----------
        name : str
            The name of the filter type
        start : float, str
            start value of filter
        end : float, str
            end value of filter
        """
        assert name in [
            "time",
            "dir",
            "speed",
            "xrange",
            "yrange",
            "phi",
        ], "name must be one of 'time', 'dir', 'speed', 'xrange', 'yrange', 'phi'"
        self.name = name
        self.start = start
        self.end = end


def filter_data(data: np.ndarray, f: TrialFilter) -> np.ndarray:
    """
    Filters the input data based on the specified TrialFilter.

    Parameters
    ----------
    data : np.ndarray
        The data to filter.
    f : TrialFilter
        The filter to apply.

    Returns
    -------
    np.ndarray
        A boolean array where Trues are the 'to-be' masked values.

    Notes
    -----
    When calculating the filters, be sure to do the calculations on the
    'data' property of the masked arrays so you get access to the
    underlying data without the mask.

    This function is used in io.recording to filter the data
    """
    if f.name == "dir":
        # modify values based if first is a str
        if isinstance(f.start, str):
            if "w" in f.start:
                f.start = 135  # rotated: 225 - 180 # orig 45
                f.end = 225  # rotated: 315 - 180 # orig 135
            elif "e" in f.start:
                f.start = 315  # rotated: 45 - 180 # orig 225
                f.end = 45  # rotated: 135 - 180 # orig 315
            elif "s" in f.start:
                f.start = 225  # rotated: 315 - 180 # orig 135
                f.end = 315  # rotated: 45 - 180 # orig 225
            elif "n" in f.start:
                f.start = 45  # rotated: 135 - 180 # orig 315
                f.end = 135  # rotated: 225 - 180 # orig 45
            else:
                raise ValueError("Invalid direction")

        if f.start < f.end:
            return np.logical_and(data >= f.start, data <= f.end)
        else:
            # Handle the case where the direction wraps around
            return np.logical_or(data >= f.start, data <= f.end)
    else:
        return np.logical_and(data >= f.start, data <= f.end)


def filter_trial_by_time(
    duration: int | float, how: str = "in_half"
) -> tuple[list[TrialFilter], ...]:
    """
    Filters the data in trial by time

    Parameters
    ----------
    duration - the duration of the trial in seconds

    how (str) - how to split the trial.
                Legal values: "in_half" or "odd_even"
                "in_half" filters for first n seconds and last n second
                "odd_even" filters for odd vs even minutes

    Returns
    -------
    tuple of TrialFilter
        A tuple of TrialFilter instances, one for each half or odd/even minutes
    """
    assert how in ["in_half", "odd_even"]

    if "in_half" in how:
        first_half = TrialFilter("time", 0, duration / 2.0)
        last_half = TrialFilter("time", duration / 2.0, duration)

        return [first_half], [last_half]
    else:
        r1 = np.arange(0, duration, 2)
        r2 = np.arange(1, duration, 2)
        evens = [(s, e) for s, e in zip(r1, r2)]
        odds = [(s, e) for s, e in zip(r2, r1[1::])]
        even_filters = [TrialFilter("time", s, e) for s, e in evens]
        odd_filters = [TrialFilter("time", s, e) for s, e in odds]

        return list(even_filters), list(odd_filters)


def memmapBinaryFile(path2file: Path, n_channels=384, **kwargs) -> np.ndarray:
    """
    Returns a numpy memmap of the int16 data in the
    file path2file, if present

    Parameters
    ----------
    path2file : Path
        The location of the file to be mapped
    n_channels : int
        the number of channels (size of the second dimension)
    **kwargs
        'data_type' : np.dtype, default np.int16
            The data type of the file to be mapped.

    Returns
    -------
    np.memmap
        The memory mapped data file
    """
    import os

    if "data_type" in kwargs.keys():
        data_type = kwargs["data_type"]
    else:
        data_type = np.int16

    if os.path.exists(path2file):
        # make sure n_channels is int as could be str
        n_channels = int(n_channels)
        status = os.stat(path2file)
        n_samples = int(status.st_size / (2.0 * n_channels))
        mmap = np.memmap(
            path2file, data_type, "r", 0, (n_channels, n_samples), order="F"
        )
        return mmap
    else:
        return np.empty(0)


def fileContainsString(pname: str, searchStr: str) -> bool:
    """
    Checks if the search string is contained in a file

    Parameters
    ----------
    pname : str
        The file to look in
    searchStr : str
        The string to look for

    Returns
    -------
    bool
        Whether the string was found or not
    """
    if os.path.exists(pname):
        with open(pname, "r") as f:
            strs = f.read()
        lines = strs.split("\n")
        found = False
        for line in lines:
            if searchStr in line:
                found = True
        return found
    else:
        return False


def clean_kwargs(func, kwargs):
    """
    This function is used to remove any keyword arguments that are not
    accepted by the function. It is useful for passing keyword arguments
    to other functions without having to worry about whether they are
    accepted by the function or not.

    Parameters
    ----------
    func : function
        The function to check for keyword arguments.
    **kwargs
        The keyword arguments to check.

    Returns
    -------
    dict
        A dictionary containing only the keyword arguments that are
        accepted by the function.
    """
    valid_kwargs = inspect.getfullargspec(func).kwonlyargs
    return {k: v for k, v in kwargs.items() if k in valid_kwargs}


def get_z_score(x: np.ndarray, mean=None, sd=None, axis: int = 0) -> np.ndarray:
    """
    Calculate the z-scores for array x based on the mean
    and standard deviation in that sample, unless stated

    Parameters
    ----------
    x : np.ndarray
        The array to z-score
    mean : float, optional
        The mean of x. Calculated from x if not provided
    sd : float, optional
        The standard deviation of x. Calculated from x if not provided
    axis : int
        The axis along which to operate

    Returns
    -------
    np.ndarray
        The z-scored version of the input array x
    """
    if mean is None:
        mean = np.nanmean(x, axis=axis)
    if sd is None:
        sd = np.nanstd(x, axis=axis)
    if axis == -1:
        return (x - mean[..., None]) / sd[..., None]
    return (x - mean) / sd


def mean_norm(x: np.ndarray, mn=None, axis: int = 0) -> np.ndarray:
    """
    Mean normalise an input array

    Parameters
    ----------
    x : np.ndarray
        The array t normalise
    mn : float, optional
        The mean of x
    axis : int
        The axis along which to operate

    Returns
    -------
    np.ndarray
        The mean normalised version of the input array
    """

    if mn is None:
        mn = np.mean(x, axis)
    x = (x - mn) / (np.max(x, axis) - np.min(x, axis))
    return x


def min_max_norm(x: np.ndarray, min=None, max=None, axis: int = 0) -> np.ndarray:
    """
    Normalise the input array x to lie between min and max

    Parameters
    ----------
    x : np.ndarray
        the array to normalise
    min : float
        the minimun value in the returned array
    max : float
        the maximum value in the returned array
    axis : int
        the axis along which to operate. Default 0

    Returns
    -------
    np.ndarray
        the normalised array
    """
    if min is None:
        min = np.nanmin(x, axis)
    if max is None:
        max = np.nanmax(x, axis)
    if axis == -1:
        return (x - np.array(min)[..., None]) / np.array(max - min)[..., None]
    return (x - min) / (max - min)


def remap_to_range(x: np.ndarray, new_min=0, new_max=1, axis=0) -> np.ndarray:
    """
    Remap the values of x to the range [new_min, new_max].

    Parameters
    ----------
    x : np.ndarray
        the array to remap
    new_min : float
        the minimun value in the returned array
    max : float
        the maximum value in the returned array

    Returns
    -------
    np.ndarray
        The remapped values
    """
    min = np.nanmin(x, axis)
    max = np.nanmax(x, axis)
    if axis == -1:
        return np.array((x.T - min) / (max - min) * (new_max - new_min) + new_min).T
    return (x - min) / (max - min) * (new_max - new_min) + new_min


def flatten_list(list_to_flatten: list) -> list:
    """
    Flattens a list of lists

    Parameters
    ----------
    list_to_flatten : list
        the list to flatten

    Returns
    -------
    list
        The flattened list
    """
    try:
        return [item for sublist in list_to_flatten for item in sublist]
    except TypeError:
        return list_to_flatten


def smooth(x, window_len=9, window="hanning"):
    """
    Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.

    Parameters
    ----------
    x : np.ndarray
        The input signal.
    window_len : int
        The length of the smoothing window.
    window : str
        The type of window from 'flat', 'hanning', 'hamming',
        'bartlett', 'blackman'. 'flat' window will produce a moving average
        smoothing.

    Returns
    -------
    np.ndarray
        The smoothed signal.

    Examples
    --------
    >>> t=linspace(-2,2,0.1)
    >>> x=sin(t)+randn(len(t))*0.1
    >>> y=smooth(x)

    See Also
    --------
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
    numpy.convolve, scipy.signal.lfilter

    Notes
    -----
    The window parameter could be the window itself if an array instead of
    a string.
    """

    if isinstance(x, list):
        x = np.array(x)

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if len(x) < window_len:
        print("length of x: ", len(x))
        print("window_len: ", window_len)
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x

    if (window_len % 2) == 0:
        window_len = window_len + 1

    if window not in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError(
            "Window is on of 'flat', 'hanning', \
                'hamming', 'bartlett', 'blackman'"
        )

    if window == "flat":  # moving average
        w = np.ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")
    y = cnv.convolve(x, w / w.sum(), normalize_kernel=False, boundary="extend")
    # return the smoothed signal
    return y


def blur_image(
    im: BinnedData, n: int, ny: int = 0, ftype: str = "boxcar", **kwargs
) -> BinnedData:
    """
    Smooths all the binned_data in an instance of BinnedData
    by convolving with a filter.

    Parameters
    ----------
    im : BinnedData
        Contains the array to smooth.
    n, ny : int
        The size of the smoothing kernel.
    ftype : str
        The type of smoothing kernel. Either 'boxcar' or 'gaussian'.

    Returns
    -------
    BinnedData
        BinnedData instance with the smoothed data.

    Notes
    -----
    This essentially does the smoothing in-place
    """
    stddev = kwargs.pop("stddev", 3)
    boundary = kwargs.pop("boundary", "extend")
    n = int(n)
    if n % 2 == 0:
        n += 1
    if ny == 0:
        ny = n
    else:
        ny = int(ny)
        if ny % 2 == 0:
            ny += 1
    ndims = len(im.bin_edges)
    g = cnv.Box2DKernel(n)
    if "box" in ftype:
        if ndims == 1:
            g = cnv.Box1DKernel(n)
        if ndims == 2:
            g = np.atleast_2d(g)
    elif "gaussian" in ftype:
        if ndims == 1:
            g = cnv.Gaussian1DKernel(stddev, x_size=n)
        if ndims == 2:
            g = cnv.Gaussian2DKernel(stddev, x_size=n, y_size=ny)
    g = np.array(g)
    for i, m in enumerate(im.binned_data):
        sm = cnv.convolve(
            m, g, boundary=boundary, normalize_kernel=True, preserve_nan=True
        )
        im.binned_data[i] = np.ma.masked_invalid(sm)
    return im


def shift_vector(v, shift, maxlen=None):
    """
    Shifts the elements of a vector by a given amount.
    A bit like numpys roll function but when the shift goes
    beyond some limit that limit is subtracted from the shift.
    The result is then sorted and returned.

    Parameters
    ----------
    v : array_like
        The input vector.
    shift : int
        The amount to shift the elements.
    fill_value : int
        The value to fill the empty spaces.

    Returns
    -------
    np.ndarray
        The shifted vector.
    """
    if shift == 0:
        return v
    if maxlen is None:
        return v
    if isinstance(v, list):
        out = []
        if shift > 0:
            for _v in v:
                shifted = _v + shift
                shifted[shifted >= maxlen] -= maxlen
                out.append(np.sort(shifted))
            return out
    else:
        if shift > 0:
            shifted = v + shift
            shifted[shifted >= maxlen] -= maxlen
            return np.sort(shifted)


def count_to(n: np.ndarray) -> np.ndarray:
    """
    This function is equivalent to hstack((arange(n_i) for n_i in n)).
    It seems to be faster for some possible inputs and encapsulates
    a task in a function.

    Examples
    --------
    >>> n = np.array([0, 0, 3, 0, 0, 2, 0, 2, 1])
    >>> count_to(n)
    array([0, 1, 2, 0, 1, 0, 1, 0])
    """
    if n.ndim != 1:
        raise Exception("n is supposed to be 1d array.")

    n_mask = n.astype(bool)
    n_cumsum = np.cumsum(n)
    ret = np.ones(n_cumsum[-1] + 1, dtype=int)
    ret[n_cumsum[n_mask]] -= n[n_mask]
    ret[0] -= 1
    return np.cumsum(ret)[:-1]


def window_rms(a: np.ndarray, window_size: int | float) -> np.ndarray:
    """
    Calculates the root mean square of the input a over a window of
    size window_size

    Parameters
    ----------
    a : np.ndarray
        The input array
    window_size : int, float
        The size of the smoothing window

    Returns
    -------
    np.ndarray
        The rms'd result
    """
    window_size = int(window_size)
    a2 = np.power(a, 2)
    window = np.ones(window_size) / float(window_size)
    return np.sqrt(np.convolve(a2, window, "same"))


def find_runs(x):
    """
    Find runs of consecutive items in an array.

    Parameters
    ----------
    x : np.ndarray, list
        The array to search for runs in

    Returns
    -------
    run_values : np.ndarray
        the values of each run
    run_starts : np.ndarray
        the indices into x at which each run starts
    run_lengths : np.ndarray
        The length of each run

    Examples
    --------
    >>> n = np.array([0, 0, 3, 3, 0, 2, 0,0, 1])
    >>> find_runs(n)
    (array([0, 3, 0, 2, 0, 1]),
    array([0, 2, 4, 5, 6, 8]),
    array([2, 2, 1, 1, 2, 1]))

    Notes
    -----
    Taken from:
    https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
    """

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError("only 1D array supported")
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths


def repeat_ind(n: np.ndarray) -> np.ndarray:
    """
    Repeat a given index a specified number of times.

    The input specifies how many times to repeat the given index.
    It is equivalent to something like this:

    hstack((zeros(n_i,dtype=int)+i for i, n_i in enumerate(n)))

    But this version seems to be faster, and probably scales better.
    At any rate, it encapsulates a task in a function.

    Parameters
    ----------
    n : np.ndarray
        A 1D array where each element specifies the number of times to repeat its index.

    Returns
    -------
    np.ndarray
        A 1D array with indices repeated according to the input array.

    Examples
    --------
    >>> n = np.array([0, 0, 3, 0, 0, 2, 0, 2, 1])
    >>> repeat_ind(n)
    array([2, 2, 2, 5, 5, 7, 7, 8])
    """
    if n.ndim != 1:
        raise Exception("n is supposed to be 1d array.")

    res = [[idx] * a for idx, a in enumerate(n) if a != 0]
    return np.concatenate(res)


def rect(r, w, deg=False):
    """
    Convert from polar (r, w) to rectangular (x, y) coordinates.

    Parameters
    ----------
    r : float or np.ndarray
        Radial coordinate(s).
    w : float or np.ndarray
        Angular coordinate(s).
    deg : bool, optional
        If True, `w` is in degrees. Default is False (radians).

    Returns
    -------
    tuple
        A tuple containing:
        - x : float or np.ndarray
            X coordinate(s).
        - y : float or np.ndarray
            Y coordinate(s).
    """
    # radian if deg=0; degree if deg=1
    if deg:
        w = np.pi * w / 180.0
    return r * np.cos(w), r * np.sin(w)


def polar(x, y, deg=False):
    """
    Converts from rectangular coordinates to polar ones.

    Parameters
    ----------
    x : array_like
        The x coordinates.
    y : array_like
        The y coordinates.
    deg : bool, optional
        If True, returns the angle in degrees. Default is False (radians).

    Returns
    -------
    r : array_like
        The radial coordinates.
    theta : array_like
        The angular coordinates.
    """
    if deg:
        return np.hypot(x, y), 180.0 * np.arctan2(y, x) / np.pi
    else:
        return np.hypot(x, y), np.arctan2(y, x)


def labelledCumSum(X, L):
    """
    Compute the cumulative sum of an array with labels, resetting the
    sum at label changes.

    Parameters
    ----------
    X : np.ndarray
        Input array to compute the cumulative sum.
    L : np.ndarray
        Label array indicating where to reset the cumulative sum.

    Returns
    -------
    np.ma.MaskedArray
        The cumulative sum array with resets at label changes, masked
        appropriately.
    """

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
    """
    Convert Cartesian coordinates to polar coordinates.

    Parameters
    ----------
    x : float or np.ndarray
        X coordinate(s).
    y : float or np.ndarray
        Y coordinate(s).

    Returns
    -------
    r : float or np.ndarray
        Radial coordinate(s).
    th : float or np.ndarray
        Angular coordinate(s) in radians.
    """
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    return r, th


def pol2cart(r, theta):
    """
    Convert polar coordinates to Cartesian coordinates.

    Parameters
    ----------
    r : float or np.ndarray
        Radial coordinate(s).
    theta : float or np.ndarray
        Angular coordinate(s) in radians.

    Returns
    -------
    x : float or np.ndarray
        X coordinate(s).
    y : float or np.ndarray
        Y coordinate(s).
    """
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
    """
    Get the indices of the start of contiguous runs of non-zero values in a
    1D numpy array.

    Parameters
    ----------
    x : np.ndarray
        The input 1D numpy array.

    Returns
    -------
    np.ndarray
        An array of indices marking the start of each contiguous run of
        non-zero values.
    """
    x = np.ravel(x)
    xx = np.ones(len(x) + 1)
    xx[1::] = x
    xx = xx[:-1] != xx[1::]
    xx[0] = True
    return np.nonzero(np.logical_and(x, xx))[0]


def getLabelEnds(x):
    """
    Get the indices of the end of contiguous runs of non-zero values
    in a 1D numpy array.

    Parameters
    ----------
    x : np.ndarray
        The input 1D numpy array.

    Returns
    -------
    np.ndarray
        An array of indices marking the end of each contiguous run of
        non-zero values.
    """
    x = np.ravel(x)
    xx = np.ones(len(x) + 1)
    xx[:-1] = x
    xx = xx[:-1] != xx[1::]
    xx[-1] = True
    return np.nonzero(np.logical_and(x, xx))[0]


def circ_abs(x):
    """
    Calculate the absolute value of an angle in radians,
    normalized to the range [-pi, pi].

    Parameters
    ----------
    x : float or np.ndarray
        Angle(s) in radians.

    Returns
    -------
    float or np.ndarray
        Absolute value of the angle(s) normalized to the range [-pi, pi].
    """
    return np.abs(np.mod(x + np.pi, 2 * np.pi) - np.pi)


def labelContigNonZeroRuns(x):
    """
    Label contiguous non-zero runs in a 1D numpy array.

    Parameters
    ----------
    x : np.ndarray
        The input 1D numpy array.

    Returns
    -------
    np.ndarray
        An array where each element is labeled with an integer representing
        the contiguous non-zero run it belongs to.
    """
    x = np.ravel(x)
    xx = np.ones(len(x) + 1)
    xx[1::] = x
    xx = xx[:-1] != xx[1::]
    xx[0] = True
    L = np.cumsum(np.logical_and(x, xx))
    L[np.logical_not(x)] = 0
    return L


def fixAngle(a):
    """
    Ensure angles lie between -pi and pi.

    Parameters
    ----------
    a : float or np.ndarray
        Angle(s) in radians.

    Returns
    -------
    float or np.ndarray
        Angle(s) normalized to the range [-pi, pi].
    """
    b = np.mod(a + np.pi, 2 * np.pi) - np.pi
    return b


def bwperim(bw, n=4):
    """
    Finds the perimeter of objects in binary images.

    A pixel is part of an object perimeter if its value is one and there
    is at least one zero-valued pixel in its neighborhood.

    By default, the neighborhood of a pixel is 4 nearest pixels, but
    if `n` is set to 8, the 8 nearest pixels will be considered.

    Parameters
    ----------
    bw : array_like
        A black-and-white image.
    n : int, optional
        Connectivity. Must be 4 or 8. Default is 4.

    Returns
    -------
    perim : array_like
        A boolean image.
    """

    if n not in (4, 8):
        raise ValueError("mahotas.bwperim: n must be 4 or 8")
    rows, cols = bw.shape

    # Translate image by one pixel in all directions
    north = np.zeros((rows, cols))
    south = np.zeros((rows, cols))
    west = np.zeros((rows, cols))
    east = np.zeros((rows, cols))

    north[:-1, :] = bw[1:, :]
    south[1:, :] = bw[:-1, :]
    west[:, :-1] = bw[:, 1:]
    east[:, 1:] = bw[:, :-1]
    idx = (north == bw) & (south == bw) & (west == bw) & (east == bw)
    if n == 8:
        north_east = np.zeros((rows, cols))
        north_west = np.zeros((rows, cols))
        south_east = np.zeros((rows, cols))
        south_west = np.zeros((rows, cols))
        north_east[:-1, 1:] = bw[1:, :-1]
        north_west[:-1, :-1] = bw[1:, 1:]
        south_east[1:, 1:] = bw[:-1, :-1]
        south_west[1:, :-1] = bw[:-1, 1:]
        idx &= (
            (north_east == bw)
            & (south_east == bw)
            & (south_west == bw)
            & (north_west == bw)
        )
    return ~idx * bw


def count_runs_and_unique_numbers(arr: np.ndarray) -> tuple:
    """
    Counts the number of continuous runs of numbers in a 1D numpy array.

    Parameters
    ----------
    arr : np.ndarray
        The input 1D numpy array of numbers.

    Returns
    -------
    tuple
        A tuple containing:
        - dict: A dictionary with the count of runs for each unique number.
        - set: The set of unique numbers in the array.

    """
    if arr.size == 0:
        return {}, set()

    unique_numbers = set(arr)
    runs_count = defaultdict(int)
    for num in unique_numbers:
        runs = np.diff(np.where(arr == num)) != 1
        # Add 1 because diff reduces the size by 1
        runs_count[num] = np.count_nonzero(runs) + 1

    return runs_count, unique_numbers


def corr_maps(map1, map2, maptype="normal") -> float:
    """
    Correlates two rate maps together, ignoring areas that have zero sampling.

    Parameters
    ----------
    map1 : np.ndarray
        The first rate map to correlate.
    map2 : np.ndarray
        The second rate map to correlate.
    maptype : str, optional
        The type of correlation to perform. Options are "normal" and "grid".
        Default is "normal".

    Returns
    -------
    float
        The correlation coefficient between the two rate maps.

    Notes
    -----
    If the shapes of the input maps are different, the smaller map will be
    resized to match the shape of the larger map using reflection mode.

    The "normal" maptype considers non-zero and non-NaN values for correlation,
    while the "grid" maptype considers only finite values.
    """
    if map1.shape > map2.shape:
        map2 = skimage.transform.resize(map2, map1.shape, mode="reflect")
    elif map1.shape < map2.shape:
        map1 = skimage.transform.resize(map1, map2.shape, mode="reflect")
    map1 = map1.flatten()
    map2 = map2.flatten()
    valid_map1 = np.zeros_like(map1)
    valid_map2 = np.zeros_like(map2)
    if "normal" in maptype:
        np.logical_or((map1 > 0), ~np.isnan(map1), out=valid_map1)
        np.logical_or((map2 > 0), ~np.isnan(map2), out=valid_map2)
    elif "grid" in maptype:
        np.isfinite(map1, out=valid_map1)
        np.isfinite(map2, out=valid_map2)
    valid = np.logical_and(valid_map1, valid_map2)
    r = np.corrcoef(map1[valid], map2[valid])
    return r[1][0]
