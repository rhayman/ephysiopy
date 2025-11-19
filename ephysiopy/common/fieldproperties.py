import numpy as np
import numpy.ma as ma
from functools import wraps
import weakref
from scipy import signal
from scipy import ndimage as ndi
from scipy import stats
from skimage.measure._regionprops import (
    RegionProperties,
    _infer_number_of_required_args,
    _require_intensity_image,
)
from skimage.measure._regionprops import PROPS as _PROPS
from ephysiopy.common.utils import (
    VariableToBin,
    bwperim,
    circ_abs,
    labelContigNonZeroRuns,
    getLabelStarts,
    getLabelEnds,
    pol2cart,
    cart2pol,
    repeat_ind,
    min_max_norm,
    flatten_list,
    find_runs,
)


"""
An adaptation of code from skimage.measure.regionprops for receptive field
analysis. THe fieldprops function inherits from regionprops but
additionally takes the xy coordinates in its constructor so
that we can extract measures to do with angles and distances etc
of each coordinate to the field peak or field perimeter for example.

As with regionprops, you can also add functions via the 'extra_properties'
argument; with regionprops you specify functions that operate on either one
or two arguments (the intensity image and / or the mask) - here you can do
this and also specify the xy_coordinates as a third argument.

Finally, some of the regionprops functions have been over-ridden so that
the nan versions of them (e.g. nanmax, nanmin etc) are called instead as
ratemaps/ receptive fields often contain nans.
"""
# add some custom properties to the FieldProps class...
PROPS = _PROPS
PROPS["Max_index"] = "max_index"
PROPS["XY_relative_to_peak"] = "xy_relative_to_peak"
PROPS["XY_angle_to_peak"] = "xy_angle_to_peak"
PROPS["xy_dist_to_peak"] = "xy_dist_to_peak"
PROPS["bw_perim"] = "bw_perim"
PROPS["perimeter_angle_from_peak"] = "perimeter_angle_from_peak"
PROPS["perimeter_dist_from_peak"] = "perimeter_dist_from_peak"
PROPS["r_unsmoothed"] = "r_unsmoothed"
PROPS["r_and_phi_to_x_and_y"] = "r_and_phi_to_x_and_y"
PROPS["phase"] = "phase"
PROP_VALS = set(PROPS.values())


def flatten_output(func):
    """
    Decorator to flatten the output of a function that would
    otherwise return a list from a list comprehension and return
    as a np.ndarray
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        return np.array(flatten_list(result))

    return wrapper


def mask_array_with_dynamic_mask(func):
    """
    Decorator to convert a numpy array to a masked array using a dynamically
    generated mask defined by an instance method or variable of the decorated
    function's class.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        # make sure result has at least 2 dimensions
        result = np.atleast_2d(result)
        if not isinstance(result, np.ndarray):
            raise TypeError("The decorated function must return an array.")

        # Generate the mask dynamically
        if hasattr(self, "get_mask") and callable(self.get_mask):
            mask = self.get_mask(result.shape)
        elif hasattr(self, "mask") and isinstance(self.mask, np.ndarray):
            mask = np.broadcast_to(self.mask, result.shape)
        else:
            raise AttributeError(
                "The class must have a 'get_mask' method or 'mask' attribute."
            )

        if mask.shape != result.shape:
            raise ValueError(
                "The mask shape must match the shape of the returned array."
            )

        return ma.MaskedArray(result, mask=mask)

    return wrapper


def spike_times(obj):
    m = getattr(obj, "_mask")
    mask = ma.MaskedArray(m, mask=m)
    clumps = ma.clump_masked(mask)
    fs = obj.sample_rate
    st = obj.slice.start
    ts_2_mask = [((st + c.start) / fs, (st + c.stop) / fs) for c in clumps]
    ts = obj.raw_spike_times
    [
        ma.masked_where(np.logical_and(ts >= t[0], ts <= t[1]), ts, copy=False)
        for t in ts_2_mask
    ]
    return ts


def spike_count(ts, slice, sample_rate, length):
    h, _ = np.histogram(
        ts,
        range=(
            slice.start / sample_rate,
            slice.stop / sample_rate,
        ),
        bins=length,
    )
    return np.atleast_2d(h)


class SpikeTimes(object):
    """
    Descriptor for getting spike times that fall within both
    the LFP segment and the run segment dealing correctly with
    masked time stamps in both cases
    """

    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = "_" + name

    def __get__(self, obj, objtype=None):
        if isinstance(obj, LFPSegment):
            lfp_times = spike_times(obj)
            run_times = spike_times(obj.parent)
            return ma.intersect1d(lfp_times, run_times).compressed()
        elif isinstance(obj, RunProps):
            run_times = spike_times(obj)
            if hasattr(obj, "lfp") and obj.lfp is not None:
                lfp_times = spike_times(obj.lfp)
            else:
                lfp_times = run_times
            return ma.intersect1d(lfp_times, run_times).compressed()

    def __set__(self, obj, value):
        setattr(obj, self.private_name, value)


class SpikingProperty(object):
    """
    Interface for getting attributes by using spike times to
    retrieve their values. Spike times can be masked to
    indicate invalid values (e.g. spikes that occurred
    outside of valid LFP segments, when run speed was too low
    etc)
    """

    spike_times = SpikeTimes()

    def __init__(self, parent, times: np.ndarray, mask: np.ndarray = None):
        self._all_spike_times = times
        self._mask = mask
        self.parent = weakref.ref(parent)()
        self.spike_times = spike_times

    def __len__(self):
        return self.slice.stop - self.slice.start

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, val):
        self._mask = val

    @property
    def n_spikes(self) -> int:
        return ma.sum(self.spike_count)

    @property
    def index(self):
        return np.round(self._all_spike_times * self.sample_rate).astype(int)

    @property
    def raw_spike_times(self):
        """
        Return the spike times that fall within the slice without masking
        """
        return self._all_spike_times[
            np.logical_and(
                self.index >= self.slice.start, self.index <= self.slice.stop
            )
        ]

    @property
    def observed_spikes(self) -> np.ndarray:
        return self.spike_count

    @property
    def spike_count(self):
        return spike_count(
            self.spike_times,
            self.slice,
            self.sample_rate,
            len(self),
        )

    @property
    def spike_index(self) -> np.ndarray:
        """
        Get the index into the LFP data of the spikes for this segment

        Returns
        -------
        np.ndarray
            the index into the LFP data of the spikes for this segment
        """
        return ma.take(
            range(self.slice.start, self.slice.stop),
            repeat_ind(self.observed_spikes.ravel()),
        )


class LFPSegment(SpikingProperty, object):
    """
    A custom class for dealing with segments of an LFP signal and how
    they relate to specific runs through a receptive field
    (see RunProps and FieldProps below)

    Attributes
    ----------
    field_label : int
        The field id
    run_label : int
        The run id
    slice : slice
        slice into the LFP data for a segment
    spike_times : np.ndarray
        the times in seconds spikes occurred for a segment
    spike_count : np.ndarray
        spikes binned into lfp samples for a segment
    signal : np.ndarray
        raw signal for a segment
    filtered_signal : np.ndarray
        bandpass filtered signal for a segment
    phase : np.ndarray
        phase data for a segment
    amplitude : np.ndarray
        amplitude for a segment
    sample_rate : float, int
        sample rate for the LFP segment
    filter_band : tuple[int,int]
        the bandpass filter values

    """

    def __init__(
        self,
        parent,
        field_label: int,
        run_label: int,
        slice: slice,
        spike_times: np.ndarray,
        mask: np.ndarray,
        signal: np.ndarray,
        filtered_signal: np.ndarray,
        phase: np.ndarray,
        cycle_label: np.ndarray,
        sample_rate: float | int,
    ):
        """
        Parameters
        ----------
        field_label : int
            the field id to which this LFP segment belongs
        run_label : int
            the run id through the field to which this LFP segment belongs
        slice : slice
            the slice corresponding the pos sample I think
        spike_count : np.ndarray
            spikes binned into lfp samples
        signal, filtered_signal, phase, : np.ndarray
            the raw, filtered, and phase of the LFP for a given segment
        sample_rate : int, float
            the sample rate for the LFP signal
        filter_band : tuple
            the bandpass filter giving filtered_signal

        """
        super().__init__(parent, spike_times, mask)
        self.field_label = field_label
        self.run_label = run_label  # integer that should match with a run
        self.slice = slice  # an index into the main LFP data
        self._signal = signal
        self._filtered_signal = filtered_signal
        self._phase = phase
        self._cycle_label = cycle_label
        self.sample_rate = sample_rate

    @property
    @mask_array_with_dynamic_mask
    def signal(self):
        return self._signal

    @property
    @mask_array_with_dynamic_mask
    def filtered_signal(self):
        return self._filtered_signal

    @property
    @mask_array_with_dynamic_mask
    def phase(self):
        return self._phase

    @property
    @mask_array_with_dynamic_mask
    def cycle_label(self):
        return self._cycle_label

    def spiking_var(self, var="phase"):
        """
        Get the value of a variable at the position of
        spikes for this run
        """
        if hasattr(self, var):
            v = getattr(self, var)
            return ma.take(v, self.spike_index - self.slice.start, axis=-1)
        else:
            raise AttributeError(f"{var} is not an attribute of LFPSegment")

    def mean_spiking_var(self, var="phase"):
        """
        Get the mean value of a variable at the posiition of
        the spikes for all runs through this field when multiple spikes
        occur in a single theta cycle
        """
        labels = ma.compressed(self.spiking_var("cycle_label"))
        if len(labels) == 1:
            # return the value directly
            return self.spiking_var(var)
        elif len(labels) > 1:
            _, _, lens = find_runs(labels)
            idx = repeat_ind(lens)
            spk_var = ma.compressed(self.spiking_var(var))
            if var == "phase":
                out = np.zeros_like(np.unique(labels), dtype=complex)
                np.add.at(out.ravel(), idx, np.exp(1j * spk_var).ravel())
                out = np.angle(out)
            else:
                out = np.zeros_like(np.unique(labels), dtype=float)
                np.add.at(out.ravel(), idx, spk_var.ravel())
                out /= lens
            return np.atleast_2d(out)


class RunProps(SpikingProperty, object):
    """
    A custom class for holding information about runs through a receptive field

    Each run needs to have some information about the field to which it belongs
    so the constructor takes in the peak x-y coordinate of the field and its
    index as well as the coordinates of the perimeter of the field

    Attributes
    ----------
    label : int
        the run id
    slice : slice
        the slice of the position data for a run
    xy: np.ndarray
        the x-y coordinates for a run (global coordinates)
    speed : np.ndarray
        the speed at each xy coordinate
    peak_xy : tuple[float, float]
        the fields max rate xy location
    max_index : int
        the index into the arrays of the field max
    perimeter_coords : np.ndarray
        xy coordinates of the field perimeter
    hdir : np.ndarray
        the heading direction
    min_speed : float
        the minimum speed
    cumulative_time : np.ndarray
        the cumulative time spent on a run
    duration: float
        the total duration of a run in seconds
    n_spikes : int
        the total number of spikes emitted on a run
    run_start : int
        the position index of the run start
    run_stop : int
        the position index of the run stop
    mean_direction : float
        the mean direction of a run
    current_direction : np.ndarray
        the current direction of a run
    cumulative_distance : np.ndarray
        the cumulative distance covered in a run
    spike_index : np.ndarray
        the index into the position data of the spikes on a run
    observed_spikes : np.ndarray
        the observed spikes on a run (binned by position samples)
    xy_angle_to_peak : np.ndarray
        the xy angle to the peak (radians)
    xy_dist_to_peak : np.ndarray
        the distance to the field max
    xy_dist_to_peak_normed : np.ndarray
        normalised distance to field max
    pos_xy : np.ndarray
        cartesian xy coordinates but normalised on a unit circle
    pos_phi : np.ndarray
        the angular distance between a runs main direction and the
        direction to the peak for each position sample
    rho : np.ndarray
        the polar radial distance (1 = field edge)
    phi : np.ndarray
        the polar angle (radians)
    r_and_phi_to_x_and_y : np.ndarray
        converts rho and phi to x and y coordinates (range = -1 -> +1)
    tortuosity : np.ndarray
        the tortuosity for a run (closer to 1 = a straighter run)
    xy_is_smoothed : bool
        whether the xy data has been smoothed
    """

    def __init__(
        self,
        parent,
        label: int,
        slice: slice,
        spike_times: np.ndarray,
        mask: np.ndarray,
        xy_coords: np.ndarray,
        speed: np.ndarray,
        peak_xy: np.ndarray,
        max_index: np.ndarray,
        perimeter_coords: np.ndarray,
        sample_rate: float = 50,
    ):
        """
        Parameters
        ----------
        label : int
            the field label the run belongs to
        slice : slice
            the slice into the position array that holds all the position data
        spike_times: np.ndarray
            the spike times for this run
        xy_coords : np.ndarray
            the xy data for this run (global coordinates)
        hdir : np.ndarray
            the heading direction for this run
        speed : np.ndarray
            the speed for this run
        peak_xy : np.ndarray
            the peak location of the field for the this run
        max_index : np.ndarray
            the index (r,c) of the maximum of the firing field
        perimeter_coords : np.ndarray
            the xy coordinates of the perimeter of the field
        """
        super().__init__(parent, spike_times, mask)
        self.label = label
        self._xy_coords = xy_coords
        self._slice = slice
        self._speed = speed
        self._peak_xy = peak_xy
        self._max_index = max_index
        self._perimeter_coords = perimeter_coords
        self.sample_rate = sample_rate
        self.xy_is_smoothed = False
        self._hdir = None

    def __str__(self):
        return f"id: {self.label}: {self.n_spikes} spikes"

    @property
    def ndim(self):
        """
        Return the dimensionality of the data

        For 1 x n linear track data dimensionality = 1
        for 2 x n open field (or other) data dimensionality = 2
        """
        return np.shape(self.xy)[0]

    @property
    @mask_array_with_dynamic_mask
    def xy(self) -> ma.MaskedArray:
        return self._xy_coords

    @xy.setter
    def xy(self, val):
        self._xy_coords = val

    @property
    def slice(self) -> slice:
        return self._slice

    @slice.setter
    def slice(self, val):
        if not isinstance(val, slice):
            raise TypeError("slice must be a slice object")
        self._slice = val

    @property
    @mask_array_with_dynamic_mask
    def hdir(self) -> ma.MaskedArray:
        if self._hdir is None:
            if self.ndim == 1:
                tmp = np.arctan2(
                    np.diff(self.xy[0]), np.diff(np.zeros_like(self.xy[0]))
                )
            elif self.ndim == 2:
                tmp = np.arctan2(np.diff(self.xy[1]), np.diff(self.xy[0]))
            self._hdir = np.insert(tmp, -1, tmp[-1])
        return self._hdir

    @hdir.setter
    def hdir(self, val):
        self._hdir = val

    @property
    @mask_array_with_dynamic_mask
    def speed(self) -> ma.MaskedArray:
        return self._speed

    @property
    def min_speed(self) -> float:
        return np.nanmin(self._speed)

    @property
    def time(self):
        return np.linspace(
            self.slice.start / self.sample_rate,
            self.slice.stop / self.sample_rate,
            len(self),
        )

    @property
    @mask_array_with_dynamic_mask
    def cumulative_time(self) -> ma.MaskedArray:
        return np.arange(len(self))

    @property
    def duration(self) -> float:
        return (self.slice.stop - self.slice.start) / self.sample_rate

    @property
    def run_start(self) -> int:
        return self.slice.start

    @property
    def run_stop(self) -> int:
        return self.slice.stop

    @property
    def mean_direction(self) -> float:
        return stats.circmean(self.hdir)

    # TODO: might be fucked for 1D
    @property
    @mask_array_with_dynamic_mask
    def current_direction(self) -> ma.MaskedArray:
        """
        Supposed to calculate current direction wrt to field centre?
        """
        return self.rho * np.cos(self.hdir - self.phi)

    @property
    @mask_array_with_dynamic_mask
    def cumulative_distance(self) -> ma.MaskedArray:
        if self.ndim == 1:
            d = np.abs(np.diff(self.xy))
        elif self.ndim == 2:
            d = np.sqrt(np.abs(np.diff(np.power(self.rho, 2))))
        d = np.insert(d, 0, 0)
        return np.cumsum(d)

    @property
    def total_distance(self) -> float:
        return self.cumulative_distance[-1]

    @property
    def spike_num_in_run(self):
        return np.arange(self.n_spikes)

    def expected_spikes(
        self, expected_rate_at_pos: np.ndarray, sample_rate: int = 50
    ) -> np.ndarray:
        """
        Calculates the expected number of spikes along this run given the
        whole ratemap.

        Parameters
        ----------
        expected_rate_at_pos : np.ndarray
            the rate seen at each xy position of the whole trial

        Returns
        -------
        expected_rate : np.ndarray
            the expected rate at each xy position of this run
        """
        return expected_rate_at_pos[..., self.slice] / sample_rate

    def overdispersion(self, spike_train: np.ndarray, fs: int = 50) -> float:
        """
        The overdispersion map for this run

        Parameters
        ----------
        spike_train : np.mdarray
            the spike train (spikes binned up by position) for the whole trial.
            Same length as the trial n_samples
        fs : int
            the sample rate of the position data
        """
        obs_spikes = self.n_spikes
        expt_spikes = np.nansum(self.expected_spikes(spike_train, fs))
        Z = np.nan
        if obs_spikes >= expt_spikes:
            Z = (obs_spikes - expt_spikes - 0.5) / np.sqrt(expt_spikes)
        else:
            Z = (obs_spikes - expt_spikes + 0.5) / np.sqrt(expt_spikes)
        return Z

    def smooth_xy(self, k: float, spatial_lp: int, sample_rate: int) -> None:
        """
        Smooth in x and y in preparation for converting the smoothed cartesian
        coordinates to polar ones

        Parameters
        ----------
        k : float
            smoothing constant for the instantaneous firing rate
        spatial_lp_cut : int
            spatial lowpass cut off
        sample_rate : int
            position sample rate in Hz
        """
        f_len = np.floor((self.run_stop - self.run_start) * k) + 1
        filt = signal.firwin(
            int(f_len),
            fs=sample_rate,
            cutoff=spatial_lp / sample_rate * 2,
            window="blackman",
        )
        padlen = 2 * len(filt)
        if padlen == self.xy.shape[1]:
            padlen = padlen - 1
        self.xy = signal.filtfilt(filt, [1], self.xy, padlen=padlen, axis=1)
        self.xy_is_smoothed = True

    @property
    @mask_array_with_dynamic_mask
    def xy_angle_to_peak(self) -> ma.MaskedArray:
        xy_to_peak = self.xy - self._peak_xy[:, None]
        if self.ndim == 1:
            xy_to_peak = np.vstack([xy_to_peak, np.zeros_like(xy_to_peak)])
        return np.arctan2(xy_to_peak[1], xy_to_peak[0])

    @property
    @mask_array_with_dynamic_mask
    def xy_dist_to_peak(self) -> ma.MaskedArray:
        xy_to_peak = self.xy - self._peak_xy[:, None]
        if self.ndim == 1:
            return np.hypot(xy_to_peak, 0)
        elif self.ndim == 2:
            return np.hypot(xy_to_peak[0], xy_to_peak[1])

    @property
    @mask_array_with_dynamic_mask
    def xy_dist_to_peak_normed(self) -> ma.MaskedArray:
        """
        Values lie between 0 and 1
        """
        x_y = self.r_and_phi_to_x_and_y
        return np.hypot(x_y[0], x_y[1])

    def perimeter_minus_field_max(self) -> tuple[np.ndarray]:
        mi = self._peak_xy
        perimeter_coords = self._perimeter_coords
        if self.ndim == 1:
            return perimeter_coords - mi
        elif self.ndim == 2:
            return [perimeter_coords[0] - mi[0], perimeter_coords[1] - mi[1]]

    def perimeter_angle_from_peak(self) -> np.ndarray:
        perimeter_minus_field_max = self.perimeter_minus_field_max()
        if self.ndim == 1:
            return np.arctan2(perimeter_minus_field_max[0], 0)
        elif self.ndim == 2:
            return np.arctan2(
                perimeter_minus_field_max[1], perimeter_minus_field_max[0]
            )

    @property
    @mask_array_with_dynamic_mask
    def pos_xy(self) -> ma.MaskedArray:
        pos_x, pos_y = pol2cart(self.pos_r, self.pos_phi)
        return np.vstack([pos_x, pos_y])

    @property
    @mask_array_with_dynamic_mask
    def pos_r(self) -> ma.MaskedArray:
        """
        Values lie between 0 and 1
        """
        angle_df = circ_abs(
            self.perimeter_angle_from_peak()[:, None] - self.xy_angle_to_peak
        )
        perimeter_idx = np.argmin(angle_df, 0)
        peak_xy = self._peak_xy
        if self.ndim == 1:
            tmp = self._perimeter_coords[0][perimeter_idx] - peak_xy
            perimeter_dist_to_peak = np.hypot(0, tmp)
        elif self.ndim == 2:
            tmp = (
                self._perimeter_coords[0][perimeter_idx] - peak_xy[0],
                self._perimeter_coords[1][perimeter_idx] - peak_xy[1],
            )

            perimeter_dist_to_peak = np.hypot(tmp[0], tmp[1])
        r = self.xy_dist_to_peak / perimeter_dist_to_peak
        capped_vals = r >= 1
        r[..., capped_vals] = 1
        return r

    # calculate the angle between the runs main direction and the
    # pos's direction to the peak centre
    @property
    @mask_array_with_dynamic_mask
    def pos_phi(self) -> ma.MaskedArray:
        """
        Values lie between 0 and 2pi
        """
        return self.xy_angle_to_peak - self.mean_direction

    @property
    @mask_array_with_dynamic_mask
    def rho(self) -> ma.MaskedArray:
        """
        Values lie between 0 and 1
        """
        rho, _ = cart2pol(self.pos_xy[0], self.pos_xy[1])
        rho[rho > 1] = 1
        return rho

    @property
    @mask_array_with_dynamic_mask
    def phi(self) -> ma.MaskedArray:
        """
        Values lie between 0 and 2pi
        """
        _, phi = cart2pol(self.pos_xy[0], self.pos_xy[1])
        return phi

    @property
    @mask_array_with_dynamic_mask
    def r_and_phi_to_x_and_y(self) -> ma.MaskedArray:
        return np.vstack(pol2cart(self.rho, self.phi))

    @property
    def normed_x(self) -> np.ndarray:
        """
        Normalise the x data to lie between -1 and 1
        with respect to the field limits
        of the parent field
        """
        be = self.parent.binned_data.bin_edges[0]
        sl = self.parent.slice[0]
        xmin = be[sl.start]
        xmax = be[sl.stop]
        fp = np.linspace(-1, 1, 1000)
        xp = np.linspace(xmin, xmax, 1000)
        return np.interp(self.xy[0], xp, fp)

    """
    Define a measure of tortuosity to see how direct the run was
    from field entry to exit. It's jsut the ratio of the distance between
    a straight line joining the entry-exit points and the actual distance
    of the run
    """

    # TODO: Won't work with 1D data
    @property
    @mask_array_with_dynamic_mask
    def tortuosity(self) -> ma.MaskedArray:
        direct_line_distance = np.hypot(
            self.xy[0, 0] - self.xy[0, -1], self.xy[1, 0] - self.xy[1, -1]
        )
        xy_df = np.diff(self.xy)
        traversed_distance = np.sum(np.hypot(xy_df[0], xy_df[1]))
        return direct_line_distance / traversed_distance

    def spiking_var(self, var="current_direction"):
        """
        Get the value of a variable at the position of
        spikes for this run
        """
        if hasattr(self, var):
            v = getattr(self, var)
            return ma.take(v, self.spike_index - self.slice.start, axis=-1)
        else:
            raise AttributeError(f"{var} is not an attribute of RunProps")

    def mean_spiking_var(self, var="current_direction"):
        """
        Get the mean value of a variable at the posiition of
        the spikes for all runs through this field when multiple spikes
        occur in a single theta cycle
        """
        if not hasattr(self, "lfp") or self.lfp is None:
            raise AttributeError("RunProps must have an lfp attribute")

        if hasattr(self, var):
            labels = ma.compressed(self.lfp.spiking_var("cycle_label"))

            if len(labels) == 1:
                # return the value directly
                return np.atleast_2d(self.spiking_var(var))

            elif len(labels) > 1:
                _, _, lens = find_runs(labels)
                idx = repeat_ind(lens)
                spike_var = ma.compress_cols(np.atleast_2d(self.spiking_var(var)))
                ndim = np.shape(spike_var)[0]
                M = np.zeros(shape=[ndim, len(np.unique(labels))], dtype=float)
                [np.add.at(M[i, :], idx, spike_var[i, :]) for i in range(ndim)]
                M /= lens
                return np.atleast_2d(M)

        else:
            raise AttributeError(f"{var} is not an attribute of RunProps")


class FieldProps(RegionProperties):
    """
    Describes various properties of a receptive field.

    Attributes
    ----------
    slice : tuple of slice
        The slice of the field in the binned data (x slice, y slice)
    label : int
        The label of the field
    image_intensity : np.ndarray
        The intensity image of the field (in Hz)
    runs : list of RunProps
        The runs through the field
    run_slices : list of slice
        The slices of the runs through the field (slices are position indices)
    run_labels : np.ndarray
        The labels of the runs
    max_index : np.ndarray
        The index of the maximum intensity in the field
    num_runs : int
        The number of runs through the field
    cumulative_time : list of np.ndarray
        The cumulative time spent on the field for each run through the field
    cumulative_distance: list of np.ndarray
        The cumulative time spent on the field for each run through the field
    runs_speed : list of np.ndarray
        The speed of each run through the field
    runs_observed_spikes : np.ndarray
        The observed spikes for each run through the field
    spike_index : np.ndarray
        The index of the spikes in the position data
    xy_at_peak : np.ndarray
        The x-y coordinate of the field max
    xy : np.ndarray
        The x-y coordinates of the field for all runs
    xy_relative_to_peak : np.ndarray
        The x-y coordinates of the field zeroed with respect to the peak
    xy_angle_to_peak : np.ndarray
        The angle each x-y coordinate makes to the field peak
    xy_dist_to_peak : np.ndarray
        The distance of each x-y coordinate to the field peak
    bw_perim : np.ndarray
        The perimeter of the field as an array of bool
    perimeter_coords : tuple
        The x-y coordinates of the field perimeter
    global_perimeter_coords : np.ndarray
        The global x-y coordinates of the field perimeter
    perimeter_minus_field_max : np.ndarray
        The x-y coordinates of the field perimeter minus the field max
    perimeter_angle_from_peak : np.ndarray
        The angle each point on the perimeter makes to the field peak
    perimeter_dist_from_peak : np.ndarray
        The distance of each point on the perimeter to the field peak
    bin_coords : np.ndarray
        The x-y coordinates of the field in the binned data
    phi : np.ndarray
        The angular distance between the mean direction of each run and
        each position samples direction to the field centre
    rho : np.ndarray
        The distance of each position sample to the field max (1 is furthest)
    pos_xy : np.ndarray
        The cartesian x-y coordinates of each position sample
    pos_phi : np.ndarray
        The angular distance between the mean direction of each run and
        each position samples direction to the field centre
    pos_r : np.ndarray
        The ratio of the distance from the field peak to the position sample
        and the distance from the field peak to the point on the perimeter
        that is most colinear with the position sample
    r_and_phi_to_x_and_y : np.ndarray
        Converts rho and phi to x and y coordinates
    r_per_run : np.ndarray
        The polar radial distance for each run
    current_direction : np.ndarray
        The direction projected onto the mean run direction
    cumulative_distance : list of np.ndarray
        The cumulative distance for each run
    projected_direction : np.ndarray
        The direction projected onto the mean run direction
    intensity_max : float
        The maximum intensity of the field (i.e. field peak rate)
    intensity_mean : float
        The mean intensity of the field
    intensity_min : float
        The minimum intensity of the field
    intensity_std : float
        The standard deviation of the field intensity


    """

    def __init__(
        self,
        slice,
        label,
        label_image,
        binned_data,
        cache,
        *,
        extra_properties,
        spacing,
        offset,
        index=0,
    ):
        intensity_image = binned_data.binned_data[index]
        super().__init__(
            slice,
            label,
            label_image,
            intensity_image,
            cache_active=cache,
            spacing=spacing,
            extra_properties=extra_properties,
            offset=offset,
        )
        self.binned_data = binned_data
        self._runs = []

    def __str__(self):
        return f"field: {self.label}: {self.num_runs} runs"

    @property
    def runs(self):
        return self._runs

    @property
    def run_slices(self):
        return [r._slice for r in self._runs]

    @runs.setter
    def runs(self, r):
        self._runs = r
        print(f"Field {self.label} has {len(r)} potential runs")

    @property
    def run_labels(self):
        return np.array([r.label for r in self.runs])

    @property
    @flatten_output
    def spike_run_labels(self):
        return [np.repeat(r.label, np.sum(r.spike_count)) for r in self.runs]

    # The maximum index of the intensity image for the region
    # returns as (row, col) so (y,x)
    @property
    def max_index(self) -> np.ndarray:
        return np.array(
            np.unravel_index(
                np.nanargmax(self.image_intensity, axis=None),
                self.image_intensity.shape,
            )
        )

    @property
    def num_runs(self) -> int:
        return len(self.runs)

    @property
    def n_spikes(self) -> int:
        """
        The total number of spikes emitted on all runs through the field
        """
        return np.sum([r.n_spikes for r in self.runs])

    @property
    @flatten_output
    def cumulative_time(self) -> list:
        return [r.cumulative_time.ravel() for r in self.runs]

    @property
    @flatten_output
    def speed(self) -> list:
        return [r._speed.ravel() for r in self.runs]

    @property
    @flatten_output
    def observed_spikes(self) -> np.ndarray:
        return [r.observed_spikes.ravel() for r in self.runs]

    @property
    def normalized_position(self) -> list:
        """
        Only makes sense to run this on linear track data unless
        we want to pass the unit circle distance or something...

        Get the normalized position for each run through the field.

        Normalized position is the position of the run relative to the
        start of the field (0) to the end of the field (1).
        """
        if (
            self.binned_data.variable.value == VariableToBin.X.value
            or self.binned_data.variable.value == VariableToBin.Y.value
            or self.binned_data.variable.value == VariableToBin.PHI.value
        ):

            w = np.diff(self.binned_data.bin_edges[0])[0]
            pos_min = self.binned_data.bin_edges[0][self.slice[0].start]
            pos_max = self.binned_data.bin_edges[0][self.slice[0].stop - 1] + w
            return [min_max_norm(r.xy[0], pos_min, pos_max) for r in self.runs]

    @property
    @flatten_output
    def phase(self) -> list:
        """
        The phases of the LFP signal for all runs through this field
        """
        phases = [r.lfp.phase for r in self.runs if r.lfp]
        if len(phases) == 0:
            return None
        return phases

    @property
    def compressed_phase(self) -> np.ndarray:
        """
        The phases of the LFP signal for all runs through this field
        compressed into a single array
        """
        phases = [r.lfp.phase.compressed() for r in self.runs if r.lfp]
        if len(phases) == 0:
            return None
        return phases

    @property
    def spike_index(self):
        return np.concatenate([r.spike_index for r in self.runs])

    @property
    def spike_phase(self):
        phases = [r.lfp.spiking_var().T for r in self.runs if r.lfp is not None]
        if len(phases) == 0:
            return None
        return np.concatenate(phases).T

    @property
    def mean_spike_phase(self):
        return np.concatenate([r.lfp.mean_spiking_var().ravel() for r in self.runs])

    # The x-y coordinate at the field peak
    @property
    def xy_at_peak(self) -> np.ndarray:
        mi = self.max_index
        if len(self.slice) == 1:
            # If the field is 1D then we only have one coordinate
            x_max = self.binned_data.bin_edges[0][mi[0] + self.slice[0].start]
            return np.array([x_max])
        # If the field is 2D then we have two coordinates
        if len(self.slice) == 2:
            x_max = self.binned_data.bin_edges[1][mi[1] + self.slice[1].start]
            y_max = self.binned_data.bin_edges[0][mi[0] + self.slice[0].start]
            return np.array([x_max, y_max])

    @property
    def xy(self) -> np.ndarray:
        return np.concatenate([r.xy.T for r in self.runs]).T

    # The x-y coordinates zeroed with respect to the peak
    @property
    def xy_relative_to_peak(self) -> np.ndarray:
        return (self.xy.T - self.xy_at_peak).T

    # The angle each x-y coordinate makes to the field peak
    @property
    def xy_angle_to_peak(self) -> np.ndarray:
        xy_to_peak = self.xy_relative_to_peak
        return np.arctan2(xy_to_peak[1], xy_to_peak[0])

    # The distance of each x-y coordinate to the field peak
    @property
    def xy_dist_to_peak(self) -> np.ndarray:
        xy_to_peak = self.xy_relative_to_peak
        return np.hypot(xy_to_peak[0], xy_to_peak[1])

    # The perimeter of the masked region as an array of bool
    # TODO: this might be fucked in the 1D case - check
    @property
    def bw_perim(self) -> np.ndarray:
        if self.image.ndim == 1:
            return self.image
        return bwperim(self.image)

    # TODO: in 2D case returns as y,x
    @property
    def perimeter_coords(self) -> tuple:
        return np.nonzero(self.bw_perim)

    @property
    def global_perimeter_coords(self) -> np.ndarray:
        perim_xy = self.perimeter_coords
        binned_data = self.binned_data
        if len(binned_data.bin_edges) == 1:
            # If the field is 1D then we only have one coordinate
            x = binned_data.bin_edges[0][perim_xy[0] + self.slice[0].start]
            return np.array([x])
        else:
            x = binned_data.bin_edges[1][perim_xy[1] + self.slice[1].start]
            y = binned_data.bin_edges[0][perim_xy[0] + self.slice[0].start]
            return np.array([x, y])

    @property
    def perimeter_minus_field_max(self) -> np.ndarray:
        mi = self.max_index
        perim_coords = self.perimeter_coords
        if len(self.slice) == 1:
            # If the field is 1D then we only have one coordinate
            return np.array([perim_coords[0] - mi[0]])
        else:
            return np.array([perim_coords[0] - mi[0], perim_coords[1] - mi[1]])

    # The angle each point on the perimeter makes to the field peak
    @property
    def perimeter_angle_from_peak(self) -> np.ndarray:
        perim_minus_field_max = self.perimeter_minus_field_max
        return np.arctan2(perim_minus_field_max[1], perim_minus_field_max[0])

    # The distance of each point on the perimeter to the field peak
    @property
    def perimeter_dist_from_peak(self) -> np.ndarray:
        perim_minus_field_max = self.perimeter_minus_field_max()
        return np.hypot(perim_minus_field_max[0], perim_minus_field_max[1])

    @property
    def bin_coords(self) -> np.ndarray:
        be = self.binned_data.bin_edges
        coords = self.coords
        if len(be) == 2:
            return np.array([be[1][coords[:, 1]], be[0][coords[:, 0]]])
        elif len(be) == 1:
            return np.array([be[0][coords[:, 0]]])

    @property
    def phi(self) -> np.ndarray:
        """
        Calculate the angular distance between the mean direction of each run
        and each position samples direction to the field centre
        """
        return ma.concatenate([r.phi.T for r in self.runs]).T

    @property
    def rho(self) -> np.ndarray:
        return ma.concatenate([r.rho.T for r in self.runs]).T

    @property
    def pos_xy(self) -> np.ndarray:
        return ma.concatenate([r.pos_xy.T for r in self.runs]).T

    @property
    def pos_phi(self) -> np.ndarray:
        """
        Calculate the angular distance between the mean direction of each run
        and each position samples direction to the field centre
        """
        return ma.concatenate([r.pos_phi.T for r in self.runs]).T

    @property
    def pos_r(self) -> np.ndarray:
        """
        Calculate the ratio of the distance from the field peak to the position
        sample and the distance from the field peak to the point on the
        perimeter that is most colinear with the position sample

        NB The values just before being returned can be >= 1 so these are
        capped to 1
        """
        return ma.concatenate([r.pos_r.T for r in self.runs]).T

    @property
    def r_and_phi_to_x_and_y(self) -> np.ndarray:
        return np.vstack(pol2cart(self.pos_r, self.pos_phi))

    @property
    def r_per_run(self) -> np.ndarray:
        perimeter_coords = self.perimeter_coords
        return np.concatenate(
            [
                run.r(
                    self.xy_at_peak,
                    self.perimeter_angle_from_peak,
                    perimeter_coords,
                    self.max_index,
                )
                for run in self.runs
            ]
        )

    @property
    @flatten_output
    def current_direction(self) -> list:
        return [r.current_direction.T for r in self.runs]

    @property
    @flatten_output
    def cumulative_distance(self) -> list:
        return [r.cumulative_distance.T for r in self.runs]

    @property
    def projected_direction(self) -> np.ndarray:
        """
        direction projected onto the mean run direction is just the x-coord
        when cartesian x and y is converted to from polar rho and phi
        """
        return ma.concatenate([r.pos_xy[0] for r in self.runs])

    def spiking_var(self, var="current_direction"):
        """
        Get the value of a variable at the position of
        spikes for all runs through this field

        Parameters
        ----------
        var : str
            the variable to get the value of at the position of spikes

        Returns
        -------
        np.ndarray
            the value of the variable at the position of spikes
            for all runs through this field
        """
        return np.concatenate(
            [r.spiking_var(var).T for r in self.runs if hasattr(r, var)]
        ).T

    def mean_spiking_var(self, var="current_direction"):
        """
        Get the mean value of a variable at the posiition of
        the spikes for all runs through this field when multiple spikes
        occur in a single theta cycle

        Parameters
        ----------
        var : str
            the variable to get the mean value of at the position of spikes

        Returns
        -------
        np.ndarray
            the mean value of the variable at the position of spikes
            for all runs through this field
        """
        if hasattr(self.runs[0], var):
            return np.concatenate([r.mean_spiking_var(var).T for r in self.runs]).T

        elif hasattr(self.runs[0], "lfp"):
            if hasattr(self.runs[0].lfp, var):
                return np.concatenate(
                    [r.lfp.mean_spiking_var(var).T for r in self.runs if r.n_spikes > 0]
                ).T

    # Over-ride the next intensity_* functions so they use the
    # nan versions
    @property
    def intensity_max(self) -> float:
        vals = self.image_intensity[self.image]
        return np.nanmax(vals, axis=0).astype(np.float64, copy=False)

    @property
    def intensity_mean(self) -> float:
        return np.nanmean(self.image_intensity[self.image], axis=0)

    @property
    def intensity_min(self) -> float:
        vals = self.image_intensity[self.image]
        return np.nanmin(vals, axis=0).astype(np.float64, copy=False)

    @property
    def intensity_std(self) -> float:
        vals = self.image_intensity[self.image]
        return np.nanstd(vals, axis=0)

    def __getattr__(self, attr):
        if self._intensity_image is None and attr in _require_intensity_image:
            raise AttributeError(
                f"Attribute '{attr}' unavailable when `intensity_image` "
                f"has not been specified."
            )
        if attr in self._extra_properties:
            func = self._extra_properties[attr]
            n_args = _infer_number_of_required_args(func)
            # determine whether func requires intensity image
            if n_args == 2:
                if self._intensity_image is not None:
                    if self._multichannel:
                        multichannel_list = [
                            func(self.image, self.image_intensity[..., i])
                            for i in range(self.image_intensity.shape[-1])
                        ]
                        return np.stack(multichannel_list, axis=-1)
                    else:
                        return func(self.image, self.image_intensity)
                else:
                    raise AttributeError(
                        f"intensity image required to calculate {attr}"
                    )
            elif n_args == 3:
                if self._intensity_image is not None:
                    return func(
                        self.image,
                        self.image_intensity,
                        self.xy_coords,
                    )
                else:
                    raise AttributeError(
                        f"intensity image required to calculate {attr}"
                    )
            elif n_args == 1:
                return func(self.image)
            else:
                raise AttributeError(
                    f"Custom regionprop function's number of arguments must "
                    f"be 1, 2 or 3 but {attr} takes {n_args} arguments."
                )
        elif attr in PROPS and attr.lower() == attr:
            if (
                self._intensity_image is None
                and PROPS[attr] in _require_intensity_image
            ):
                raise AttributeError(
                    f"Attribute '{attr}' unavailable when `intensity_image` "
                    f"has not been specified."
                )
            # retrieve deprecated property (excluding old CamelCase ones)
            return getattr(self, PROPS[attr])
        else:
            raise AttributeError(f"'{type(self)}' has no attribute '{attr}'")

    def __str__(self):
        """
        Override the string representation printed to the console
        """
        return (
            f"Field {self.label} has {len(self.runs)} runs with {self.n_spikes} spikes"
        )

    def smooth_runs(self, k: float, spatial_lp_cut: int, sample_rate: int):
        """
        Smooth in x and y in preparation for converting the smoothed cartesian
        coordinates to polar ones

        Parameters
        ----------
        k : float
            smoothing constant for the instantaneous firing rate
        spatial_lp_cut : int
            spatial lowpass cut off
        sample_rate : int
            position sample rate in Hz
        """
        [r.smooth_xy(k, spatial_lp_cut, sample_rate) for r in self.runs]

    def runs_expected_spikes(
        self, expected_rate: np.ndarray, sample_rate: int = 50
    ) -> np.ndarray:
        """
        Calculate the expected number of spikes along each run given the
        whole ratemap.

        Parameters
        ----------
        expected_rate: np.ndarray
            the rate seen at each xy position of the whole trial
        sample_rate : int
            the sample rate of the position data

        Returns
        -------
        np.ndarray
            the expected rate at each xy position for each run

        Notes
        -----
        The expected spikes should be calculated from the smoothed
        ratemap and the xy position data using np.digitize:

        >> xbins = np.digitize(xy[0], binned_data.bin_edges[1][:-1]) - 1
        >> ybins = np.digitize(xy[1], binned_data.bin_edges[0][:-1]) - 1
        >> expected_rate_at_pos = binned_data.binned_data[0][ybins, xbins]
        >> exptd_spks = fieldprops.runs_expected_spikes(expected_rate_at_pos)
        """
        return np.concatenate(
            [r.expected_spikes(expected_rate, sample_rate) for r in self.runs]
        )

    def overdispersion(self, spikes: np.ndarray, fs: int = 50) -> np.ndarray:
        """
        Calculate the overdispersion for each run through the field

        Parameters
        ----------
        spike_train : np.ndarray
            the spike train (spikes binned up by position) for the whole trial.
            Same length as the trial n_samples
        fs : int
            the sample rate of the position data

        Returns
        -------
        np.ndarray
            the overdispersion for each run through the field
        """
        return np.array([r.overdispersion(spikes, fs) for r in self.runs])


# TODO: Document valid kwargs items
def fieldprops(
    label_image,
    binned_data,
    spike_times,
    xy,
    method="field",
    cache=True,
    *,
    extra_properties=None,
    spacing=None,
    offset=None,
    **kwargs,
):
    r"""Measure properties of labeled image regions.

    Parameters
    ----------
    label_image : (M, N[, P]) ndarray
        Labeled input image. Labels with value 0 are ignored.

        .. versionchanged:: 0.14.1
            Previously, ``label_image`` was processed by ``numpy.squeeze`` and
            so any number of singleton dimensions was allowed. This resulted in
            inconsistent handling of images with singleton dimensions. To
            recover the old behaviour, use
            ``regionprops(np.squeeze(label_image), ...)``.
    xy : (2 x n_samples) np.ndarray
        The x-y coordinates for all runs through the field corresponding to
        a particular label
    binned_data : BinnedData instance from ephysiopy.common.utils
    spike_times : np.ndarray
        The spike times for the neuron being analysed
    method: {'field', 'clump_runs'}, optional
        Method used to calculate region properties:

        - 'field': Standard method using discrete pixel counts based
            on a segmentation of the rate map into labeled regions (fields).
            This method
            is faster, but can be inaccurate for small regions and will not
            work well for positional data that has been masked for direction of
            running say (ie linear track)
        - 'clump_runs': Exact method which accounts for filtered data better by
            looking for contiguous areas of the positional data that are NOT
            masked (uses np.ma.clump_unmasked)
        cache : bool, optional
        Determine whether to cache calculated properties. The computation is
        much faster for cached properties, whereas the memory consumption
        increases.
    extra_properties : Iterable of callables
        Add extra property computation functions that are not included with
        skimage. The name of the property is derived from the function name,
        the dtype is inferred by calling the function on a small sample.
        If the name of an extra property clashes with the name of an existing
        property the extra property will not be visible and a UserWarning is
        issued. A property computation function must take a region mask as its
        first argument. If the property requires an intensity image, it must
        accept the intensity image as the second argument.
    spacing: tuple of float, shape (ndim,)
        The pixel spacing along each axis of the image.
    offset : array-like of int, shape `(label_image.ndim,)`, optional
        Coordinates of the origin ("top-left" corner) of the label image.
        Normally this is ([0, ]0, 0), but it might be different if one wants
        to obtain regionprops of subvolumes within a larger volume.
    **kwargs : keyword arguments
        Additional arguments passed to the FieldProps constructor.
        Legal arguments are:
            pos_sample_rate : int
            min_run_length : int



        .. versionadded:: 0.14.1

    Returns
    -------
    properties : list of RegionProperties
        Each item describes one labeled region, and can be accessed using the
        attributes listed below.

    Notes
    -----
    The following properties can be accessed as attributes or keys:

    **area** : float
        Area of the region i.e. number of pixels of the region scaled
        by pixel-area.
    **area_bbox** : float
        Area of the bounding box i.e. number of pixels of bounding box scaled
        by pixel-area.
    **area_convex** : float
        Area of the convex hull image, which is the smallest convex
        polygon that encloses the region.
    **area_filled** : float
        Area of the region with all the holes filled in.
    **axis_major_length** : float
        The length of the major axis of the ellipse that has the same
        normalized second central moments as the region.
    **axis_minor_length** : float
        The length of the minor axis of the ellipse that has the same
        normalized second central moments as the region.
    **bbox** : tuple
        Bounding box ``(min_row, min_col, max_row, max_col)``.
        Pixels belonging to the bounding box are in the half-open interval
        ``[min_row; max_row)`` and ``[min_col; max_col)``.
    **centroid** : array
        Centroid coordinate tuple ``(row, col)``.
    **centroid_local** : array
        Centroid coordinate tuple ``(row, col)``, relative to region bounding
        box.
    **centroid_weighted** : array
        Centroid coordinate tuple ``(row, col)`` weighted with intensity
        image.
    **centroid_weighted_local** : array
        Centroid coordinate tuple ``(row, col)``, relative to region bounding
        box, weighted with intensity image.
    **coords_scaled** : (K, 2) ndarray
        Coordinate list ``(row, col)`` of the region scaled by ``spacing``.
    **coords** : (K, 2) ndarray
        Coordinate list ``(row, col)`` of the region.
    **eccentricity** : float
        Eccentricity of the ellipse that has the same second-moments as the
        region. The eccentricity is the ratio of the focal distance
        (distance between focal points) over the major axis length.
        The value is in the interval [0, 1).
        When it is 0, the ellipse becomes a circle.
    **equivalent_diameter_area** : float
        The diameter of a circle with the same area as the region.
    **euler_number** : int
        Euler characteristic of the set of non-zero pixels.
        Computed as number of connected components subtracted by number of
        holes (input.ndim connectivity). In 3D, number of connected
        components plus number of holes subtracted by number of tunnels.
    **extent** : float
        Ratio of pixels in the region to pixels in the total bounding box.
        Computed as ``area / (rows * cols)``
    **feret_diameter_max** : float
        Maximum Feret's diameter computed as the longest distance between
        points around a region's convex hull contour as determined by
        ``find_contours``. [5]_
    **image** : (H, J) ndarray
        Sliced binary region image which has the same size as bounding box.
    **image_convex** : (H, J) ndarray
        Binary convex hull image which has the same size as bounding box.
    **image_filled** : (H, J) ndarray
        Binary region image with filled holes which has the same size as
        bounding box.
    **image_intensity** : ndarray
        Image inside region bounding box.
    **inertia_tensor** : ndarray
        Inertia tensor of the region for the rotation around its mass.
    **inertia_tensor_eigvals** : tuple
        The eigenvalues of the inertia tensor in decreasing order.
    **intensity_max** : float
        Value with the greatest intensity in the region.
    **intensity_mean** : float
        Value with the mean intensity in the region.
    **intensity_min** : float
        Value with the least intensity in the region.
    **intensity_std** : float
        Standard deviation of the intensity in the region.
    **label** : int
        The label in the labeled input image.
    **moments** : (3, 3) ndarray
        Spatial moments up to 3rd order::

            m_ij = sum{ array(row, col) * row^i * col^j }

        where the sum is over the `row`, `col` coordinates of the region.
    **moments_central** : (3, 3) ndarray
        Central moments (translation invariant) up to 3rd order::

            mu_ij = sum{ array(row, col) * (row - row_c)^i * (col - col_c)^j }

        where the sum is over the `row`, `col` coordinates of the region,
        and `row_c` and `col_c` are the coordinates of the region's centroid.
    **moments_hu** : tuple
        Hu moments (translation, scale and rotation invariant).
    **moments_normalized** : (3, 3) ndarray
        Normalized moments (translation and scale invariant) up to 3rd order::

            nu_ij = mu_ij / m_00^[(i+j)/2 + 1]

        where `m_00` is the zeroth spatial moment.
    **moments_weighted** : (3, 3) ndarray
        Spatial moments of intensity image up to 3rd order::

            wm_ij = sum{ array(row, col) * row^i * col^j }

        where the sum is over the `row`, `col` coordinates of the region.
    **moments_weighted_central** : (3, 3) ndarray
        Central moments (translation invariant) of intensity image up to
        3rd order::

            wmu_ij = sum{ array(row, col) * (row - row_c)^i * (col - col_c)^j }

        where the sum is over the `row`, `col` coordinates of the region,
        and `row_c` and `col_c` are the coordinates of the region's weighted
        centroid.
    **moments_weighted_hu** : tuple
        Hu moments (translation, scale and rotation invariant) of intensity
        image.
    **moments_weighted_normalized** : (3, 3) ndarray
        Normalized moments (translation and scale invariant) of intensity
        image up to 3rd order::

            wnu_ij = wmu_ij / wm_00^[(i+j)/2 + 1]

        where ``wm_00`` is the zeroth spatial moment (intensity-weighted area).
    **num_pixels** : int
        Number of foreground pixels.
    **orientation** : float
        Angle between the 0th axis (rows) and the major
        axis of the ellipse that has the same second moments as the region,
        ranging from `-pi/2` to `pi/2` counter-clockwise.
    **perimeter** : float
        Perimeter of object which approximates the contour as a line
        through the centers of border pixels using a 4-connectivity.
    **perimeter_crofton** : float
        Perimeter of object approximated by the Crofton formula in 4
        directions.
    **slice** : tuple of slices
        A slice to extract the object from the source image.
    **solidity** : float
        Ratio of pixels in the region to pixels of the convex hull image.

    Each region also supports iteration, so that you can do::

      for prop in region:
          print(prop, region[prop])

    See Also
    --------
    label

    References
    ----------
    .. [1] Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:
           Core Algorithms. Springer-Verlag, London, 2009.
    .. [2] B. Jähne. Digital Image Processing. Springer-Verlag,
           Berlin-Heidelberg, 6. edition, 2005.
    .. [3] T. H. Reiss. Recognizing Planar Objects Using Invariant Image
           Features, from Lecture notes in computer science, p. 676. Springer,
           Berlin, 1993.
    .. [4] https://en.wikipedia.org/wiki/Image_moment
    .. [5] W. Pabst, E. Gregorová. Characterization of particles and particle
           systems, pp. 27-28. ICT Prague, 2007.
           https://old.vscht.cz/sil/keramika/Characterization_of_particles/CPPS%20_English%20version_.pdf

    Examples
    --------
    >>> from skimage import data, util
    >>> from skimage.measure import label, regionprops
    >>> img = util.img_as_ubyte(data.coins()) > 110
    >>> label_img = label(img, connectivity=img.ndim)
    >>> props = regionprops(label_img)
    >>> # centroid of first labeled object
    >>> props[0].centroid
    (22.72987986048314, 81.91228523446583)
    >>> # centroid of first labeled object
    >>> props[0]['centroid']
    (22.72987986048314, 81.91228523446583)

    Add custom measurements by passing functions as ``extra_properties``

    >>> from skimage import data, util
    >>> from skimage.measure import label, regionprops
    >>> import numpy as np
    >>> img = util.img_as_ubyte(data.coins()) > 110
    >>> label_img = label(img, connectivity=img.ndim)
    >>> def pixelcount(regionmask):
    ...     return np.sum(regionmask)
    >>> props = regionprops(label_img, extra_properties=(pixelcount,))
    >>> props[0].pixelcount
    7741
    >>> props[1]['pixelcount']
    42

    """
    assert label_image.shape == binned_data.binned_data[0].shape

    if label_image.ndim not in (1, 2, 3):
        raise TypeError("Only 1-D, 2-D and 3-D images supported.")

    if not np.issubdtype(label_image.dtype, np.integer):
        if np.issubdtype(label_image.dtype, bool):
            raise TypeError(
                "Non-integer image types are ambiguous: "
                "use skimage.measure.label to label the connected "
                "components of label_image, "
                "or label_image.astype(np.uint8) to interpret "
                "the True values as a single label."
            )
        else:
            raise TypeError("Non-integer label_image types are ambiguous")

    offset_arr = None

    if len(binned_data.bin_edges) == 1:
        be = binned_data.bin_edges[0]
        x_bins = np.digitize(xy, be[:-1])
        xy_field_label = label_image[x_bins - 1]

    elif len(binned_data.bin_edges) == 2:
        ye, xe = binned_data.bin_edges
        x_bins = np.digitize(xy[0], xe[:-1])
        y_bins = np.digitize(xy[1], ye[:-1])
        xy_field_label = label_image[y_bins - 1, x_bins - 1]
    # deal with pos values that are masked and set to 0
    # if the mask ndim == 0 then no mask has been applied
    # if xy.mask.ndim == 1:
    #     xy_field_label[xy.mask] = 0

    min_run = kwargs.get("min_run_length", 2)

    if method == "field":

        labelled_runs = labelContigNonZeroRuns(xy_field_label)
        run_starts = getLabelStarts(labelled_runs)
        run_stops = getLabelEnds(labelled_runs)
        all_run_slices = [
            slice(run_starts[i], run_stops[i] + 1)
            for i in range(len(run_starts))
            if (run_stops[i] + 1 - run_starts[i]) >= min_run
        ]

    elif method == "clump_runs":
        # still need xy_field_label to get which field each
        # run belongs to
        if xy.ndim == 1:
            clumps = ma.clump_unmasked(xy)
        else:
            clumps = ma.clump_unmasked(xy[0])

        all_run_slices = [
            slice(c.start, c.stop) for c in clumps if (c.stop - c.start) >= min_run
        ]
    # remove runs that aren't in any field
    run_slices = [rs for rs in all_run_slices if np.any(xy_field_label[rs] != 0)]
    # TODO: try getting run starts and stops via np.ma.clump_unmasked
    # Need to add a method argument to this function that determines
    # whether to use ma.clump_unmasked or the old "field"-based method

    pos_sample_rate = kwargs.get("pos_sample_rate", 50)

    # calculate the speed for possibly filtering runs later
    speed = None
    if xy is not None:
        if xy.ndim == 1:  # linear track data
            speed = ma.MaskedArray(np.abs(ma.ediff1d(xy, to_begin=0)) * pos_sample_rate)
        else:
            speed = ma.MaskedArray(
                np.abs(ma.ediff1d(np.hypot(xy[0], xy[1]), to_begin=0)) * pos_sample_rate
            )

        # speed = ma.append(speed, speed[-1])
    # make a mask array initially all False
    if xy.ndim == 1:
        mask = np.zeros(xy.shape[0], dtype=bool)
    else:
        mask = np.zeros(xy.shape[1], dtype=bool)

    regions = []

    run_id = 0

    objects = ndi.find_objects(label_image)

    for i, sl in enumerate(objects):
        if sl is None:
            continue

        label = i + 1

        props = FieldProps(
            sl,
            label,
            label_image,
            binned_data,
            cache=cache,
            spacing=spacing,
            extra_properties=extra_properties,
            offset=offset_arr,
        )
        # perimeter_coords = props.perimeter_coords
        # get the runs through this field
        sub_slices = []
        for i_slice in run_slices:
            if label in xy_field_label[i_slice]:
                idx = xy_field_label[i_slice] == label
                # find sub runs through this run that are contiguous
                # note that there may be more than one run through
                # the field in this slice
                run_vals, run_starts, run_lens = find_runs(idx)
                # find those sub runs corresponding to True values
                true_idx = np.nonzero(run_vals)[0]
                for v in true_idx:
                    sub_slice = slice(
                        i_slice.start + run_starts[v],
                        i_slice.start + run_starts[v] + run_lens[v],
                    )
                    if sub_slice.stop - sub_slice.start >= min_run:
                        sub_slices.append(sub_slice)

        # extract a few metrics for instantiating the RunProps objects...
        peak_xy = props.xy_at_peak
        max_index = props.max_index
        perimeter_coords = props.global_perimeter_coords

        runs = []
        for rs in sub_slices:
            # make sure the run isn't completely masked
            if not np.all(xy[..., rs]):
                continue

            r = RunProps(
                props,
                run_id,
                rs,
                spike_times,
                mask[..., rs],
                xy[..., rs],
                speed[rs],
                peak_xy,
                max_index,
                perimeter_coords,
            )
            run_id += 1
            runs.append(r)
        # ... and add the list of runs to the FieldProps instance
        # this will print out the number of potential runs to the console
        props.runs = runs

        regions.append(props)

    return regions
