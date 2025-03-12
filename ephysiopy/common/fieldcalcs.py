import numpy as np
from scipy import signal
from scipy import ndimage as ndi
from scipy import spatial
from scipy import stats
import warnings
from skimage.segmentation import watershed
from ephysiopy.common.utils import (
    blur_image,
    BinnedData,
    MapType,
    bwperim,
    circ_abs,
    labelContigNonZeroRuns,
    getLabelStarts,
    getLabelEnds,
    pol2cart,
    cart2pol,
    repeat_ind,
)
import skimage
from skimage.measure._regionprops import (
    RegionProperties,
    _infer_number_of_required_args,
    _require_intensity_image,
)
from skimage.measure._regionprops import PROPS as _PROPS
from astropy.convolution import Gaussian2DKernel as gk2d
from astropy.convolution import interpolate_replace_nans
import copy

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
PROP_VALS = set(PROPS.values())


class LFPSegment(object):
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
        field_label: int,
        run_label: int,
        slice: slice,
        spike_times: np.ndarray,
        signal: np.ndarray,
        filtered_signal: np.ndarray,
        phase: np.ndarray,
        amplitude: np.ndarray,
        sample_rate: float | int,
        filter_band: tuple,
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
        spike_times : np.ndarray
            the times at which the spikes were emitted (seonds)
        signal, filtered_signal, phase, amplitude : np.ndarray
            the raw, filtered, phase and amplitude of the LFP for a given segment
        sample_rate : int, float
            the sample rate for the LFP signal
        filter_band : tuple
            the bandpass filter values applied to the raw signal to give filtered_signal

        """
        self.field_label = field_label
        self.run_label = run_label  # integer identity that should match with a run
        self.slice = slice  # an index into the main LFP data
        self.spike_times = spike_times
        self.signal = signal
        self.filtered_signal = filtered_signal
        self.phase = phase
        self.amplitude = amplitude
        self.sample_rate = sample_rate
        self.filter_band = filter_band


class RunProps(object):
    """
    A custom class for holding information about runs through a receptive field

    Each run needs to have some information about the field to which it belongs
    so the constructor takes in the peak x-y coordinate of the field and its index
    as well as the coordinates of the perimeter of the field

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
    duration: int
        the total duration of a run
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
    spike_position_index : np.ndarray
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
        label,
        slice,
        xy_coords,
        spike_count,
        speed,
        peak_xy,
        max_index,
        perimeter_coords,
    ):
        """
        Parameters
        ----------
        label : int
            the field label the run belongs to
        slice : slice
            the slice into the position array that holds all the position data
        xy_coords : np.ndarray
            the xy data for this run (global coordinates)
        spike_count : np.ndarray
            the spike count for each position sample in this run
        speed : np.ndarray
            the speed for this run
        peak_xy : np.ndarray
            the peak location of the field for the this run
        max_index : np.ndarray
            the index (r,c) of the maximum of the firing field containing this run
        perimeter_coords : np.ndarray
            the xy coordinates of the perimeter of the field containing this run
        """
        assert xy_coords.shape[1] == len(spike_count)
        self.label = label
        self._xy_coords = xy_coords
        self._slice = slice
        self._spike_count = spike_count
        self._speed = speed
        self._peak_xy = peak_xy
        self._max_index = max_index
        self._perimeter_coords = perimeter_coords
        self.xy_is_smoothed = False

    def __str__(self):
        return f"id: {self.label}: {np.sum(self._spike_count)} spikes"

    @property
    def xy(self):
        return self._xy_coords

    @xy.setter
    def xy(self, val):
        self._xy_coords = val

    @property
    def hdir(self):
        d = np.arctan2(np.diff(self.xy[1]), np.diff(self.xy[0]))
        d = np.append(d, d[-1])
        return np.rad2deg(d)

    @property
    def speed(self):
        return self._speed

    @property
    def min_speed(self):
        return np.nanmin(self._speed)

    def __len__(self):
        return self._slice.stop - self._slice.start

    @property
    def cumulative_time(self) -> np.ndarray:
        return np.arange(len(self))

    @property
    def duration(self):
        return self._slice.stop - self._slice.start

    @property
    def n_spikes(self):
        return np.nansum(self._spike_count)

    @property
    def run_start(self):
        return self._slice.start

    @property
    def run_stop(self):
        return self._slice.stop

    @property
    def mean_direction(self):
        return stats.circmean(np.deg2rad(self.hdir))

    @property
    def current_direction(self):
        return self.rho * np.cos(np.deg2rad(self.hdir) - self.phi)

    @property
    def cumulative_distance(self):
        d = np.sqrt(np.abs(np.diff(np.power(self.rho, 2))))
        d = np.insert(d, 0, 0)
        return np.cumsum(d)

    @property
    def spike_position_index(self):
        return np.take(
            range(self._slice.start, self._slice.stop), repeat_ind(self._spike_count)
        )

    @property
    def observed_spikes(self):
        return self._spike_count

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
        return expected_rate_at_pos[self._slice] / sample_rate

    def overdispersion(self, spike_train: np.ndarray, sample_rate: int = 50) -> float:
        """
        The overdispersion map for this run

        Parameters
        ----------
        spike_train : np.mdarray
            the spike train (spikes binned up by position) for the whole trial. Same
            length as the trial n_samples
        sample_rate : int
        """
        obs_spikes = np.sum(self._spike_count)
        expt_spikes = self.expected_spikes(spike_train, sample_rate)
        Z = np.nan
        if obs_spikes >= expt_spikes:
            Z = (obs_spikes - expt_spikes - 0.5) / np.sqrt(expt_spikes)
        else:
            Z = (obs_spikes - expt_spikes + 0.5) / np.sqrt(expt_spikes)
        return Z

    def smooth_xy(self, k: float, spatial_lp_cut: int, sample_rate: int):
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
        h = signal.firwin(
            int(f_len),
            fs=sample_rate,
            cutoff=spatial_lp_cut / sample_rate * 2,
            window="blackman",
        )
        padlen = 2 * len(h)
        if padlen == self.xy.shape[1]:
            padlen = padlen - 1
        self.xy = signal.filtfilt(h, [1], self.xy, padlen=padlen, axis=1)
        self.xy_is_smoothed = True

    @property
    def xy_angle_to_peak(self):
        xy_to_peak = (self.xy.T - self._peak_xy).T
        return np.arctan2(xy_to_peak[1], xy_to_peak[0])

    @property
    def xy_dist_to_peak(self):
        xy_to_peak = (self.xy.T - self._peak_xy).T
        return np.hypot(xy_to_peak[0], xy_to_peak[1])

    @property
    def xy_dist_to_peak_normed(self):
        x_y = self.r_and_phi_to_x_and_y
        return np.hypot(x_y[0], x_y[1])

    def perimeter_minus_field_max(self):
        mi = self._max_index
        perimeter_coords = self._perimeter_coords
        return (
            perimeter_coords[0] - mi[0],
            perimeter_coords[1] - mi[1],
        )

    def perimeter_angle_from_peak(self):
        perimeter_minus_field_max = self.perimeter_minus_field_max()
        return np.arctan2(perimeter_minus_field_max[0], perimeter_minus_field_max[1])

    @property
    def pos_xy(self):
        pos_x, pos_y = pol2cart(self.pos_r, self.pos_phi)
        return np.vstack([pos_x, pos_y])

    @property
    def pos_r(self):
        angle_df = circ_abs(
            self.perimeter_angle_from_peak()[:, np.newaxis]
            - self.xy_angle_to_peak[np.newaxis, :]
        )
        perimeter_idx = np.argmin(angle_df, 0)
        tmp = (
            self._perimeter_coords[1][perimeter_idx] - self._max_index[1],
            self._perimeter_coords[0][perimeter_idx] - self._max_index[0],
        )

        perimeter_dist_to_peak = np.hypot(tmp[0], tmp[1])
        r = self.xy_dist_to_peak / perimeter_dist_to_peak
        capped_vals = r >= 1
        r[capped_vals] = 1
        return r

    # calculate the angular distance between the runs main direction and the
    # pos's direction to the peak centre
    @property
    def pos_phi(self):
        return self.xy_angle_to_peak - self.mean_direction

    @property
    def rho(self):
        rho, _ = cart2pol(self.pos_xy[0], self.pos_xy[1])
        return rho

    @property
    def phi(self):
        _, phi = cart2pol(self.pos_xy[0], self.pos_xy[1])
        return phi

    @property
    def r_and_phi_to_x_and_y(self):
        return np.vstack(pol2cart(self.rho, self.phi))

    """
    Define a measure of tortuosity to see how direct the run was
    from field entry to exit. It's jsut the ratio of the distance between
    a straight line joining the entry-exit points and the actual distance
    of the run
    """

    @property
    def tortuosity(self):
        direct_line_distance = np.hypot(
            self.xy[0, 0] - self.xy[0, -1], self.xy[1, 0] - self.xy[1, -1]
        )
        xy_df = np.diff(self.xy)
        traversed_distance = np.sum(np.hypot(xy_df[0], xy_df[1]))
        return direct_line_distance / traversed_distance


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
    spike_position_index : np.ndarray
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
        and the distance from the field peak to the point on the perimeter that is most
        colinear with the position sample
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

    # The maximum index of the intensity image for the region
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
    def cumulative_time(self) -> list:
        return [r.cumulative_time for r in self.runs]

    @property
    def runs_speed(self) -> list:
        return [r._speed for r in self.runs]

    @property
    def runs_observed_spikes(self) -> np.ndarray:
        return np.concatenate([r.observed_spikes for r in self.runs])

    @property
    def spike_position_index(self):
        return np.concatenate([r.spike_position_index for r in self.runs])

    # The x-y coordinate at the field peak
    @property
    def xy_at_peak(self) -> np.ndarray:
        mi = self.max_index
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
    @property
    def bw_perim(self) -> np.ndarray:
        return bwperim(self.image)

    @property
    def perimeter_coords(self) -> tuple:
        return np.nonzero(self.bw_perim)

    @property
    def global_perimeter_coords(self) -> np.ndarray:
        perim_xy = self.perimeter_coords
        x = self.binned_data.bin_edges[1][perim_xy[1] + self.slice[1].start]
        y = self.binned_data.bin_edges[0][perim_xy[0] + self.slice[0].start]
        return np.array([x, y])

    @property
    def perimeter_minus_field_max(self) -> np.ndarray:
        mi = self.max_index
        perimeter_coords = self.perimeter_coords
        return np.array([perimeter_coords[0] - mi[0], perimeter_coords[1] - mi[1]])

    # The angle each point on the perimeter makes to the field peak
    @property
    def perimeter_angle_from_peak(self) -> np.ndarray:
        perimeter_minus_field_max = self.perimeter_minus_field_max()
        return np.arctan2(perimeter_minus_field_max[0], perimeter_minus_field_max[1])

    # The distance of each point on the perimeter to the field peak
    @property
    def perimeter_dist_from_peak(self) -> np.ndarray:
        perimeter_minus_field_max = self.perimeter_minus_field_max()
        return np.hypot(perimeter_minus_field_max[0], perimeter_minus_field_max[1])

    @property
    def bin_coords(self) -> np.ndarray:
        bin_edges = self.binned_data.bin_edges
        return np.array(
            [bin_edges[1][self.coords[:, 1]], bin_edges[0][self.coords[:, 0]]]
        )

    @property
    def phi(self) -> np.ndarray:
        """
        Calculate the angular distance between the mean direction of each run and
        each position samples direction to the field centre
        """
        return np.concatenate([r.phi for r in self.runs]).T

    @property
    def rho(self) -> np.ndarray:
        return np.concatenate([r.rho for r in self.runs])

    @property
    def pos_xy(self) -> np.ndarray:
        return np.concatenate([r.pos_xy.T for r in self.runs]).T

    @property
    def pos_phi(self) -> np.ndarray:
        """
        Calculate the angular distance between the mean direction of each run and
        each position samples direction to the field centre
        """
        return np.concatenate([r.pos_phi for r in self.runs]).T

    @property
    def pos_r(self) -> np.ndarray:
        """
        Calculate the ratio of the distance from the field peak to the position sample
        and the distance from the field peak to the point on the perimeter that is most
        colinear with the position sample

        NB The values just before being returned can be >= 1 so these are capped to 1
        """
        return np.concatenate([r.pos_r for r in self.runs])

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
    def current_direction(self) -> list:
        return [r.current_direction for r in self.runs]

    @property
    def cumulative_distance(self) -> list:
        return [r.cumulative_distance for r in self.runs]

    @property
    def projected_direction(self) -> np.ndarray:
        """
        direction projected onto the mean run direction is just the x-coord
        when cartesian x and y is converted to from polar rho and phi
        """
        return np.concatenate([r.pos_xy[0] for r in self.runs])

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
            raise AttributeError(f"'{type(self)}' object has no attribute '{attr}'")

    def __str__(self):
        """
        Override the string representation printed to the console
        """
        return f"Field {self.label} has {len(self.runs)} runs"

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
        self, expected_rate_at_pos: np.ndarray, sample_rate: int = 50
    ) -> np.ndarray:
        """
        Calculate the expected number of spikes along each run given the
        whole ratemap.

        Parameters
        ----------
        expected_rate_at_pos : np.ndarray
            the rate seen at each xy position of the whole trial
        sample_rate : int
            the sample rate of the position data

        Returns
        -------
        np.ndarray
            the expected rate at each xy position for each run
        """
        return np.concatenate(
            [r.expected_spikes(expected_rate_at_pos, sample_rate) for r in self.runs]
        )

    def overdispersion(
        self, spike_train: np.ndarray, sample_rate: int = 50
    ) -> np.ndarray:
        """
        Calculate the overdispersion for each run through the field

        Parameters
        ----------
        spike_train : np.ndarray
            the spike train (spikes binned up by position) for the whole trial. Same
            length as the trial n_samples
        sample_rate : int
            the sample rate of the position data

        Returns
        -------
        np.ndarray
            the overdispersion for each run through the field
        """
        return np.array([r.overdispersion(spike_train, sample_rate) for r in self.runs])


def fieldprops(
    label_image,
    binned_data,
    xy,
    spikes_per_pos,
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
    xy : (2 x N) ndarray
        The x-y coordinates for all runs through the field corresponding to
        a particular label
    binned_data : BinnedData instance from ephysiopy.common.utils
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

    Returns
    -------
    properties : list of RegionProperties
        Each item describes one labeled region, and can be accessed using the
        attributes listed below.

    Notes
    -----
    The following properties can be accessed as attributes or keys:

    **area** : float
        Area of the region i.e. number of pixels of the region scaled by pixel-area.
    **area_bbox** : float
        Area of the bounding box i.e. number of pixels of bounding box scaled by pixel-area.
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
    if spikes_per_pos is not None:
        assert len(spikes_per_pos) == xy.shape[1]

    if label_image.ndim not in (2, 3):
        raise TypeError("Only 2-D and 3-D images supported.")

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

    if offset is None:
        offset_arr = np.zeros((label_image.ndim,), dtype=int)
    else:
        offset_arr = np.asarray(offset)
        if offset_arr.ndim != 1 or offset_arr.size != label_image.ndim:
            raise ValueError(
                "Offset should be an array-like of integers "
                "of shape (label_image.ndim,); "
                f"{offset} was provided."
            )

    pos_sample_rate = kwargs.get("pos_sample_rate", 50)
    ye, xe = binned_data.bin_edges
    x_bins = np.digitize(xy[0], xe[:-1])
    y_bins = np.digitize(xy[1], ye[:-1])
    xy_field_label = label_image[y_bins - 1, x_bins - 1]
    labelled_runs = labelContigNonZeroRuns(xy_field_label)
    run_starts = getLabelStarts(labelled_runs)
    run_stops = getLabelEnds(labelled_runs)

    # calculate the speed for possibly filtering runs later
    speed = None
    if xy is not None:
        speed = np.ma.MaskedArray(
            np.abs(np.ma.ediff1d(np.hypot(xy[0], xy[1])) * pos_sample_rate)
        )
        speed = np.append(speed, speed[-1])

    regions = []

    run_id = 0

    objects = ndi.find_objects(label_image)

    for i, sl in enumerate(objects):
        if sl is None:
            continue

        label = i + 1

        # get the runs through this field and filter for min run length
        run_index = np.unique(labelled_runs[xy_field_label == label])
        run_slices = [
            slice(run_starts[ri - 1], run_stops[ri - 1] + 1)
            for ri in run_index
            if (run_stops[ri - 1] - run_starts[ri - 1]) > 2
        ]
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
        # extract a few metrics for instantiating the RunProps objects...
        peak_xy = props.xy_at_peak
        max_index = props.max_index
        perimeter_coords = props.perimeter_coords
        runs = []
        for rs in run_slices:
            r = RunProps(
                run_id,
                rs,
                xy[:, rs],
                spikes_per_pos[rs],
                speed[rs],
                peak_xy,
                max_index,
                perimeter_coords,
            )
            run_id += 1
            runs.append(r)
        # ... and add the list of runs to the FieldProps instance
        props.runs = runs

        regions.append(props)

    return regions


def infill_ratemap(rmap: np.ndarray) -> np.ndarray:
    """
    The ratemaps used in the phasePrecession2D class are a) super smoothed and
    b) very large i.e. the bins per cm is low. This
    results in firing fields that have lots of holes (nans) in them. We want to
    smooth over these holes so we can construct measures such as the expected
    rate in a given bin whilst also preserving whatever 'geometry' of the
    environment exists in the ratemap as a result of where position has been
    sampled. That is, if non-sampled positions are designated with nans, then we
    want to smooth over those that in theory could have been sampled and keep
    those that never could have been.

    Parameters
    ----------
    rmap : np.ndarray
        The ratemap to be filled

    Returns
    -------
    np.ndarray
        The filled ratemap
    """
    outline = np.isfinite(rmap)
    mask = ndi.binary_fill_holes(outline)
    rmap = np.ma.MaskedArray(rmap, np.invert(mask))
    rmap[np.invert(mask)] = 0
    k = gk2d(x_stddev=1)
    output = interpolate_replace_nans(rmap, k)
    output[np.invert(mask)] = np.nan
    return output


def reduce_labels(A: np.ndarray, labels: np.ndarray, reduce_by: float = 50) -> list:
    """
    Reduce the labelled data in A by restricting the values to reduce_by % of
    the maximum in each local labeled section of A - kind of a quantitative local watershed

    Parameters
    ----------
    A : np.ndarray
        The data to be reduced
    labels : np.ndarray
        The labels to be used to partition the data
    reduce_by : float
        The percentage of the maximum value in each label to reduce by

    Returns
    -------
    list of np.ndarray
        The reduced data
    """
    assert A.shape == labels.shape
    out = []
    for label in np.unique(labels):
        m = np.ma.MaskedArray(A, np.invert(labels == label))
        m = np.ma.masked_where(
            A[labels == label] < np.nanmax(A[labels == label]) * reduce_by / 100, A
        )
        out.append(m)
    return out


def partitionFields(
    binned_data: BinnedData,
    field_threshold_percent: int = 50,
    field_rate_threshold: float = 0.5,
    area_threshold=0.01,
) -> tuple[np.ndarray, ...]:
    """
    Partitions spikes into fields by finding the watersheds around the
    peaks of a super-smoothed ratemap

    Parameters
    ----------
    binned_data : BinnedData
        an instance of ephysiopy.common.utils.BinnedData
    field_threshold_percent : int
        removes pixels in a field that fall below this percent of the maximum firing rate in the field
    field_rate_threshold : float
        anything below this firing rate in Hz threshold is set to 0
    area_threshold : float
        defines the minimum field size as a proportion of the
        environment size. Default of 0.01 says a field has to be at
        least 1% of the size of the environment i.e.
        binned_area_width * binned_area_height to be counted as a field

    Returns
    -------
    tuple of np.ndarray
        peaksXY - The xy coordinates of the peak rates in
        each field
        peaksRate - The peak rates in peaksXY
        labels - An array of the labels corresponding to each field (starting  1)
        rmap - The ratemap of the tetrode / cluster
    """
    ye, xe = binned_data.bin_edges
    rmap = binned_data[0]
    # start image processing:
    # Usually the binned_data has a large number of bins which potentially
    # leaves lots of "holes" in the ratemap (nans) as there will be lots of
    # positions that aren't sampled. Get over this by preserving areas outside
    # the sampled area as nans whilst filling in the nans that live within the
    # receptive fields
    rmap_filled = infill_ratemap(rmap.binned_data)
    # get the labels
    # binarise the ratemap so that anything above field_rate_threshold is set to 1
    # and anything below to 0
    rmap_to_label = copy.copy(rmap_filled.data)
    rmap_to_label[np.isnan(rmap_filled)] = 0
    rmap_to_label[rmap_to_label > field_rate_threshold] = 1
    rmap_to_label[rmap_to_label < field_rate_threshold] = 0
    labels = skimage.measure.label(rmap_to_label, background=0)
    # labels is now a labelled int array from 0 to however many fields have
    # been detected
    # Get the coordinates of the peak firing rate within each firing field...
    fieldId, _ = np.unique(labels, return_index=True)
    fieldId = fieldId[1::]
    peakCoords = np.array(
        ndi.maximum_position(rmap_filled, labels=labels, index=fieldId)
    ).astype(int)
    # ... and convert these to actual x-y coordinates wrt to the position data
    peaksXY = np.vstack((xe[peakCoords[:, 1]], ye[peakCoords[:, 0]]))

    # TODO: this labeled_comprehension is not working too well for fields that
    # have a fairly uniform firing rate distribution across them (using np.nanmax
    # in the function fn defined for use in the labeled_comprehension)
    # or those that have nicely gaussian shaped fields (which was using np.nanmedian)
    # find the peak rate at each of the centre of the detected fields to
    # subsequently threshold the field at some fraction of the peak value
    # use a labeled_comprehension to do this

    def fn(val, pos):
        return pos[val < (np.nanmax(val) * (field_threshold_percent / 100))]

    #
    indices = ndi.labeled_comprehension(
        rmap_filled, labels, None, fn, np.ndarray, 0, True
    )

    # breakpoint()
    labels[np.unravel_index(indices=indices, shape=labels.shape)] = 0
    min_field_size = np.ceil(np.prod(labels.shape) * area_threshold).astype(int)
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
    # breakpoint()
    peakCoords = np.array(
        ndi.maximum_position(rmap_filled, labels=labels, index=fieldId)
    ).astype(int)
    peaksXY = np.vstack((xe[peakCoords[:, 1]], ye[peakCoords[:, 0]]))
    peakRates = rmap_filled[peakCoords[:, 0], peakCoords[:, 1]]
    peakLabels = labels[peakCoords[:, 0], peakCoords[:, 1]]
    peaksXY = peaksXY[:, peakLabels - 1]
    peaksRate = peakRates[peakLabels - 1]
    return peaksXY, peaksRate, labels, rmap_filled


"""
These methods differ from MapCalcsGeneric in that they are mostly
concerned with treating rate maps as images as opposed to using
the spiking information contained within them. They therefore mostly
deals with spatial rate maps of place and grid cells.
"""


def get_mean_resultant(ego_boundary_map: np.ndarray) -> np.complex128 | float:
    """
    Calculates the mean resultant vector of a boundary map in egocentric coordinates

    Parameters
    ----------
    ego_boundary_map : np.ndarray
        The egocentric boundary map

    Returns
    -------
    float 
        The mean resultant vector of the egocentric boundary map

    Notes
    -----
    See Hinman et al., 2019 for more details

    """
    if np.nansum(ego_boundary_map) == 0:
        return np.nan
    m, n = ego_boundary_map.shape
    angles = np.linspace(0, 2 * np.pi, n)
    MR = np.nansum(np.nansum(ego_boundary_map, 0) * np.power(np.e, angles * 1j)) / (
        n * m
    )
    return MR


def get_mean_resultant_length(ego_boundary_map: np.ndarray, **kwargs) -> float:
    '''
    Calculates the length of the mean resultant vector of a 
    boundary map in egocentric coordinates

    Parameters
    ----------
    ego_boundary_map : np.ndarray
        The egocentric boundary map

    Returns
    -------
    float 
        The length of the mean resultant vector of the egocentric boundary map

    Notes
    -----
    See Hinman et al., 2019 for more details

    '''
    MR = get_mean_resultant(ego_boundary_map, **kwargs)
    return np.abs(MR)


def get_mean_resultant_angle(ego_boundary_map: np.ndarray, **kwargs) -> float:
    '''
    Calculates the angle of the mean resultant vector of a 
    boundary map in egocentric coordinates

    Parameters
    ----------
    ego_boundary_map : np.ndarray
        The egocentric boundary map

    Returns
    -------
    float 
        The angle mean resultant vector of the egocentric boundary map

    Notes
    -----
    See Hinman et al., 2019 for more details

    '''
    MR = get_mean_resultant(ego_boundary_map, **kwargs)
    return np.rad2deg(np.arctan2(np.imag(MR), np.real(MR)))


def field_lims(A):
    """
    Returns a labelled matrix of the ratemap A.
    Uses anything greater than the half peak rate to select as a field.
    Data is heavily smoothed.

    Parameters
    ----------
    A : BinnedData
        A BinnedData instance containing the ratemap

    Returns
    -------
    np.ndarray
        The labelled ratemap
    """
    Ac = A.binned_data[0]
    nan_idx = np.isnan(Ac)
    Ac[nan_idx] = 0
    h = int(np.max(Ac.shape) / 2)
    sm_rmap = blur_image(A, h, ftype="gaussian").binned_data[0]
    thresh = np.max(sm_rmap.ravel()) * 0.2  # select area > 20% of peak
    distance = ndi.distance_transform_edt(sm_rmap > thresh)
    peak_idx = skimage.feature.peak_local_max(
        distance, exclude_border=False, labels=sm_rmap > thresh
    )
    mask = np.zeros_like(distance, dtype=bool)
    mask[tuple(peak_idx.T)] = True
    label = ndi.label(mask)[0]
    w = watershed(image=-distance, markers=label, mask=sm_rmap > thresh)
    label = ndi.label(w)[0]
    return label


def limit_to_one(A, prc=50, min_dist=5):
    """
    Processes a multi-peaked ratemap and returns a matrix
    where the multi-peaked ratemap consist of a single peaked field that is
    a) not connected to the border and b) close to the middle of the
    ratemap

    Parameters
    ----------
    A : np.ndarray
        The ratemap
    prc : int
        The percentage of the peak rate to threshold the ratemap at
    min_dist : int
        The minimum distance between peaks

    Returns
    -------
    tuple
        RegionProperties of the fields (list of RegionProperties)
        The single peaked ratemap (np.ndarray)
        The index of the field (int)

    """
    Ac = A.copy()
    Ac[np.isnan(A)] = 0
    # smooth Ac more to remove local irregularities
    n = ny = 5
    x, y = np.mgrid[-n : n + 1, -ny : ny + 1]
    g = np.exp(-(x**2 / float(n) + y**2 / float(ny)))
    g = g / g.sum()
    Ac = signal.convolve(Ac, g, mode="same")
    # remove really small values
    Ac[Ac < 1e-10] = 0
    Ac_r = skimage.exposure.rescale_intensity(
        Ac, in_range="image", out_range=(0, 1000)
    ).astype(np.int32)
    peak_idx = skimage.feature.peak_local_max(
        Ac_r, min_distance=min_dist, exclude_border=False
    )
    peak_mask = np.zeros_like(Ac, dtype=bool)
    peak_mask[tuple(peak_idx.T)] = True
    peak_labels = skimage.measure.label(peak_mask, connectivity=2)
    field_labels = watershed(image=Ac * -1, markers=peak_labels)
    nFields = np.max(field_labels)
    sub_field_mask = np.zeros((nFields, Ac.shape[0], Ac.shape[1]))
    labelled_sub_field_mask = np.zeros_like(sub_field_mask)
    sub_field_props = skimage.measure.regionprops(field_labels, intensity_image=Ac)
    sub_field_centroids = []
    sub_field_size = []

    for sub_field in sub_field_props:
        tmp = np.zeros(Ac.shape).astype(bool)
        tmp[sub_field.coords[:, 0], sub_field.coords[:, 1]] = True
        tmp2 = Ac > sub_field.max_intensity * (prc / float(100))
        sub_field_mask[sub_field.label - 1, :, :] = np.logical_and(tmp2, tmp)
        labelled_sub_field_mask[sub_field.label - 1, np.logical_and(tmp2, tmp)] = (
            sub_field.label
        )
        sub_field_centroids.append(sub_field.centroid)
        sub_field_size.append(sub_field.area)  # in bins
    sub_field_mask = np.sum(sub_field_mask, 0)
    middle = np.round(np.array(A.shape) / 2)
    normd_dists = sub_field_centroids - middle
    field_dists_from_middle = np.hypot(normd_dists[:, 0], normd_dists[:, 1])
    central_field_idx = np.argmin(field_dists_from_middle)
    central_field = np.squeeze(labelled_sub_field_mask[central_field_idx, :, :])
    # collapse the labelled mask down to an 2d array
    labelled_sub_field_mask = np.sum(labelled_sub_field_mask, 0)
    # clear the border
    cleared_mask = skimage.segmentation.clear_border(central_field)
    # check we've still got stuff in the matrix or fail
    if ~np.any(cleared_mask):
        print("No fields were detected away from edges so nothing returned")
        return None, None, None
    else:
        central_field_props = sub_field_props[central_field_idx]
    return central_field_props, central_field, central_field_idx


def global_threshold(A, prc=50, min_dist=5)->int:
    """
    Globally thresholds a ratemap and counts number of fields found

    Parameters
    ----------
    A : np.ndarray
        The ratemap
    prc : int
        The percentage of the peak rate to threshold the ratemap at
    min_dist : int
        The minimum distance between peaks

    Returns
    -------
    int
        The number of fields found in the ratemap
    """
    Ac = A.copy()
    Ac[np.isnan(A)] = 0
    n = ny = 5
    x, y = np.mgrid[-n : n + 1, -ny : ny + 1]
    g = np.exp(-(x**2 / float(n) + y**2 / float(ny)))
    g = g / g.sum()
    Ac = signal.convolve(Ac, g, mode="same")
    maxRate = np.nanmax(np.ravel(Ac))
    Ac[Ac < maxRate * (prc / float(100))] = 0
    Ac_r = skimage.exposure.rescale_intensity(
        Ac, in_range="image", out_range=(0, 1000)
    ).astype(np.int32)
    peak_idx = skimage.feature.peak_local_max(
        Ac_r, min_distance=min_dist, exclude_border=False
    )
    peak_mask = np.zeros_like(Ac, dtype=bool)
    peak_mask[tuple(peak_idx.T)] = True
    peak_labels = skimage.measure.label(peak_mask, connectivity=2)
    field_labels = watershed(image=Ac * -1, markers=peak_labels)
    nFields = np.max(field_labels)
    return nFields


def local_threshold(A, prc=50, min_dist=5)->np.ndarray:
    """
    Locally thresholds a ratemap to take only the surrounding prc amount
    around any local peak

    Parameters
    ----------
    A : np.ndarray
        The ratemap
    prc : int
        The percentage of the peak rate to threshold the ratemap at
    min_dist : int
        The minimum distance between peaks

    Returns
    -------
    np.ndarray
        The thresholded ratemap

    """`
    Ac = A.copy()
    nanidx = np.isnan(Ac)
    Ac[nanidx] = 0
    # smooth Ac more to remove local irregularities
    n = ny = 5
    x, y = np.mgrid[-n : n + 1, -ny : ny + 1]
    g = np.exp(-(x**2 / float(n) + y**2 / float(ny)))
    g = g / g.sum()
    Ac = signal.convolve(Ac, g, mode="same")
    # rescale the image going in to peak_local_max and cast to
    # integer dtype as there is an invert operation internally
    # that only works in int or bool dtypes
    Ac_r = skimage.exposure.rescale_intensity(
        Ac, in_range="image", out_range=(0, 1000)
    ).astype(np.int32)
    peak_idx = skimage.feature.peak_local_max(
        Ac_r, min_distance=min_dist, exclude_border=False
    )
    peak_mask = np.zeros_like(Ac, dtype=bool)
    peak_mask[tuple(peak_idx.T)] = True
    peak_labels = skimage.measure.label(peak_mask, connectivity=2)
    field_labels = watershed(image=Ac * -1, markers=peak_labels)
    nFields = np.max(field_labels)
    sub_field_mask = np.zeros((nFields, Ac.shape[0], Ac.shape[1]))
    sub_field_props = skimage.measure.regionprops(field_labels, intensity_image=Ac)
    sub_field_centroids = []
    sub_field_size = []

    for sub_field in sub_field_props:
        tmp = np.zeros(Ac.shape).astype(bool)
        tmp[sub_field.coords[:, 0], sub_field.coords[:, 1]] = True
        tmp2 = Ac > sub_field.max_intensity * (prc / float(100))
        sub_field_mask[sub_field.label - 1, :, :] = np.logical_and(tmp2, tmp)
        sub_field_centroids.append(sub_field.centroid)
        sub_field_size.append(sub_field.area)  # in bins
    sub_field_mask = np.sum(sub_field_mask, 0)
    A_out = np.zeros_like(A)
    A_out[sub_field_mask.astype(bool)] = A[sub_field_mask.astype(bool)]
    A_out[nanidx] = np.nan
    return A_out


def border_score(
    A,
    B=None,
    shape="square",
    fieldThresh=0.3,
    circumPrc=0.2,
    binSize=3.0,
    minArea=200,
):
    """

    Calculates a border score totally dis-similar to that calculated in
    Solstad et al (2008)

    Parameters
    ----------
    A : np.ndarray
        the ratemap
    B : np.ndarray, default None
        This should be a boolean mask where True (1)
        is equivalent to the presence of a border and False (0)
        is equivalent to 'open space'. Naievely this will be the
        edges of the ratemap but could be used to take account of
        boundary insertions/ creations to check tuning to multiple
        environmental boundaries. Default None: when the mask is
        None then a mask is created that has 1's at the edges of the
        ratemap i.e. it is assumed that occupancy = environmental
        shape
    shape : str, default 'square'
        description of environment shape. Currently
        only 'square' or 'circle' accepted. Used to calculate the
        proportion of the environmental boundaries to examine for
        firing
    fieldThresh : float, default 0.3
        Between 0 and 1 this is the percentage
        amount of the maximum firing rate
        to remove from the ratemap (i.e. to remove noise)
    circumPrc : float, default 0.2
        The percentage amount of the circumference
        of the environment that the field needs to be to count
        as long enough to make it through
    binSize : float, default 3.0
        bin size in cm
    minArea : float, default 200
        min area for a field to be considered

    Returns
    -------
    float
        the border score

    Notes
    -----
    If the cell is a border cell (BVC) then we know that it should
    fire at a fixed distance from a given boundary (possibly more
    than one). In essence this algorithm estimates the amount of
    variance in this distance i.e. if the cell is a border cell this
    number should be small. This is achieved by first doing a bunch of
    morphological operations to isolate individual fields in the
    ratemap (similar to the code used in phasePrecession.py - see
    the partitionFields method therein). These partitioned fields are then
    thinned out (using skimage's skeletonize) to a single pixel
    wide field which will lie more or less in the middle of the
    (highly smoothed) sub-field. It is the variance in distance from the
    nearest boundary along this pseudo-iso-line that is the boundary
    measure

    Other things to note are that the pixel-wide field has to have some
    minimum length. In the case of a circular environment this is set to
    20% of the circumference; in the case of a square environment markers
    this is at least half the length of the longest side
    """
    # need to know borders of the environment so we can see if a field
    # touches the edges, and the perimeter length of the environment
    # deal with square or circles differently
    borderMask = np.zeros_like(A)
    A_rows, A_cols = np.shape(A)
    if "circle" in shape:
        radius = np.max(np.array(np.shape(A))) / 2.0
        dist_mask = skimage.morphology.disk(radius)
        if np.shape(dist_mask) > np.shape(A):
            dist_mask = dist_mask[1 : A_rows + 1, 1 : A_cols + 1]
        tmp = np.zeros([A_rows + 2, A_cols + 2])
        tmp[1:-1, 1:-1] = dist_mask
        dists = ndi.distance_transform_bf(tmp)
        dists = dists[1:-1, 1:-1]
        borderMask = np.logical_xor(dists <= 0, dists < 2)
        # open up the border mask a little
        borderMask = skimage.morphology.binary_dilation(
            borderMask, skimage.morphology.disk(1)
        )
    elif "square" in shape:
        borderMask[0:3, :] = 1
        borderMask[-3:, :] = 1
        borderMask[:, 0:3] = 1
        borderMask[:, -3:] = 1
        tmp = np.zeros([A_rows + 2, A_cols + 2])
        dist_mask = np.ones_like(A)
        tmp[1:-1, 1:-1] = dist_mask
        dists = ndi.distance_transform_bf(tmp)
        # remove edges to make same shape as input ratemap
        dists = dists[1:-1, 1:-1]
    A[~np.isfinite(A)] = 0
    # get some morphological info about the fields in the ratemap
    # start image processing:
    # get some markers
    # NB I've tried a variety of techniques to optimise this part and the
    # best seems to be the local adaptive thresholding technique which)
    # smooths locally with a gaussian - see the skimage docs for more
    idx = A >= np.nanmax(np.ravel(A)) * fieldThresh
    A_thresh = np.zeros_like(A)
    A_thresh[idx] = A[idx]

    # label these markers so each blob has a unique id
    labels, nFields = ndi.label(A_thresh)
    # remove small objects
    min_size = int(minArea / binSize) - 1
    skimage.morphology.remove_small_objects(labels, min_size=min_size, connectivity=2)
    labels = skimage.segmentation.relabel_sequential(labels)[0]
    nFields = np.nanmax(labels)
    if nFields == 0:
        return np.nan

    # Iterate over the labelled parts of the array labels calculating
    # how much of the total circumference of the environment edge it
    # covers

    fieldAngularCoverage = np.zeros([1, nFields]) * np.nan
    fractionOfPixelsOnBorder = np.zeros([1, nFields]) * np.nan
    fieldsToKeep = np.zeros_like(A).astype(bool)
    for i in range(1, nFields + 1):
        fieldMask = np.logical_and(labels == i, borderMask)

        # check the angle subtended by the fieldMask
        if np.nansum(fieldMask.astype(int)) > 0:
            s = skimage.measure.regionprops(
                fieldMask.astype(int), intensity_image=A_thresh
            )[0]
            x = s.coords[:, 0] - (A_cols / 2.0)
            y = s.coords[:, 1] - (A_rows / 2.0)
            subtended_angle = np.rad2deg(np.ptp(np.arctan2(x, y)))
            if subtended_angle > (360 * circumPrc):
                pixelsOnBorder = np.count_nonzero(fieldMask) / float(
                    np.count_nonzero(labels == i)
                )
                fractionOfPixelsOnBorder[:, i - 1] = pixelsOnBorder
                if pixelsOnBorder > 0.5:
                    fieldAngularCoverage[0, i - 1] = subtended_angle

            fieldsToKeep = np.logical_or(fieldsToKeep, labels == i)

    # Check the fields are big enough to qualify (minArea)
    # returning nan if not
    def fn(val):
        return np.count_nonzero(val)

    field_sizes = ndi.labeled_comprehension(
        A, labels, range(1, nFields + 1), fn, float, 0
    )
    field_sizes /= binSize
    if not np.any(field_sizes) > (minArea / binSize):
        warnings.warn(
            f"No fields bigger than the minimum size of {minArea/binSize} (minArea/binSize) could be found"
        )
        return np.nan

    fieldAngularCoverage = fieldAngularCoverage / 360.0
    rateInField = A[fieldsToKeep]
    # normalize firing rate in the field to sum to 1
    rateInField = rateInField / np.nansum(rateInField)
    dist2WallInField = dists[fieldsToKeep]
    Dm = np.dot(dist2WallInField, rateInField)
    if "circle" in shape:
        Dm = Dm / radius
    elif "square" in shape:
        Dm = Dm / (np.nanmax(np.shape(A)) / 2.0)
    borderScore = (fractionOfPixelsOnBorder - Dm) / (fractionOfPixelsOnBorder + Dm)
    return np.nanmax(borderScore)


def _get_field_labels(A: np.ndarray, **kwargs) -> tuple:
    """
    Returns a labeled version of A after finding the peaks
    in A and finding the watershed basins from the markers
    found from those peaks. Used in field_props() and
    grid_field_props()

    Parameters
    ----------
    A : np.ndarray
        The array to process
    **kwargs
        min_distance (float, optional): The distance in bins between fields to
        separate the regions of the image
        clear_border (bool, optional): Input to skimage.feature.peak_local_max.
        The number of pixels to ignore at the edge of the image
    """
    clear_border = True
    if "clear_border" in kwargs:
        clear_border = kwargs.pop("clear_border")

    min_distance = 1
    if "min_distance" in kwargs:
        min_distance = kwargs.pop("min_distance")

    A[~np.isfinite(A)] = -1
    A[A < 0] = -1
    Ac_r = skimage.exposure.rescale_intensity(
        A, in_range="image", out_range=(0, 1000)
    ).astype(np.int32)
    peak_coords = skimage.feature.peak_local_max(
        Ac_r, min_distance=min_distance, exclude_border=clear_border
    )
    peaksMask = np.zeros_like(A, dtype=bool)
    peaksMask[tuple(peak_coords.T)] = True
    peaksLabel, _ = ndi.label(peaksMask)
    ws = watershed(image=-1 * A, markers=peaksLabel)
    return peak_coords, ws


def field_props(
    A,
    min_dist=5,
    neighbours=2,
    prc=50,
    plot=False,
    ax=None,
    tri=False,
    verbose=True,
    **kwargs,
):
    """
    Returns a dictionary of properties of the field(s) in a ratemap A

    Args:
        A (array_like): a ratemap (but could be any image)
        min_dist (float): the separation (in bins) between fields for measures
            such as field distance to make sense. Used to
            partition the image into separate fields in the call to
            feature.peak_local_max
        neighbours (int): the number of fields to consider as neighbours to
            any given field. Defaults to 2
        prc (float): percent of fields to consider
        ax (matplotlib.Axes): user supplied axis. If None a new figure window
        is created
        tri (bool): whether to do Delaunay triangulation between fields
            and add to plot
        verbose (bool): dumps the properties to the console
        plot (bool): whether to plot some output - currently consists of the
            ratemap A, the fields of which are outline in a black
            contour. Default False

    Returns:
        result (dict): The properties of the field(s) in the input ratemap A
    """

    from skimage.measure import find_contours
    from sklearn.neighbors import NearestNeighbors

    nan_idx = np.isnan(A)
    Ac = A.copy()
    Ac[np.isnan(A)] = 0
    # smooth Ac more to remove local irregularities
    n = ny = 5
    x, y = np.mgrid[-n : n + 1, -ny : ny + 1]
    g = np.exp(-(x**2 / float(n) + y**2 / float(ny)))
    g = g / g.sum()
    Ac = signal.convolve(Ac, g, mode="same")

    peak_idx, field_labels = _get_field_labels(Ac, **kwargs)

    nFields = np.max(field_labels)
    if neighbours > nFields:
        print(
            "neighbours value of {0} > the {1} peaks found".format(neighbours, nFields)
        )
        print("Reducing neighbours to number of peaks found")
        neighbours = nFields
    sub_field_mask = np.zeros((nFields, Ac.shape[0], Ac.shape[1]))
    sub_field_props = skimage.measure.regionprops(field_labels, intensity_image=Ac)
    sub_field_centroids = []
    sub_field_size = []

    for sub_field in sub_field_props:
        tmp = np.zeros(Ac.shape).astype(bool)
        tmp[sub_field.coords[:, 0], sub_field.coords[:, 1]] = True
        tmp2 = Ac > sub_field.max_intensity * (prc / float(100))
        sub_field_mask[sub_field.label - 1, :, :] = np.logical_and(tmp2, tmp)
        sub_field_centroids.append(sub_field.centroid)
        sub_field_size.append(sub_field.area)  # in bins
    sub_field_mask = np.sum(sub_field_mask, 0)
    contours = skimage.measure.find_contours(sub_field_mask, 0.5)
    # find the nearest neighbors to the peaks of each sub-field
    nbrs = NearestNeighbors(n_neighbors=neighbours, algorithm="ball_tree").fit(peak_idx)
    distances, _ = nbrs.kneighbors(peak_idx)
    mean_field_distance = np.mean(distances[:, 1:neighbours])

    nValid_bins = np.sum(~nan_idx)
    # calculate the amount of out of field firing
    A_non_field = np.zeros_like(A) * np.nan
    A_non_field[~sub_field_mask.astype(bool)] = A[~sub_field_mask.astype(bool)]
    A_non_field[nan_idx] = np.nan
    out_of_field_firing_prc = (
        np.count_nonzero(A_non_field > 0) / float(nValid_bins)
    ) * 100
    Ac[np.isnan(A)] = np.nan
    # get some stats about the field ellipticity
    ellipse_ratio = np.nan
    _, central_field, _ = limit_to_one(A, prc=50)

    contour_coords = find_contours(central_field, 0.5)
    from skimage.measure import EllipseModel

    E = EllipseModel()
    E.estimate(contour_coords[0])
    ellipse_axes = E.params[2:4]
    ellipse_ratio = np.min(ellipse_axes) / np.max(ellipse_axes)

    """ using the peak_idx values calculate the angles of the triangles that
    make up a delaunay tesselation of the space if the calc_angles arg is
    in kwargs
    """
    if "calc_angs" in kwargs.keys():
        angs = calc_angs(peak_idx)
    else:
        angs = None

    props = {
        "Ac": Ac,
        "Peak_rate": np.nanmax(A),
        "Mean_rate": np.nanmean(A),
        "Field_size": np.mean(sub_field_size),
        "Pct_bins_with_firing": (np.sum(sub_field_mask) / nValid_bins) * 100,
        "Out_of_field_firing_prc": out_of_field_firing_prc,
        "Dist_between_fields": mean_field_distance,
        "Num_fields": float(nFields),
        "Sub_field_mask": sub_field_mask,
        "Smoothed_map": Ac,
        "field_labels": field_labels,
        "Peak_idx": peak_idx,
        "angles": angs,
        "contours": contours,
        "ellipse_ratio": ellipse_ratio,
    }

    if verbose:
        print(
            "\nPercentage of bins with firing: {:.2%}".format(
                np.sum(sub_field_mask) / nValid_bins
            )
        )
        print(
            "Percentage out of field firing: {:.2%}".format(
                np.count_nonzero(A_non_field > 0) / float(nValid_bins)
            )
        )
        print(f"Peak firing rate: {np.nanmax(A)} Hz")
        print(f"Mean firing rate: {np.nanmean(A)} Hz")
        print(f"Number of fields: {nFields}")
        print(f"Mean field size: {np.mean(sub_field_size)} cm")
        print(f"Mean inter-peak distance between fields: {mean_field_distance} cm")
    return props


def calc_angs(points):
    """
    Calculates the angles for all triangles in a delaunay tesselation of
    the peak points in the ratemap
    """

    # calculate the lengths of the sides of the triangles
    tri = spatial.Delaunay(points)
    angs = []
    for s in tri.simplices:
        A = tri.points[s[1]] - tri.points[s[0]]
        B = tri.points[s[2]] - tri.points[s[1]]
        C = tri.points[s[0]] - tri.points[s[2]]
        for e1, e2 in ((A, -B), (B, -C), (C, -A)):
            num = np.dot(e1, e2)
            denom = np.linalg.norm(e1) * np.linalg.norm(e2)
            angs.append(np.arccos(num / denom) * 180 / np.pi)
    return np.array(angs).T


def kl_spatial_sparsity(pos_map: BinnedData) -> float:
    """
    Calculates the spatial sampling of an arena by comparing the
    observed spatial sampling to an expected uniform one using kl divergence

    Data in pos_map should be unsmoothed (not checked) and the MapType should
    be POS (checked)

    Parameters
    ----------
    pos_map : BinnedData
        The position map

    Returns
    -------
    float
        The spatial sparsity of the position map
    """
    assert pos_map.map_type == MapType.POS
    return kldiv_dir(np.ravel(pos_map.binned_data[0]))


def spatial_sparsity(rate_map: np.ndarray, pos_map: np.ndarray) -> float:
    """
    Calculates the spatial sparsity of a rate map as defined by
    Markus et al (1994)

    For example, a sparsity score of 0.10 indicates that the cell fired on
    10% of the maze surface

    Parameters
    ----------
    rate_map : np.ndarray
        The rate map
    pos_map : np.ndarray
        The occupancy map

    Returns
    -------
    float
        The spatial sparsity of the rate map

    References
    ----------
    Markus, E.J., Barnes, C.A., McNaughton, B.L., Gladden, V.L. &
    Skaggs, W.E. Spatial information content and reliability of
    hippocampal CA1 neurons: effects of visual input. Hippocampus
    4, 410–421 (1994).

    """
    p_i = pos_map / np.nansum(pos_map)
    sparsity = np.nansum(p_i * rate_map) ** 2 / np.nansum(p_i * rate_map**2)
    return sparsity


def coherence(smthd_rate, unsmthd_rate):
    """
    Calculates the coherence of receptive field via correlation of smoothed
    and unsmoothed ratemaps

    Parameters
    ----------
    smthd_rate : np.ndarray
        The smoothed rate map
    unsmthd_rate : np.ndarray
        The unsmoothed rate map

    Returns
    -------
    float
        The coherence of the rate maps
    """
    smthd = smthd_rate.ravel()
    unsmthd = unsmthd_rate.ravel()
    si = ~np.isnan(smthd)
    ui = ~np.isnan(unsmthd)
    idx = ~(~si | ~ui)
    coherence = np.corrcoef(unsmthd[idx], smthd[idx])
    return coherence[1, 0]


def kldiv_dir(polarPlot: np.ndarray) -> float:
    """
    Returns a kl divergence for directional firing: measure of directionality.
    Calculates kl diveregence between a smoothed ratemap (probably should be
    smoothed otherwise information theoretic measures
    don't 'care' about position of bins relative to one another) and a
    pure circular distribution.
    The larger the divergence the more tendancy the cell has to fire when the
    animal faces a specific direction.

    Parameters
    ----------
    polarPlot np.ndarray
        The binned and smoothed directional ratemap

    Returns
    -------
    float
        The divergence from circular of the 1D-array
        from a uniform circular distribution
    """

    __inc = 0.00001
    polarPlot = np.atleast_2d(polarPlot)
    polarPlot[np.isnan(polarPlot)] = __inc
    polarPlot[polarPlot == 0] = __inc
    normdPolar = polarPlot / float(np.nansum(polarPlot))
    nDirBins = polarPlot.shape[1]
    compCirc = np.ones_like(polarPlot) / float(nDirBins)
    X = np.arange(0, nDirBins)
    kldivergence = kldiv(np.atleast_2d(X), normdPolar, compCirc)
    return kldivergence


def kldiv(
    X: np.ndarray, pvect1: np.ndarray, pvect2: np.ndarray, variant: str = ""
) -> float:
    """
    Calculates the Kullback-Leibler or Jensen-Shannon divergence between
    two distributions.

    Parameters
    ----------
    X : np.ndarray
        Vector of M variable values
    P1, P2 : np.ndarray
        Length-M vectors of probabilities representing distribution 1 and 2
    variant : str, default 'sym'
        If 'sym', returns a symmetric variant of the
        Kullback-Leibler divergence, given by [KL(P1,P2)+KL(P2,P1)]/2
        If 'js', returns the Jensen-Shannon divergence, given by
        [KL(P1,Q)+KL(P2,Q)]/2, where Q = (P1+P2)/2

    Returns
    -------
    float
        The Kullback-Leibler divergence or Jensen-Shannon divergence

    Notes
    -----
    The Kullback-Leibler divergence is given by:

    .. math:: KL(P1(x),P2(x)) = sum_[P1(x).log(P1(x)/P2(x))]

    If X contains duplicate values, there will be an warning message,
    and these values will be treated as distinct values.  (I.e., the
    actual values do not enter into the computation, but the probabilities
    for the two duplicate values will be considered as probabilities
    corresponding to two unique values.).
    The elements of probability vectors P1 and P2 must
    each sum to 1 +/- .00001.

    This function is taken from one on the Mathworks file exchange

    See Also
    --------
    Cover, T.M. and J.A. Thomas. "Elements of Information Theory," Wiley,
    1991.

    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    """

    if len(np.unique(X)) != len(np.sort(X)):
        warnings.warn(
            "X contains duplicate values. Treated as distinct values.", UserWarning
        )
    if (
        not np.equal(np.shape(X), np.shape(pvect1)).all()
        or not np.equal(np.shape(X), np.shape(pvect2)).all()
    ):
        raise ValueError("Inputs are not the same size")
    if (np.abs(np.sum(pvect1) - 1) > 0.00001) or (np.abs(np.sum(pvect2) - 1) > 0.00001):
        print(f"Probabilities sum to {np.abs(np.sum(pvect1))} for pvect1")
        print(f"Probabilities sum to {np.abs(np.sum(pvect2))} for pvect2")
        warnings.warn("Probabilities don" "t sum to 1.", UserWarning)
    if variant:
        if variant == "js":
            logqvect = np.log2((pvect2 + pvect1) / 2)
            KL = 0.5 * (
                np.nansum(pvect1 * (np.log2(pvect1) - logqvect))
                + np.sum(pvect2 * (np.log2(pvect2) - logqvect))
            )
            return float(KL)
        elif variant == "sym":
            KL1 = np.nansum(pvect1 * (np.log2(pvect1) - np.log2(pvect2)))
            KL2 = np.nansum(pvect2 * (np.log2(pvect2) - np.log2(pvect1)))
            KL = (KL1 + KL2) / 2
            return float(KL)
        else:
            warnings.warn("Last argument not recognised", UserWarning)
    KL = np.nansum(pvect1 * (np.log2(pvect1) - np.log2(pvect2)))
    return float(KL)


def skaggs_info(ratemap, dwelltimes, **kwargs):
    """
    Calculates Skaggs information measure

    Parameters
    ----------
    ratemap, dwelltimes :np.ndarray
        The binned up ratemap and dwelltimes. Must be the same size

    Returns
    -------
    float
        Skaggs information score in bits spike

    Notes
    -----
    The ratemap data should have undergone adaptive binning as per
    the original paper. See getAdaptiveMap() in binning class

    The estimate of spatial information in bits per spike:

    .. math:: I = sum_{x} p(x).r(x).log(r(x)/r)
    """
    sample_rate = kwargs.get("sample_rate", 50)

    dwelltimes = dwelltimes / sample_rate  # assumed sample rate of 50Hz
    if ratemap.ndim > 1:
        ratemap = np.reshape(ratemap, (np.prod(np.shape(ratemap)), 1))
        dwelltimes = np.reshape(dwelltimes, (np.prod(np.shape(dwelltimes)), 1))
    duration = np.nansum(dwelltimes)
    meanrate = np.nansum(ratemap * dwelltimes) / duration
    if meanrate <= 0.0:
        bits_per_spike = np.nan
        return bits_per_spike
    p_x = dwelltimes / duration
    p_r = ratemap / meanrate
    dum = p_x * ratemap
    ind = np.nonzero(dum)[0]
    bits_per_spike = np.nansum(dum[ind] * np.log2(p_r[ind]))
    bits_per_spike = bits_per_spike / meanrate
    return bits_per_spike


def grid_field_props(A: BinnedData, maxima="centroid", allProps=True, **kwargs):
    """
    Extracts various measures from a spatial autocorrelogram

    Parameters
    ----------
    A : BinnedData
        object containing the spatial autocorrelogram (SAC) in
            A.binned_data[0]
    maxima (str, optional): The method used to detect the peaks in the SAC.
            Legal values are 'single' and 'centroid'. Default 'centroid'
    allProps : bool default=True
        Whether to return a dictionary that contains the attempt to fit 
        an ellipse around the edges of the central size peaks. See below

    Returns
    -------
    dict
        Measures of the SAC.
        Keys include:
            * gridness score
            * scale
            * orientation
            * coordinates of the peaks (nominally 6) closest to SAC centre
            * a binary mask around the extent of the 6 central fields
            * values of the rotation procedure used to calculate gridness
            * ellipse axes and angle (if allProps is True and the it worked)

    Notes
    -----
    The output from this method can be used as input to the show() method
    of this class.
    When it is the plot produced will display a lot more informative.
    The coordinate system internally used is centred on the image centre.

    See Also
    --------
    ephysiopy.common.binning.autoCorr2D()
    """
    """
    Assign the output dictionary now as we want to return immediately if
    the input is bad
    """
    dictKeys = (
        "gridscore",
        "scale",
        "orientation",
        "closest_peak_coords",
        "dist_to_centre",
        "ellipse_axes",
        "ellipse_angle",
        "ellipseXY",
        "circleXY",
        "rotationArr",
        "rotationCorrVals",
    )

    outDict = dict.fromkeys(dictKeys, np.nan)

    A_tmp = A.binned_data[0].copy()

    if np.all(np.isnan(A_tmp)):
        warnings.warn("No data in SAC - returning nans in measures dict")
        outDict["dist_to_centre"] = np.atleast_2d(np.array([0, 0]))
        outDict["scale"] = 0
        outDict["closest_peak_coords"] = np.atleast_2d(np.array([0, 0]))
        return outDict

    A_tmp[~np.isfinite(A_tmp)] = -1
    A_tmp[A_tmp <= 0] = -1
    A_sz = np.array(np.shape(A_tmp))
    # [STAGE 1] find peaks & identify 7 closest to centre
    min_distance = np.ceil(np.min(A_sz / 2) / 8.0).astype(int)
    min_distance = kwargs.get("min_distance", min_distance)

    _, _, field_labels, _ = partitionFields(
        A, field_threshold_percent=10, field_rate_threshold=0.001
    )
    # peak_idx, field_labels = _get_field_labels(A_tmp, neighbours=7, **kwargs)
    # a fcn for the labeled_comprehension function that returns
    # linear indices in A where the values in A for each label are
    # greater than half the max in that labeled region

    def fn(val, pos):
        return pos[val > (np.max(val) / 2)]

    nLbls = np.max(field_labels)
    indices = ndi.labeled_comprehension(
        A_tmp, field_labels, np.arange(0, nLbls), fn, np.ndarray, 0, True
    )
    # turn linear indices into coordinates
    coords = [np.unravel_index(i, A_sz) for i in indices]
    half_peak_labels = np.zeros(shape=A_sz)
    for peak_id, coord in enumerate(coords):
        xc, yc = coord
        half_peak_labels[xc, yc] = peak_id

    # Get some statistics about the labeled regions
    lbl_range = np.arange(0, nLbls)
    peak_coords = ndi.maximum_position(A.binned_data[0], half_peak_labels, lbl_range)
    peak_coords = np.array(peak_coords)
    # Now convert the peak_coords to the image centre coordinate system
    x_peaks, y_peaks = peak_coords.T
    x_peaks_ij = A.bin_edges[0][x_peaks]
    y_peaks_ij = A.bin_edges[1][y_peaks]
    peak_coords = np.array([x_peaks_ij, y_peaks_ij]).T
    # Get some distance and morphology measures
    peak_dist_to_centre = np.hypot(peak_coords[:, 0], peak_coords[:, 1])
    closest_peak_idx = np.argsort(peak_dist_to_centre)
    central_peak_label = closest_peak_idx[0]
    closest_peak_idx = closest_peak_idx[1 : np.min((7, len(closest_peak_idx) - 1))]
    # closest_peak_idx should now the indices of the labeled 6 peaks
    # surrounding the central peak at the image centre
    scale = np.median(peak_dist_to_centre[closest_peak_idx])
    orientation = np.nan
    orientation = grid_orientation(peak_coords, closest_peak_idx)

    xv, yv = np.meshgrid(A.bin_edges[0], A.bin_edges[1], indexing="ij")
    xv = xv[:-1, :-1]  # remove last row and column
    yv = yv[:-1:, :-1]  # remove last row and column
    dist_to_centre = np.hypot(xv, yv)
    # get the max distance of the half-peak width labeled fields
    # from the centre of the image
    max_dist_from_centre = 0
    for peak_id, _coords in enumerate(coords):
        if peak_id in closest_peak_idx:
            xc, yc = _coords
            if np.any(xc) and np.any(yc):
                xc = A.bin_edges[0][xc]
                yc = A.bin_edges[1][yc]
                d = np.max(np.hypot(xc, yc))
                if d > max_dist_from_centre:
                    max_dist_from_centre = d

    # Set the outer bits and the central region of the SAC to nans
    # getting ready for the correlation procedure
    dist_to_centre[np.abs(dist_to_centre) > max_dist_from_centre] = 0
    dist_to_centre[half_peak_labels == central_peak_label] = 0
    dist_to_centre[dist_to_centre != 0] = 1
    dist_to_centre = dist_to_centre.astype(bool)
    sac_middle = A.binned_data[0].copy()
    sac_middle[~dist_to_centre] = np.nan

    if "step" in kwargs.keys():
        step = kwargs.pop("step")
    else:
        step = 30
    try:
        gridscore, rotationCorrVals, rotationArr = gridness(sac_middle, step=step)
    except Exception:
        gridscore, rotationCorrVals, rotationArr = np.nan, np.nan, np.nan

    if allProps:
        # attempt to fit an ellipse around the outer edges of the nearest
        # peaks to the centre of the SAC. First find the outer edges for
        # the closest peaks using a ndimages labeled_comprehension
        try:

            def fn2(val, pos):
                xc, yc = np.unravel_index(pos, A_sz)
                xc = xc - np.floor(A_sz[0] / 2)
                yc = yc - np.floor(A_sz[1] / 2)
                idx = np.argmax(np.hypot(xc, yc))
                return xc[idx], yc[idx]

            ellipse_coords = ndi.labeled_comprehension(
                A.binned_data[0],
                half_peak_labels,
                closest_peak_idx,
                fn2,
                tuple,
                0,
                True,
            )

            ellipse_fit_coords = np.array([(x, y) for x, y in ellipse_coords])
            from skimage.measure import EllipseModel

            E = EllipseModel()
            E.estimate(ellipse_fit_coords)
            im_centre = E.params[0:2]
            ellipse_axes = E.params[2:4]
            ellipse_angle = E.params[-1]
            ellipseXY = E.predict_xy(np.linspace(0, 2 * np.pi, 50), E.params)

            # get the min containing circle given the eliipse minor axis
            from skimage.measure import CircleModel

            _params = [im_centre, np.min(ellipse_axes)]
            circleXY = CircleModel().predict_xy(
                np.linspace(0, 2 * np.pi, 50), params=_params
            )
        except (TypeError, ValueError):  # non-iterable x and y
            ellipse_axes = None
            ellipse_angle = (None, None)
            ellipseXY = None
            circleXY = None

    # collect all the following keywords into a dict for output
    closest_peak_coords = np.array(peak_coords)[closest_peak_idx]

    # Assign values to the output dictionary created at the start
    for thiskey in outDict.keys():
        outDict[thiskey] = locals()[thiskey]
        # neat trick: locals is a dict holding all locally scoped variables
    return outDict


def grid_orientation(peakCoords, closestPeakIdx):
    """
    Calculates the orientation angle of a grid field.

    The orientation angle is the angle of the first peak working
    counter-clockwise from 3 o'clock

    Parameters
    ----------
    peakCoords : np.ndarray
        The peak coordinates as pairs of xy
    closestPeakIdx : np.ndarray
        A 1D array of the indices in peakCoords
        of the peaks closest to the centre of the SAC

    Returns
    -------
    float
        The first value in an array of the angles of
        the peaks in the SAC working counter-clockwise from a line
        extending from the middle of the SAC to 3 o'clock.
    """
    if len(peakCoords) < 3 or closestPeakIdx.size == 0:
        return np.nan
    else:
        from ephysiopy.common.utils import polar

        peaks = peakCoords[closestPeakIdx]
        theta = polar(peaks[:, 1], -peaks[:, 0], deg=1)[1]
        return np.sort(theta.compress(theta >= 0))[0]


def gridness(image, step=30)->tuple:
    """
    Calculates the gridness score in a grid cell SAC.

    The data in `image` is rotated in `step` amounts and
    each rotated array is correlated with the original.
    The maximum of the values at 30, 90 and 150 degrees
    is the subtracted from the minimum of the values at 60, 120
    and 180 degrees to give the grid score.

    Parameters
    ----------
    image : np.ndarray
        The spatial autocorrelogram
    step : int, default=30
        The amount to rotate the SAC in each step of the
        rotational correlation procedure

    Returns
    -------
    3-tuple
        The gridscore, the correlation values at each
        `step` and the rotational array

    Notes
    -----
    The correlation performed is a Pearsons R. Some rescaling of the
    values in `image` is performed following rotation.

    See Also
    --------
    skimage.transform.rotate : for how the rotation of `image` is done
    skimage.exposure.rescale_intensity : for the resscaling following
    rotation
    """
    # TODO: add options in here for whether the full range of correlations
    # are wanted or whether a reduced set is wanted (i.e. at the 30-tuples)
    from collections import OrderedDict

    rotationalCorrVals = OrderedDict.fromkeys(np.arange(0, 181, step), np.nan)
    rotationArr = np.zeros(len(rotationalCorrVals)) * np.nan
    # autoCorrMiddle needs to be rescaled or the image rotation falls down
    # as values are cropped to lie between 0 and 1.0
    in_range = (np.nanmin(image), np.nanmax(image))
    out_range = (0, 1)
    import skimage

    autoCorrMiddleRescaled = skimage.exposure.rescale_intensity(
        image, in_range=in_range, out_range=out_range
    )
    origNanIdx = np.isnan(autoCorrMiddleRescaled.ravel())
    gridscore = np.nan
    try:
        for idx, angle in enumerate(rotationalCorrVals.keys()):
            rotatedA = skimage.transform.rotate(
                autoCorrMiddleRescaled, angle=angle, cval=np.nan, order=3
            )
            # ignore nans
            rotatedNanIdx = np.isnan(rotatedA.ravel())
            allNans = np.logical_or(origNanIdx, rotatedNanIdx)
            # get the correlation between the original and rotated images
            rotationalCorrVals[angle] = stats.pearsonr(
                autoCorrMiddleRescaled.ravel()[~allNans], rotatedA.ravel()[~allNans]
            )[0]
            rotationArr[idx] = rotationalCorrVals[angle]
    except Exception:
        return gridscore, rotationalCorrVals, rotationArr
    gridscore = np.min((rotationalCorrVals[60], rotationalCorrVals[120])) - np.max(
        (rotationalCorrVals[150], rotationalCorrVals[30], rotationalCorrVals[90])
    )
    return gridscore, rotationalCorrVals, rotationArr


def deform_SAC(A, circleXY=None, ellipseXY=None):
    """
    Deforms an elliptical SAC to be circular

    Parameters
    ----------
    A : np.ndarray
        The SAC
    circleXY : np.ndarray, default=None
        The xy coordinates defining a circle.
    ellipseXY : np.ndarray, default=None
        The xy coordinates defining an ellipse.

    Returns
    -------
    np.ndarray
        The SAC deformed to be more circular

    See Also
    --------
    ephysiopy.common.ephys_generic.FieldCalcs.grid_field_props
    skimage.transform.AffineTransform
    skimage.transform.warp
    skimage.exposure.rescale_intensity
    """
    if circleXY is None or ellipseXY is None:
        SAC_stats = grid_field_props(A)
        circleXY = SAC_stats["circleXY"]
        ellipseXY = SAC_stats["ellipseXY"]
        # The ellipse detection stuff might have failed, if so
        # return the original SAC
        if circleXY is None:
            warnings.warn("Ellipse detection failed. Returning original SAC")
            return A

    tform = skimage.transform.AffineTransform()
    tform.estimate(ellipseXY, circleXY)

    """
    the transformation algorithms used here crop values < 0 to 0. Need to
    rescale the SAC values before doing the deformation and then rescale
    again so the values assume the same range as in the unadulterated SAC
    """
    A[np.isnan(A)] = 0
    SACmin = np.nanmin(A.flatten())
    SACmax = np.nanmax(A.flatten())  # should be 1 if autocorr
    AA = A + 1
    deformedSAC = skimage.transform.warp(
        AA / np.nanmax(AA.flatten()), inverse_map=tform.inverse, cval=0
    )
    return skimage.exposure.rescale_intensity(deformedSAC, out_range=(SACmin, SACmax))


def get_circular_regions(A: np.ndarray, **kwargs) -> list:
    """
    Returns a list of images which are expanding circular
    regions centred on the middle of the image out to the
    image edge. Used for calculating the grid score of each
    image to find the one with the max grid score. 

    Parameters
    ----------
    A : np.ndarray
        The SAC

    **kwargs
        min_radius (int): The smallest radius circle to start with

    Returns
    -------
    list
        A list of images which are circular sub-regions of the
        original SAC
    """
    from skimage.measure import CircleModel, grid_points_in_poly


    min_radius = kwargs.get("min_radius", 5)

    centre = tuple([d // 2 for d in np.shape(A)])
    max_radius = min(tuple(np.subtract(np.shape(A), centre)))
    t = np.linspace(0, 2 * np.pi, 51)
    circle = CircleModel()

    result = []
    for radius in range(min_radius, max_radius):
        circle.params = [*centre, radius]
        xy = circle.predict_xy(t)
        mask = grid_points_in_poly(np.shape(A), xy)
        im = A.copy()
        im[~mask] = np.nan
        result.append(im)
    return result


def get_basic_gridscore(A: np.ndarray, **kwargs)->float:
    '''
    Calculates the grid score of a spatial autocorrelogram

    Parameters
    ----------
    A : np.ndarray
        The spatial autocorrelogram

    Returns
    -------
    float
        The grid score of the SAC

    '''
    return gridness(A, **kwargs)[0]


def get_expanding_circle_gridscore(A: np.ndarray, **kwargs):
    """
    Calculates the gridscore for each circular sub-region of image A
    where the circles are centred on the image centre and expanded to
    the edge of the image. The maximum of the get_basic_gridscore() for
    each of these circular sub-regions is returned as the gridscore

    Parameters
    ----------
    A : np.ndarray
        The SAC

    Returns
    -------
    float
        The maximum grid score of the circular sub
        regions of the SAC
    """

    images = get_circular_regions(A, **kwargs)
    gridscores = [gridness(im)[0] for im in images]
    return max(gridscores)


def get_deformed_sac_gridscore(A: np.ndarray)->float:
    """
    Deforms a non-circular SAC into a circular SAC (circular meaning
    the ellipse drawn around the edges of the 6 nearest peaks to the
    SAC centre) and returns get_basic_griscore() calculated on the
    deformed (or re-formed?!) SAC

    Parameters
    ----------
    A : np.ndarray
        The SAC

    Returns
    -------
    float
        The gridscore of the deformed SAC
    """
    deformed_SAC = deform_SAC(A)
    return gridness(deformed_SAC)[0]


def get_thigmotaxis_score(xy: np.ndarray, shape: str = "circle") -> float:
    """
    Returns a score which is the ratio of the time spent in the inner
    portion of an environment to the time spent in the outer portion.
    The portions are allocated so that they have equal area.

    Parameters
    ----------
    xy : np.ndarray
        The xy coordinates of the animal's position. 2 x nsamples
    shape :str, default='circle'
        The shape of the environment. Legal values are 'circle' and 'square'

    Returns
    -------
    float
        Values closer to 1 mean more time was spent in the inner portion of the environment.
        Values closer to -1 mean more time in the outer portion of the environment.
        A value of 0 indicates the animal spent equal time in both portions of the
        environment.
    """
    # centre the coords to get the max distance from the centre
    xc, yc = np.min(xy, -1) + np.ptp(xy, -1) / 2
    xy = xy - np.array([[xc], [yc]])
    n_pos = np.shape(xy)[1]
    inner_mask = np.zeros((n_pos), dtype=bool)
    if shape == "circle":
        outer_radius = np.max(np.hypot(xy[0], xy[1]))
        inner_radius = outer_radius / np.sqrt(2)
        inner_mask = np.less(np.hypot(xy[0], xy[1]), inner_radius, out=inner_mask)
    elif shape == "square":
        width, height = np.ptp(xy, -1)
        inner_width = width / np.sqrt(2)
        inner_height = height / np.sqrt(2)
        x_gap = (width - inner_width) / 2
        y_gap = (height - inner_height) / 2
        x_mask = (xy[0] > np.min(xy[0]) + x_gap) & (xy[0] < np.max(xy[0]) - x_gap)
        y_mask = (xy[1] > np.min(xy[1]) + y_gap) & (xy[1] < np.max(xy[1]) - y_gap)
        inner_mask = np.logical_and(x_mask, y_mask, out=inner_mask)
    return (np.count_nonzero(inner_mask) - np.count_nonzero(~inner_mask)) / n_pos
