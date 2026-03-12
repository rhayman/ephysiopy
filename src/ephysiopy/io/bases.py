from pathlib import Path, PurePath
import os
import numpy as np
from scipy.signal import argrelextrema
from skimage.segmentation import watershed
import abc
from ephysiopy.common.spikecalcs import SpikeCalcsGeneric
from ephysiopy.common.fieldcalcs import (
    skaggs_info,
    fancy_partition,
    simple_partition,
)
from ephysiopy.common.phasecoding import LFPOscillations, get_bad_cycles
from ephysiopy.common.fieldproperties import FieldProps, LFPSegment, fieldprops
from ephysiopy.common.binning import RateMap
from ephysiopy.visualise.plotting import FigureMaker
from ephysiopy.common.utils import (
    TrialFilter,
    VariableToBin,
    MapType,
    BinnedData,
    filter_data,
    shift_vector,
    make_cluster_ids,
    flatten_list,
)


def find_path_to_ripple_ttl(trial_root: Path, **kwargs) -> Path:
    """
    Iterates through a directory tree and finds the path to the
    Ripple Detector plugin data and returns its location
    """
    exp_name = kwargs.pop("experiment", "experiment1")
    rec_name = kwargs.pop("recording", "recording1")
    ripple_match = (
        trial_root
        / Path("Record Node [0-9][0-9][0-9]")
        / Path(exp_name)
        / Path(rec_name)
        / Path("events")
        / Path("Ripple_Detector-[0-9][0-9][0-9].*")
        / Path("TTL")
    )
    for d, c, f in os.walk(trial_root):
        for ff in f:
            if "." not in c:  # ignore hidden directories
                if "timestamps.npy" in ff:
                    if PurePath(d).match(str(ripple_match)):
                        return Path(d)
    return Path()


class TrialInterface(FigureMaker, metaclass=abc.ABCMeta):
    """
    Defines a minimal and required set of methods for loading
    electrophysiology data recorded using Axona or OpenEphys
    (OpenEphysNWB is there but not used)

    Parameters
    ----------
    pname (Path) : The path to the top-level directory containing the recording

    Attributes
    ----------
    pname (str) : the absolute pathname of the top-level data directory
    settings (dict) : contains metadata about the trial
    PosCalcs (PosCalcsGeneric) : contains the positional data for the trial
    RateMap : RateMap
        A class for binning data
    EEGCalcs : EEGCalcs
        For dealing with LFP data
    clusterData : clusterData
        contains results of a spike sorting session (i.e. KiloSort)
    recording_start_time : float
        the start time of the recording in seconds
    sync_message_file : Path
        the location of the sync_message_file (OpenEphys)
    ttl_data : dict
        ttl data including timestamps, ids and states
    accelerometer_data : np.ndarray
        data relating to headstage accelerometers
    path2PosData : Path
        location of the positional data
    mask_array : np.ma.MaskedArray
        contains the mask (if applied) for positional data
    filter : TrialFilter
        contains details of the filter applied to the positional data

    """

    def __init__(self, pname: Path, **kwargs) -> None:
        assert Path(pname).exists(), f"Path provided doesnt exist: {pname}"
        self._pname = pname
        self._settings = None
        self._PosCalcs = None
        self._RateMap = None
        self._EEGCalcs = None
        self._sync_message_file = None
        self._clusterData = None  # KiloSortSession
        self._recording_start_time = None  # float
        self._ttl_data = None  # dict
        self._accelerometer_data = None
        self._path2PosData = None  # Path or str
        self._filter: list = []
        self._mask_array = None
        self._concatenated = False  # whether this is a concatenated trial
        self._concatenated_trials = None  # list of trials if a concatenated trial

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "load_neural_data")
            and callable(subclass.load_neural_data)
            and hasattr(subclass, "load_lfp")
            and callable(subclass.load_lfp)
            and hasattr(subclass, "load_pos")
            and callable(subclass.load_pos)
            and hasattr(subclass, "load_cluster_data")
            and callable(subclass.load_cluster_data)
            and hasattr(subclass, "load_settings")
            and callable(subclass.load_settings)
            and hasattr(subclass, "get_spike_times")
            and callable(subclass.get_spike_times)
            and hasattr(subclass, "get_waveforms")
            and callable(subclass.get_waveforms)
            and hasattr(subclass, "get_available_clusters_channels")
            and callable(subclass.get_available_clusters_channels)
            and hasattr(subclass, "load_ttl")
            and callable(subclass.load_ttl)
            or NotImplemented
        )

    @property
    def pname(self):
        return self._pname

    @pname.setter
    def pname(self, val):
        self._pname = val

    @property
    def settings(self):
        return self._settings

    @settings.setter
    def settings(self, val):
        self._settings = val

    @property
    def PosCalcs(self):
        return self._PosCalcs

    @PosCalcs.setter
    def PosCalcs(self, val):
        self._PosCalcs = val

    @property
    def RateMap(self):
        return self._RateMap

    @RateMap.setter
    def RateMap(self, value):
        self._RateMap = value

    @property
    def EEGCalcs(self):
        return self._EEGCalcs

    @EEGCalcs.setter
    def EEGCalcs(self, val):
        self._EEGCalcs = val

    @property
    def clusterData(self):
        return self._clusterData

    @clusterData.setter
    def clusterData(self, val):
        self._clusterData = val

    @property
    def recording_start_time(self):
        return self._recording_start_time

    @recording_start_time.setter
    def recording_start_time(self, val):
        self._recording_start_time = val

    @property
    def sync_message_file(self):
        return self._sync_message_file

    @sync_message_file.setter
    def sync_message_file(self, val):
        self._sync_message_file = val

    @property
    def ttl_data(self):
        return self._ttl_data

    @ttl_data.setter
    def ttl_data(self, val):
        self._ttl_data = val

    @property
    def accelerometer_data(self):
        return self._accelerometer_data

    @accelerometer_data.setter
    def accelerometer_data(self, val):
        self._accelerometer_data = val

    @property
    def path2PosData(self):
        return self._path2PosData

    @path2PosData.setter
    def path2PosData(self, val):
        self._path2PosData = val

    @property
    def mask_array(self):
        if self._mask_array is None:
            if self.PosCalcs:
                self._mask_array = np.array(
                    np.zeros(shape=(1, self.PosCalcs.npos), dtype=bool)
                )
            else:
                self._mask_array = np.array(np.zeros(shape=(1, 1), dtype=bool))
            return self._mask_array
        else:
            return self._mask_array

    @mask_array.setter
    def mask_array(self, val):
        if self._mask_array is None:
            self._mask_array = np.array(np.zeros(shape=(1, 1), dtype=bool))
        if not isinstance(val, np.ndarray):
            if isinstance(val, (np.ndarray, list)):
                self._mask_array = np.array(val)
            elif isinstance(val, bool):
                self._mask_array.fill(val)
            else:
                raise TypeError("Need an array-like input")
        else:
            self._mask_array = val

    @property
    def concatenated(self):
        return self._concatenated

    @concatenated.setter
    def concatenated(self, val: bool):
        if not isinstance(val, bool):
            raise TypeError("concatenated must be a boolean")
        self._concatenated = val

    @property
    def concatenated_trials(self):
        return self._concatenated_trials

    @concatenated_trials.setter
    def concatenated_trials(self, val: list):
        if not isinstance(val, list):
            raise TypeError("concatenated_trials must be a list")
        self._concatenated_trials = val

    @property
    def filter(self):
        return self._filter

    def _update_filter(self, val: TrialFilter | None):
        if val is None:
            self._filter = []
        else:
            if val not in self._filter:
                self.filter.append(val)
        return self._filter

    @abc.abstractmethod
    def load_lfp(self, *args, **kwargs):
        """Load the LFP data"""
        raise NotImplementedError

    @abc.abstractmethod
    def load_neural_data(self, *args, **kwargs):
        """Load the neural data"""
        raise NotImplementedError

    @abc.abstractmethod
    def load_pos_data(self, ppm: int = 300, jumpmax: int = 100, *args, **kws):
        """
        Load the position data

        Parameters
        ----------
        ppm : int
            pixels per metre
        jumpmax : int
            max jump in pixels between positions, more
            than this and the position is interpolated over
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_cluster_data(self, *args, **kwargs) -> bool:
        """Load the cluster data (Kilosort/ Axona cut/ whatever else"""
        raise NotImplementedError

    @abc.abstractmethod
    def load_settings(self, *args, **kwargs):
        """Loads the format specific settings file"""
        raise NotImplementedError

    @abc.abstractmethod
    def load_ttl(self, *args, **kwargs) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def get_waveforms(self, cluster: int | list, channel: int | list, *args, **kwargs):
        """Returns the waveforms for a given cluster and channel

        Parameters
        ----------
        cluster : int | list
            The cluster(s) to get the waveforms for
        channel : int | list
            The channel(s) to get the waveforms for

        Returns
        -------
        list | np.ndarray
            the waveforms for the cluster(s) and channel(s)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_spike_times(
        self, cluster: int | list, channel: int | list, *args, **kwargs
    ) -> list | np.ndarray:
        """Returns the times of an individual cluster

        Parameters
        ----------
        cluster : int | list
            The cluster(s) to get the spike times for
        channel : int | list
            The channel(s) to get the spike times for

        Returns
        -------
        list | np.ndarray
            the spike times
        """
        raise NotImplementedError

    def get_field_properties(
        self, cluster: int | list, channel: int | list, **kwargs
    ) -> list[FieldProps]:
        """
        Gets the properties of a given field (area, runs through the field,
        etc)

        Parameters
        ----------
        cluster : int | list
            The cluster(s) to get the field properties for
        channel : int | list
            The channel(s) to get the field properties for
        **kwargs
            partition : str
                How the field is separated from the background. This is passed to
                the fieldproperties function and can be used to specify the partition
                to use for the field properties.

                Valid options are 'simple' and 'fancy'

                Other kwargs get passed to get_rate_map and
                fieldprops, the most important of which may be
                how the runs are split in fieldprops (options are
                'field' and 'clump_runs') which differ depending on
                if the position data is open-field (field) or linear track
                in which case you should probably use 'clunmp_runs'


        Returns
        -------
        list[FieldProps]
            A list of FieldProps namedtuples containing the properties of the field
        """
        partition = kwargs.pop("partition", "fancy")
        min_theta = kwargs.pop("min_theta", 6)
        max_theta = kwargs.pop("max_theta", 12)
        min_power_percent_threshold = kwargs.pop("min_power_percent_threshold", 0)

        if not self.RateMap:
            self.initialise()

        # First create the list of FieldProps objects. After that add the LFPSegments...
        # 1) get the rate map for the cluster and channel
        rmap = self.get_rate_map(cluster, channel, **kwargs)

        if partition == "simple":
            _, _, labels, _ = simple_partition(rmap)
        elif partition == "fancy":
            _, _, labels, _ = fancy_partition(rmap)

        breakpoint()

        spike_times = self.get_spike_times(cluster, channel)
        xy = getattr(self.PosCalcs, "xy")

        # 2) create the FieldProps list
        fp = fieldprops(labels, rmap, spike_times, xy, **kwargs)

        # 3) make sure the LFP is loaded and extract the phase and filter out
        # the bad segments
        if not self.EEGCalcs:
            self.load_lfp()

        L = LFPOscillations(self.EEGCalcs.sig, self.EEGCalcs.fs)

        FreqPhase = L.getFreqPhase(self.EEGCalcs.sig, [min_theta, max_theta], 2)

        phase = FreqPhase.phase

        minima = argrelextrema(phase, np.less)[0]
        markers = np.bincount(minima, minlength=len(phase))
        markers = np.cumsum(markers)
        cycle_label = watershed(phase, markers=markers)
        is_neg_freq = np.diff(np.unwrap(phase)) < 0
        is_neg_freq = np.append(is_neg_freq, is_neg_freq[-1])

        filt_sig = FreqPhase.filt_sig

        is_bad = get_bad_cycles(
            filt_sig,
            is_neg_freq,
            cycle_label,
            min_power_percent_threshold,
            min_theta,
            max_theta,
            self.EEGCalcs.fs,
        )

        lfp_to_pos_ratio = self.EEGCalcs.fs / self.PosCalcs.sample_rate

        for field in fp:
            for run in field.runs:
                lfp_slice = slice(
                    int(run.slice.start * lfp_to_pos_ratio),
                    int(run.slice.stop * lfp_to_pos_ratio),
                )

                lfp_segment = LFPSegment(
                    run,
                    field.label,
                    run.label,
                    lfp_slice,
                    spike_times=spike_times,
                    mask=is_bad[lfp_slice],
                    signal=self.EEGCalcs.sig[lfp_slice],
                    filtered_signal=filt_sig[lfp_slice],
                    phase=phase[lfp_slice],
                    cycle_label=cycle_label[lfp_slice],
                    sample_rate=self.EEGCalcs.fs,
                )
                run.lfp = lfp_segment

        return fp

    def apply_filter(self, *trial_filter: TrialFilter) -> np.ndarray:
        """
        Apply a mask to the recorded data. This will mask all the currently
        loaded data (LFP, position etc)

        Parameters
        ----------
        trial_filter : TrialFilter
            A namedtuple containing the filter
            name, start and end values
            name (str): The name of the filter
            start (float): The start value of the filter
            end (float): The end value of the filter

            Valid names are:
                'dir' - the directional range to filter for
                    NB Following mathmatical convention, 0/360 degrees is
                    3 o'clock, 90 degrees is 12 o'clock, 180 degrees is
                    9 o'clock and 270 degrees
                'speed' - min and max speed to filter for
                'xrange' - min and max values to filter x pos values
                'yrange' - same as xrange but for y pos
                'time' - the times to keep / remove specified in ms

            Values are pairs specifying the range of values to filter for
            from the namedtuple TrialFilter that has fields 'start' and 'end'
            where 'start' and 'end' are the ranges to filter for

        Returns
        -------
        np.ndarray
            An array of bools that is True where the mask is applied
        """
        # Remove any previously applied filter
        mask = False
        self.__apply_mask_to_subcls__(mask)
        self.mask_array = False
        self._update_filter(None)

        for i_filter in trial_filter:
            self._update_filter(i_filter)
            if "dir" in i_filter.name:
                mask = filter_data(self.PosCalcs.dir, i_filter)
            elif "time" in i_filter.name:
                mask = filter_data(self.PosCalcs.xyTS, i_filter)
            elif "speed" in i_filter.name:
                mask = filter_data(self.PosCalcs.speed, i_filter)
            elif "xrange" in i_filter.name:
                mask = filter_data(self.PosCalcs.xy[0, :], i_filter)
            elif "yrange" in i_filter.name:
                mask = filter_data(self.PosCalcs.xy[1, :], i_filter)
            elif "phi" in i_filter.name:
                mask = filter_data(self.PosCalcs.phi, i_filter)
            else:
                raise KeyError("Unrecognised key")
            self.mask_array = np.logical_or(self.mask_array, mask)

        mask = np.expand_dims(np.any(self.mask_array, axis=0), 0)
        self.mask_array = mask
        self.__apply_mask_to_subcls__(mask)
        return mask

    def __apply_mask_to_subcls__(self, mask: np.ndarray):
        """
        Applies a mask to the sub-class specific data
        """
        if self.EEGCalcs:
            self.EEGCalcs.apply_mask(mask)
        if self.PosCalcs:
            self.PosCalcs.apply_mask(mask)
        if self.RateMap:
            self.RateMap.apply_mask(mask)
        if self.clusterData:
            self.clusterData.apply_mask(
                mask,
                xy_ts=self.PosCalcs.xyTS,
                sample_rate=self.PosCalcs.sample_rate,
            )

    def initialise(self):
        self.RateMap = RateMap(self.PosCalcs)
        self.npos = self.PosCalcs.xy.shape[1]

    def get_available_clusters_channels(self) -> dict:
        raise NotImplementedError

    def get_binned_spike_times(
        self, cluster: int | list, channel: int | list, bin_into: str = "pos"
    ) -> np.ndarray:
        """
        Get the spike times binned into either position ("pos") or LFP ("lfp") data

        Parameters
        ----------
        cluster (int | list)
            The cluster(s).
        channel (int | list)
            The channel(s).
        bin_into (str)


        Returns
        -------
        np.ndarray
            the spike times binned into the position data
        """
        ts = self.get_spike_times(cluster, channel)
        if not isinstance(ts, list):
            ts = [ts]
        n_clusters = 1
        if isinstance(cluster, list):
            n_clusters = len(cluster)
        if bin_into == "pos":
            n_pos = self.PosCalcs.npos
            sample_rate = self.PosCalcs.sample_rate
        if bin_into == "lfp":
            n_pos = self.EEGCalcs.sig.shape[0]
            sample_rate = self.EEGCalcs.fs
        binned = np.zeros((n_clusters, n_pos))
        for i, t in enumerate(ts):
            spk_binned = np.bincount((t * sample_rate).astype(int), minlength=n_pos)
            if len(spk_binned) > n_pos:
                spk_binned = spk_binned[:n_pos]
            binned[i, :] = spk_binned
        return binned

    def _get_spike_pos_idx(
        self, cluster: int | list | None, channel: int | list, **kwargs
    ) -> list[np.ndarray]:
        """
        Returns the indices into the position data at which some cluster
        on a given channel emitted putative spikes.

        Parameters
        ----------
        cluster : int | list
            The cluster(s). NB this can be None in which
            case the "spike times" are equal to the position times, which
            means data binned using these indices will be equivalent to
            binning up just the position data alone.

        channel : int | list
            The channel identity. Ignored if cluster is None

        Returns
        -------
        list of np.ndarray
            The indices into the position data at which the spikes
            occurred.
        """
        pos_times = getattr(self.PosCalcs, "xyTS")
        if cluster is None:
            spk_times = getattr(self.PosCalcs, "xyTS")
        elif isinstance(cluster, int):
            spk_times = self.get_spike_times(cluster, channel)
        elif isinstance(cluster, list) and len(cluster) == 1:
            spk_times = self.get_spike_times(cluster[0], channel[0])
        elif isinstance(cluster, list) and len(cluster) > 1:
            if isinstance(channel, int):
                channel = [channel for c in cluster]
            spk_times = []
            for clust, chan in zip(cluster, channel):
                spk_times.append(self.get_spike_times(clust, chan))

        if isinstance(spk_times, list):
            idx = []
            for spk in spk_times:
                _idx = np.searchsorted(pos_times, spk, side="right") - 1
                if np.any(_idx >= self.PosCalcs.npos):
                    _idx = np.delete(
                        _idx, np.s_[np.argmax(_idx >= self.PosCalcs.npos) :]
                    )
                idx.append(_idx)
        else:
            idx = np.searchsorted(pos_times, spk_times, side="right") - 1
            if np.any(idx >= self.PosCalcs.npos):
                idx = np.delete(idx, np.s_[np.argmax(idx >= self.PosCalcs.npos) :])

        if kwargs.get("do_shuffle", False):
            n_shuffles = kwargs.get("n_shuffles", 100)
            random_seed = kwargs.get("random_seed", None)
            r = np.random.default_rng(random_seed)
            time_shifts = r.integers(
                low=30 * self.PosCalcs.sample_rate,
                high=self.PosCalcs.npos - (30 * self.PosCalcs.sample_rate),
                size=n_shuffles,
            )
            shifted_idx = []
            for shift in time_shifts:
                shifted_idx.append(shift_vector(idx, shift, maxlen=self.PosCalcs.npos))
            return shifted_idx

        if isinstance(idx, list):
            return idx
        else:
            return [idx]

    def get_all_maps(
        self,
        channels_clusters: dict,
        var2bin: VariableToBin = VariableToBin.XY,
        maptype: MapType = MapType.RATE,
        **kwargs,
    ) -> dict:
        old_data = None
        for channel, clusters in channels_clusters.items():
            channels = [channel] * len(clusters)
            data = self._get_map(
                clusters, channels, var2bin, map_type=maptype, **kwargs
            )
            if old_data is None:
                old_data = data
            else:
                old_data = old_data + data
        return old_data

    def _get_map(
        self,
        cluster: int | list,
        channel: int | list,
        var2bin: VariableToBin.XY,
        **kwargs,
    ) -> BinnedData:
        """
        This function generates a rate map for a given cluster and channel.

        Parameters
        ----------
        cluster : int or list
            The cluster(s).
        channel : int or list
            The channel(s).
        var2bin : VariableToBin.XY
            The variable to bin.
        **kwargs : dict, optional
            Additional keyword arguments for the _get_spike_pos_idx function.
            - do_shuffle (bool): If True, the rate map will be shuffled by
                        n_shuffles
            - map_type (MapType): the type of map to generate, default
                        is MapType.POS but can be any of the options
                        in MapType
                                 the default number of shuffles (100).
            - n_shuffles (int): the number of shuffles for the rate map
                                A list of shuffled rate maps will be returned.
            - random_seed (int): The random seed to use for the shuffles.

        Returns
        -------
        np.ndarray
            The rate map as a numpy array.

        """
        if not self.RateMap:
            self.initialise()
        # TODO: _get_spike_pos_idx always returns a list now so this fcn needs
        # amending as will the get_map() one in binning.RateMap as it looks
        # like that expects a np.ndarray
        spk_times_in_pos_samples = self._get_spike_pos_idx(cluster, channel, **kwargs)
        npos = self._PosCalcs.npos
        # This conditional just picks out the right spk_weights
        # given the inputs
        if len(spk_times_in_pos_samples) == 1:
            spk_times_in_pos_samples = np.array(spk_times_in_pos_samples[0])
        if isinstance(spk_times_in_pos_samples, np.ndarray):
            spk_weights = np.bincount(spk_times_in_pos_samples, minlength=npos)
            if len(spk_weights) > npos:
                spk_weights = np.delete(spk_weights, np.s_[npos:], 0)

        elif (
            isinstance(spk_times_in_pos_samples, list)
            and len(spk_times_in_pos_samples) > 1
        ):  # likely the result of a shuffle arg passed to get_spike_pos_idx
            # TODO: could be multiple clusters/ channels have been passed
            weights = []
            if isinstance(spk_times_in_pos_samples[0], list):
                spk_times_in_pos_samples = flatten_list(spk_times_in_pos_samples)
            for spk_idx in spk_times_in_pos_samples:
                w = np.bincount(spk_idx, minlength=npos)
                if len(w) > npos:
                    w = np.delete(w, np.s_[npos:], 0)
                weights.append(w)
            spk_weights = np.array(weights)

        kwargs["var_type"] = var2bin
        rmap = self.RateMap.get_map(spk_weights, **kwargs)
        # add the cluster and channel id to the rate map
        # I assume this will be in the same order as they are added...
        ids = make_cluster_ids(channel, cluster)
        rmap.cluster_id = ids
        return rmap

    def get_rate_map(
        self, cluster: int | list, channel: int | list, **kwargs
    ) -> BinnedData:
        """
        Gets the rate map for the specified cluster(s) and channel.

        Parameters
        ----------
        cluster : int, list
            The cluster(s) to get the speed vs rate for.
        channel : int, list
            The channel(s) number.
        **kwargs
            Additional keyword arguments passed to _get_map

        Returns
        -------
        BinnedData
            the binned data
        """
        var_type = kwargs.pop("var_type", VariableToBin.XY)
        return self._get_map(cluster, channel, var2bin=var_type, **kwargs)

    def get_linear_rate_map(self, cluster: int | list, channel: int | list, **kwargs):
        """
        Gets the linear rate map for the specified cluster(s) and channel.

        Parameters
        ----------
        cluster : int, list
            The cluster(s) to get the speed vs rate for.
        channel : int, list
            The channel(s) number.
        **kwargs
            Additional keyword arguments passed to _get_map

        Returns
        -------
        BinnedData
            the binned data
        """
        var_type = kwargs.pop("var_type", VariableToBin.PHI)
        return self.get_rate_map(cluster, channel, var_type=var_type, **kwargs)

    def get_hd_map(
        self, cluster: int | list, channel: int | list, **kwargs
    ) -> BinnedData:
        """
        Gets the head direction map for the specified cluster(s) and channel.

        Parameters
        ----------
        cluster : int, list
            The cluster(s) to get the speed vs rate for.
        channel : int,  list
            The channel(s) number.
        **kwargs: Additional keyword arguments passed to _get_map

        Returns
        -------
        BinnedData - the binned data
        """

        return self._get_map(cluster, channel, VariableToBin.DIR, **kwargs)

    def get_eb_map(
        self, cluster: int | list, channel: int | list, **kwargs
    ) -> BinnedData:
        """
        Gets the egocentric boundary map for the cluster(s) and channel.

        Parameters
        ----------
        cluster : int, list
            The cluster(s) to get the speed vs rate for.
        channel : int, list
            The channel(s) number.
        **kwargs: Additional keyword arguments passed to _get_map

        Returns
        -------
        BinnedData
            the binned data
        """
        return self._get_map(cluster, channel, VariableToBin.EGO_BOUNDARY, **kwargs)

    def get_speed_v_rate_map(
        self, cluster: int | list, channel: int | list, **kwargs
    ) -> BinnedData:
        """
        Gets the speed vs rate for the specified cluster(s) and channel.

        Parameters
        ----------
        cluster : int, list
                        The cluster(s) to get the speed vs rate for.
        channel : int, list
                        The channel(s) number.
        **kwargs: Additional keyword arguments passed to _get_map

        Returns
        -------
        BinnedData - the binned data
        """
        return self._get_map(cluster, channel, VariableToBin.SPEED, **kwargs)

    def get_speed_v_hd_map(
        self, cluster: int | list, channel: int | list, **kwargs
    ) -> BinnedData:
        """
        Gets the speed vs head direction map for the cluster(s) and channel.

        Parameters
        ----------
        cluster : int, list
                        The cluster(s)
        channel : int, list
                        The channel number.
        **kwargs: Additional keyword arguments passed to _get_map
        """
        # binsize is in cm/s and degrees
        binsize = kwargs.get("binsize", (2.5, 3))
        return self._get_map(
            cluster, channel, VariableToBin.SPEED_DIR, **dict(kwargs, binsize=binsize)
        )

    def get_grid_map(
        self, cluster: int | list, channel: int | list, **kwargs
    ) -> BinnedData:
        """
        Generates a grid map for a given cluster and channel.

        Parameters
        ----------
        cluster : int or list
            The cluster(s).
        channel : int or list
            The channel(s).
        **kwargs : dict, optional
            Additional keyword arguments passed to the autoCorr2D function.

        Returns
        -------
        BinnedData
            The grid map as a BinnedData object.
        """
        rmap = self.get_rate_map(cluster, channel, **kwargs)
        ids = make_cluster_ids(cluster, channel)
        if kwargs.get("do_shuffle", True):
            if len(rmap.binned_data) != len(ids):
                # repeat the ids for each shuffle
                ids = ids * len(rmap.binned_data)
        kwargs["cluster_id"] = ids
        sac = self.RateMap.autoCorr2D(rmap, **kwargs)
        return sac

    def get_adaptive_map(
        self, cluster: int | list, channel: int | list, **kwargs
    ) -> BinnedData:
        """
        Generates an adaptive map for a given cluster and channel.

        Parameters
        ----------
        cluster : int or list
            The cluster(s).
        channel : int or list
            The channel(s).
        **kwargs : dict, optional
            Additional keyword arguments passed to the _get_map function.

        Returns
        -------
        BinnedData
            The adaptive map as a BinnedData object.
        """
        return self._get_map(
            cluster, channel, VariableToBin.XY, map_type=MapType.ADAPTIVE, **kwargs
        )

    def get_acorr(
        self, cluster: int | list, channel: int | list, **kwargs
    ) -> BinnedData:
        """
        Computes the cross-correlation for a given cluster and channel.

        Parameters
        ----------
        cluster : int or list
            The cluster(s).
        channel : int or list
            The channel(s).
        **kwargs : dict, optional
            Additional keyword arguments passed to the xcorr function.

        Returns
        -------
        BinnedData
            The cross-correlation as a BinnedData object.
        """
        from ephysiopy.common.spikecalcs import xcorr

        ts = self.get_spike_times(cluster, channel)
        ids = make_cluster_ids(cluster, channel)
        kwargs["cluster_id"] = ids
        return xcorr(ts, **kwargs)

    def get_xcorr(
        self,
        cluster_a: int | list,
        cluster_b: int | list,
        channel_a: int | list,
        channel_b: int | list,
        **kwargs,
    ) -> BinnedData:
        """
        Computes the cross-correlation for a given cluster and channel.

        Parameters
        ----------
        cluster : int or list
            The cluster(s).
        channel : int or list
            The channel(s).
        **kwargs : dict, optional
            Additional keyword arguments passed to the xcorr function.

        Returns
        -------
        BinnedData
            The cross-correlation as a BinnedData object.
        """
        from ephysiopy.common.spikecalcs import xcorr

        ts_a = self.get_spike_times(cluster_a, channel_a)
        ts_b = self.get_spike_times(cluster_b, channel_b)
        ids = make_cluster_ids([cluster_a, cluster_b], [channel_a, channel_b])
        kwargs["cluster_id"] = ids
        return xcorr(ts_a, ts_b, **kwargs)

    def get_psth(
        self, cluster: int | list, channel: int | list, **kwargs
    ) -> tuple[list, list]:
        """
        Computes the peri-stimulus time histogram (PSTH) for a given cluster and channel.

        Parameters
        ----------
        cluster : int or list
            The cluster(s).
        channel : int or list
            The channel(s).
        **kwargs : dict, optional
            Additional keyword arguments passed to the psth function.

        Returns
        -------
        tuple of lists
            The list of time differences between the spikes of the cluster
            and the events (0) and the trials (1)

        """
        spike_times = self.get_spike_times(cluster, channel)
        S = SpikeCalcsGeneric(spike_times, cluster=cluster)
        if self.ttl_data is None:
            self.load_ttl()
        S.event_ts = self.ttl_data["ttl_timestamps"]
        dt = kwargs.get("dt", (-0.05, 0.1))
        S.event_window = np.array(dt)
        return S.psth()

    def get_spatial_info_score(
        self, cluster: int | list, channel: int | list, **kwargs
    ) -> list[float]:
        """
        Computes the spatial information score

        Parameters
        ----------
        cluster : int or list
            The cluster(s).
        channel : int or list
            The channel(s).
        **kwargs : dict, optional
            Additional keyword arguments passed to the binning function.

        Returns
        -------
        float
            The spatial information score

        """
        pos_map = self.get_rate_map(
            cluster, channel, map_type=MapType.POS, smoothing=False, **kwargs
        )
        r_map = self.get_rate_map(
            cluster, channel, map_type=MapType.ADAPTIVE, smoothing=False, **kwargs
        )
        result = []
        for rm in r_map:
            result.append(skaggs_info(rm.binned_data[0], pos_map.binned_data[0]))

        return result
