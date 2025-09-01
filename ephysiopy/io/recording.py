import abc
import os
import re
import warnings
from enum import Enum
from pathlib import Path, PurePath

import h5py
import numpy as np
from phylib.io.model import TemplateModel

from ephysiopy.axona.axonaIO import IO, Pos
from ephysiopy.axona.tetrode_dict import TetrodeDict
from ephysiopy.common.ephys_generic import (
    EEGCalcsGeneric,
    PosCalcsGeneric,
)
from ephysiopy.common.spikecalcs import SpikeCalcsGeneric
from ephysiopy.common.fieldcalcs import skaggs_info
from ephysiopy.common.binning import RateMap
from ephysiopy.common.utils import VariableToBin, MapType, BinnedData, filter_data
from ephysiopy.openephys2py.KiloSort import KiloSortSession
from ephysiopy.openephys2py.OESettings import Settings
from ephysiopy.visualise.plotting import FigureMaker
from ephysiopy.common.utils import (
    shift_vector,
    TrialFilter,
    memmapBinaryFile,
    fileContainsString,
    ClusterID,
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


class RecordingKind(Enum):
    FPGA = 1
    NEUROPIXELS = 2
    ACQUISITIONBOARD = 3
    NWB = 4


Xml2RecordingKind = {
    "Acquisition Board": RecordingKind.ACQUISITIONBOARD,
    "Neuropix-PXI": RecordingKind.NEUROPIXELS,
    "Rhythm FPGA": RecordingKind.FPGA,
    "Rhythm": RecordingKind.FPGA,
    "Acquistion": RecordingKind.ACQUISITIONBOARD,
    "Neuropix": RecordingKind.NEUROPIXELS,
}


def make_cluster_ids(cluster: int | list, channel: int | list) -> list:
    # add the cluster and channel id to the rate map
    # I assume this will be in the same order as they are added...
    ids = []
    if cluster is None:
        return [None, None]
    if isinstance(channel, int) and isinstance(cluster, list):
        channel = [channel for c in cluster]
    if isinstance(cluster, int):
        cluster = [cluster]
    if isinstance(channel, int):
        channel = [channel]
    for cl_ch in zip(cluster, channel):
        ids.append(ClusterID(cl_ch[1], cl_ch[0]))
    return ids


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
        methods for binning data mostly
    EEGCalcs : EEGCalcs
        methods for dealing with LFP data
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
        self._concatenated = False  # whether this is a concatenated trial or not

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

    def get_spike_times_binned_into_position(
        self, cluster: int | list, channel: int | list
    ) -> np.ndarray:
        """
        Parameters
        ----------
        cluster (int | list)
            The cluster(s).
        channel (int | list)
            The channel(s).

        Returns
        -------
        np.ndarray
            the spike times binned into the position data
        """
        ts = self.get_spike_times(cluster, channel)
        if not isinstance(ts, list):
            ts = [ts]
        n_clusters = 1
        if isinstance(n_clusters, list):
            n_clusters = len(cluster)
        n_pos = self.PosCalcs.npos
        binned = np.zeros((n_clusters, n_pos))
        for i, t in enumerate(ts):
            spk_binned = np.bincount(
                (t * self.PosCalcs.sample_rate).astype(int), minlength=n_pos
            )
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
                        _idx, np.s_[np.argmax(_idx >= self.PosCalcs.npos):]
                    )
                idx.append(_idx)
        else:
            idx = np.searchsorted(pos_times, spk_times, side="right") - 1
            if np.any(idx >= self.PosCalcs.npos):
                idx = np.delete(
                    idx, np.s_[np.argmax(idx >= self.PosCalcs.npos):])

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
                shifted_idx.append(shift_vector(
                    idx, shift, maxlen=self.PosCalcs.npos))
            return shifted_idx

        if isinstance(idx, list):
            return idx
        else:
            return [idx]

    def get_all_maps(
        self,
        channels_clusters: dict,
        var2bin: VariableToBin.XY,
        maptype: MapType,
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
        spk_times_in_pos_samples = self._get_spike_pos_idx(
            cluster, channel, **kwargs)
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
                from ephysiopy.common.utils import flatten_list

                spk_times_in_pos_samples = flatten_list(
                    spk_times_in_pos_samples)
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
    ) -> BinnedData:
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
        BinnedData
            The PSTH as a BinnedData object.
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
            result.append(skaggs_info(
                rm.binned_data[0], pos_map.binned_data[0]))

        return result


class AxonaTrial(TrialInterface):
    def __init__(self, pname: Path, **kwargs) -> None:
        use_volts = kwargs.pop("volts", True)
        pname = Path(pname)
        super().__init__(pname, **kwargs)
        self._settings = None
        self.TETRODE = TetrodeDict(
            str(self.pname.with_suffix("")), volts=use_volts)
        self.load_settings()

    def __add__(self, other):
        if isinstance(other, AxonaTrial):
            if self.pname == other.pname:
                return self
            else:
                new_AxonaTrial = AxonaTrial(self.pname)
                # make sure position data is loaded
                print("Merging position data...")
                ppm = self.settings["tracker_pixels_per_metre"]
                self.load_pos_data(int(ppm))
                ppm = other.settings["tracker_pixels_per_metre"]
                other.load_pos_data(int(ppm))
                # merge position data
                new_AxonaTrial.PosCalcs = self.PosCalcs + other.PosCalcs
                new_AxonaTrial.PosCalcs.postprocesspos({"SampleRate": 50})

                print("Done merging position data.")
                print("Merging LFP data...")

                # load EEG data
                self.load_lfp()
                other.load_lfp()
                # merge EEG data
                if self.EEGCalcs and other.EEGCalcs:
                    new_AxonaTrial.EEGCalcs = self.EEGCalcs + other.EEGCalcs
                elif self.EEGCalcs:
                    new_AxonaTrial.EEGCalcs = self.EEGCalcs
                elif other.EEGCalcs:
                    new_AxonaTrial.EEGCalcs = other.EEGCalcs
                else:
                    new_AxonaTrial.EEGCalcs = None
                print("Done merging LFP data.")

                # merge tetrode data
                print("Merging tetrode data...")
                self_tetrodes = self.get_available_clusters_channels().keys()
                other_tetrodes = other.get_available_clusters_channels().keys()
                print("Got all tetrodes...")
                for tetrode in self_tetrodes:
                    if tetrode in other_tetrodes:
                        new_AxonaTrial.TETRODE[tetrode] = (
                            self.TETRODE[tetrode] + other.TETRODE[tetrode]
                        )
                    else:
                        print(f"Missing tetrode {tetrode} in other trial")
                        new_AxonaTrial.TETRODE[tetrode] = self.TETRODE[tetrode]

                print("Done merging tetrode data.")

                new_AxonaTrial.concatenated = True
                return new_AxonaTrial

        else:
            raise TypeError("Can only add AxonaTrial instances")

    def load_lfp(self, *args, **kwargs):
        from ephysiopy.axona.axonaIO import EEG

        if not self.concatenated:
            if "egf" in args:
                lfp = EEG(self.pname, egf=1)
            else:
                lfp = EEG(self.pname)
            if lfp is not None:
                self.EEGCalcs = EEGCalcsGeneric(lfp.sig, lfp.sample_rate)

    def load_neural_data(self, *args, **kwargs):
        if "tetrode" in kwargs.keys():
            use_volts = kwargs.get("volts", True)
            self.TETRODE[kwargs["tetrode"], use_volts]  # lazy load

    def load_cluster_data(self, *args, **kwargs):
        return False

    def get_available_clusters_channels(self, remove0=True) -> dict:
        """
        Slightly laborious and low-level way of getting the cut
        data but it's faster than accessing the TETRODE's as that
        will load the waveforms as well as everything else
        """
        clust_chans = {}
        pattern = re.compile(
            str(self.pname.name).replace(".set", ".[0-9]\.cut"))
        cuts = sorted(
            [Path(f)
             for f in os.listdir(self.pname.parent) if pattern.match(f)]
        )

        def load_cut(fname: Path):
            a = []
            with open(fname, "r") as f:
                data = f.read()
                f.close()
            tmp = data.split("spikes: ")
            tmp1 = tmp[1].split("\n")
            cut = tmp1[1:]
            for line in cut:
                m = line.split()
                for i in m:
                    a.append(int(i))
            return np.array(a)

        if cuts:
            for cut in cuts:
                cut_path = self.pname.parent / cut
                if cut_path.exists():
                    clusters = np.unique(load_cut(cut_path)).tolist()
                    if remove0:
                        clusters.remove(0)
                    if clusters:
                        tetrode_num = int(cut_path.stem.rsplit("_")[-1])
                        clust_chans[tetrode_num] = clusters

        return clust_chans

    def load_settings(self, *args, **kwargs):
        if self._settings is None:
            try:
                settings_io = IO()
                self.settings = settings_io.getHeader(str(self.pname))
            except IOError:
                print(".set file not loaded")
                self.settings = None

    def load_pos_data(
        self, ppm: int = 300, jumpmax: int = 100, *args, **kwargs
    ) -> None:
        try:
            if not self.concatenated:
                AxonaPos = Pos(Path(self.pname))
                P = PosCalcsGeneric(
                    AxonaPos.led_pos[:, 0],
                    AxonaPos.led_pos[:, 1],
                    cm=True,
                    ppm=ppm,
                    jumpmax=jumpmax,
                )
                P.sample_rate = AxonaPos.getHeaderVal(
                    AxonaPos.header, "sample_rate")
                P.xyTS = AxonaPos.ts / P.sample_rate  # in seconds now
                P.postprocesspos(tracker_params={"SampleRate": P.sample_rate})
                print("Loaded pos data")
                self.PosCalcs = P
        except IOError:
            print("Couldn't load the pos data")

    def load_ttl(self, *args, **kwargs) -> bool:
        from ephysiopy.axona.axonaIO import Stim

        try:
            self.ttl_data = Stim(self.pname)
            # ttl times in Stim are in seconds
        except IOError:
            return False
        print("Loaded ttl data")
        return True

    def get_spike_times(
        self, cluster: int | list = None, tetrode: int | list = None, *args, **kwargs
    ) -> list | np.ndarray:
        if tetrode is not None:
            if isinstance(cluster, int):
                return self.TETRODE.get_spike_samples(int(tetrode), int(cluster))
            elif isinstance(cluster, list) and isinstance(tetrode, list):
                if len(cluster) == 1:
                    tetrode = tetrode[0]
                    cluster = cluster[0]
                    return self.TETRODE.get_spike_samples(int(tetrode), int(cluster))
                else:
                    spikes = []
                    for tc in zip(tetrode, cluster):
                        spikes.append(
                            self.TETRODE.get_spike_samples(tc[0], tc[1]))
                    return spikes

    def get_waveforms(self, cluster: int | list, channel: int | list, *args, **kwargs):
        if isinstance(cluster, int) and isinstance(channel, int):
            return self.TETRODE[channel].get_waveforms(int(cluster))
        elif isinstance(cluster, list) and isinstance(channel, int):
            if len(cluster) == 1:
                return self.TETRODE[channel].get_waveforms(int(cluster[0]))
        elif isinstance(cluster, list) and isinstance(channel, list):
            waveforms = []
            for c, ch in zip(cluster, channel):
                waveforms.append(self.TETRODE[int(ch)].get_waveforms(int(c)))
            return waveforms

    def apply_filter(self, *trial_filter: TrialFilter) -> np.ndarray:
        mask = super().apply_filter(*trial_filter)
        for tetrode in self.TETRODE.valid_keys:
            if self.TETRODE[tetrode] is not None:
                self.TETRODE[tetrode].apply_mask(
                    mask, sample_rate=self.PosCalcs.sample_rate
                )
        return mask


class OpenEphysBase(TrialInterface):
    def __init__(self, pname: Path, **kwargs) -> None:
        pname = Path(pname)
        super().__init__(pname, **kwargs)
        setattr(self, "sync_message_file", None)
        self.load_settings()
        # The numbers after the strings in this list are the node id's
        # in openephys
        record_methods = [
            "Acquisition Board [0-9][0-9][0-9]",
            "Acquisition Board",
            "Neuropix-PXI [0-9][0-9][0-9]",
            "Neuropix-PXI",
            "Sources/Neuropix-PXI [0-9][0-9][0-9]",
            "Rhythm FPGA [0-9][0-9][0-9]",
            "Rhythm",
            "Sources/Rhythm FPGA [0-9][0-9][0-9]",
        ]
        rec_method = [
            re.search(m, k).string
            for k in self.settings.processors.keys()
            for m in record_methods
            if re.search(m, k) is not None
        ][0]
        if "Sources/" in rec_method:
            rec_method = rec_method.lstrip("Sources/")

        self.rec_kind = Xml2RecordingKind[rec_method.rpartition(" ")[0]]

        # Attempt to find the files contained in the parent directory
        # related to the recording with the default experiment and
        # recording name
        self.find_files(pname, **kwargs)
        self.sample_rate = None
        self.sample_rate = self.settings.processors[rec_method].sample_rate
        if self.sample_rate is None:
            if self.rec_kind == RecordingKind.NEUROPIXELS:
                self.sample_rate = 30000
        else:  # rubbish fix - many strs need casting to int/float
            self.sample_rate = float(self.sample_rate)
        self.channel_count = self.settings.processors[rec_method].channel_count
        if self.channel_count is None:
            if self.rec_kind == RecordingKind.NEUROPIXELS:
                self.channel_count = 384
        self.kilodata = None
        self.template_model = None

    def _get_recording_start_time(self) -> float:
        recording_start_time = 0.0
        if self.sync_message_file is not None:
            with open(self.sync_message_file, "r") as f:
                sync_strs = f.read()
            sync_lines = sync_strs.split("\n")
            for line in sync_lines:
                if "Start Time" in line:
                    tokens = line.split(":")
                    start_time = int(tokens[-1])
                    sample_rate = int(tokens[0].split(
                        "@")[-1].strip().split()[0])
                    recording_start_time = start_time / float(sample_rate)
        self.recording_start_time = recording_start_time
        return recording_start_time

    def get_spike_times(
        self, cluster: int | list = None, tetrode: int | list = None, *args, **kwargs
    ) -> list | np.ndarray:
        if not self.clusterData:
            self.load_cluster_data()
        if isinstance(cluster, int) and isinstance(tetrode, int):
            if cluster in self.clusterData.spk_clusters:
                times = self.clusterData.get_cluster_spike_times(cluster)
                return times.astype(np.int64) / self.sample_rate
            else:
                warnings.warn("Cluster not present")
        elif isinstance(cluster, list) and isinstance(tetrode, list):
            times = []
            for c in cluster:
                if c in self.clusterData.spk_clusters:
                    t = self.clusterData.get_cluster_spike_times(c)
                    times.append(t.astype(np.int64) / self.sample_rate)
                else:
                    warnings.warn("Cluster not present")
            return times

    def load_lfp(self, *args, **kwargs):
        from scipy import signal

        if self.path2LFPdata is not None:
            lfp = memmapBinaryFile(
                os.path.join(self.path2LFPdata, "continuous.dat"),
                n_channels=self.channel_count,
            )
            channel = kwargs.get("channel", 0)
            # set the target sample rate to 250Hz by default to match
            # Axona EEG data
            target_sample_rate = kwargs.get("target_sample_rate", 250)
            denom = np.gcd(int(target_sample_rate), int(self.sample_rate))
            data = lfp[channel, :]
            sig = signal.resample_poly(
                data.astype(float),
                target_sample_rate / denom,
                self.sample_rate / denom,
                0,
            )
            self.EEGCalcs = EEGCalcsGeneric(sig, target_sample_rate)

    def load_neural_data(self, *args, **kwargs):
        if "path2APdata" in kwargs.keys():
            self.path2APdata: Path = Path(kwargs["path2APdata"])
        n_channels: int = self.channel_count or kwargs["nChannels"]
        try:
            self.template_model = TemplateModel(
                dir_path=self.path2KiloSortData,
                sample_rate=3e4,
                dat_path=Path(self.path2KiloSortData).joinpath(
                    "continuous.dat"),
                n_channels_dat=int(n_channels),
            )
            print("Loaded neural data")
        except Exception:
            warnings.warn("Could not find raw data file")

    def load_settings(self, *args, **kwargs):
        if self._settings is None:
            # pname_root gets walked through and over-written with
            # correct location of settings.xml
            self.settings = Settings(self.pname)
            print("Loaded settings data\n")

    def get_available_clusters_channels(self) -> dict:
        """
        Get available clusters and their corresponding channels.

        Returns
        -------
        dict
            A dict where keys are channels and values are lists of clusters
        """
        if self.template_model is None:
            self.load_neural_data()
        unique_clusters = np.unique(self.template_model.spike_clusters)
        clust_chans = dict.fromkeys(
            np.unique(self.template_model.clusters_channels))
        for clust_id, chan in enumerate(self.template_model.clusters_channels):
            if clust_id in unique_clusters:
                clust_chans[chan] = [
                    ch
                    for ch, cl in enumerate(self.template_model.clusters_channels)
                    if cl == chan
                ]

        return clust_chans

    def load_cluster_data(self, removeNoiseClusters=True, *args, **kwargs) -> bool:
        if self.path2KiloSortData is not None:
            clusterData = KiloSortSession(self.pname)
        else:
            return False
        if clusterData is not None:
            if clusterData.load():
                print("Loaded KiloSort data")
                if removeNoiseClusters:
                    try:
                        clusterData.removeKSNoiseClusters()
                        print("Removed noise clusters")
                    except Exception:
                        pass
        else:
            return False
        self.clusterData = clusterData
        return True

    def load_pos_data(
        self, ppm: int = 300, jumpmax: int = 100, *args, **kwargs
    ) -> None:
        # kwargs valid keys = "loadTTLPos" - if present loads the ttl
        # timestamps not the ones in the plugin folder

        # Only sub-class that doesn't use this is OpenEphysNWB
        # which needs updating
        # TODO: Update / overhaul OpenEphysNWB
        # Load the start time from the sync_messages file
        if "cm" in kwargs:
            cm = kwargs["cm"]
        else:
            cm = True

        recording_start_time = self._get_recording_start_time()

        if self.path2PosData is not None:
            pos_method = [
                "Pos Tracker [0-9][0-9][0-9]",
                "PosTracker [0-9][0-9][0-9]",
                "TrackMe [0-9][0-9][0-9]",
                "TrackingPlugin [0-9][0-9][0-9]",
                "Tracking Port",
            ]
            pos_plugin_name = [
                re.search(m, k).string
                for k in self.settings.processors.keys()
                for m in pos_method
                if re.search(m, k) is not None
            ][0]
            if "Sources/" in pos_plugin_name:
                pos_plugin_name = pos_plugin_name.lstrip("Sources/")

            self.pos_plugin_name = pos_plugin_name
            tracker = self.settings.get_processor(pos_plugin_name)
            pos_data = tracker.load(self.path2PosData)

            # TODO: Can't find trials where this plugin is used...maybe on some backup...
            # if "TrackingPlugin" in pos_plugin_name:
            #     print("Loading TrackingPlugin data...")
            #     pos_data = loadTrackingPluginData(
            #         os.path.join(self.path2PosData, "data_array.npy")
            #     )

            pos_ts = tracker.load_times(self.path2PosData)
            # pos_ts in seconds
            pos_ts = np.ravel(pos_ts)
            if "TrackMe" in pos_plugin_name:
                if "loadTTLPos" in kwargs.keys():
                    pos_ts = tracker.load_ttl_times(Path(self.path2EventsData))
                else:
                    pos_ts = tracker.load_times(Path(self.path2PosData))
                pos_ts = pos_ts[0: len(pos_data)]
            sample_rate = tracker.sample_rate
            # sample_rate = float(sample_rate) if sample_rate is not None else 50
            # the timestamps for the Tracker Port plugin are fucked so
            # we have to infer from the shape of the position data
            if "Tracking Port" in pos_plugin_name:
                sample_rate = kwargs.get("sample_rate", 50)
                # pos_ts in seconds
                pos_ts = np.arange(
                    0, pos_data.shape[0] / sample_rate, 1.0 / sample_rate
                )
            if "TrackMe" not in pos_plugin_name:
                xyTS = pos_ts - recording_start_time
            else:
                xyTS = pos_ts
            if self.sync_message_file is not None:
                recording_start_time = xyTS[0]

            # This is the gateway to all the position processing so if you want
            # to load your own pos data you'll need to create an instance of
            # PosCalcsGeneric yourself and apply it to self
            P = PosCalcsGeneric(
                pos_data[:, 0],
                pos_data[:, 1],
                cm=cm,
                ppm=ppm,
                jumpmax=jumpmax,
            )
            P.xyTS = xyTS
            P.sample_rate = sample_rate
            P.postprocesspos({"SampleRate": sample_rate})
            print("Loaded pos data")
            self.PosCalcs = P
        else:
            warnings.warn(
                "Could not find the pos data. \
                Make sure there is a pos_data folder with data_array.npy \
                and timestamps.npy in"
            )
        self.recording_start_time = recording_start_time

    def load_ttl(self, *args, **kwargs) -> bool:
        if not Path(self.path2EventsData).exists:
            return False
        ttl_ts = np.load(os.path.join(self.path2EventsData, "timestamps.npy"))
        states = np.load(os.path.join(self.path2EventsData, "states.npy"))
        recording_start_time = self._get_recording_start_time()
        self.ttl_data = {}
        if "StimControl_id" in kwargs.keys():
            stim_id = kwargs["StimControl_id"]
            if stim_id in self.settings.processors.keys():
                duration = getattr(
                    self.settings.processors[stim_id], "Duration")
            else:
                return False
            self.ttl_data["stim_duration"] = int(duration)
        if "TTL_channel_number" in kwargs.keys():
            chan = kwargs["TTL_channel_number"]
            high_ttl = ttl_ts[states == chan]
            # get into seconds
            high_ttl = (high_ttl * 1000.0) - recording_start_time
            self.ttl_data["ttl_timestamps"] = high_ttl / \
                1000.0  # in seconds now
        if "RippleDetector" in args:
            if self.path2RippleDetector:
                detector_settings = self.settings.get_processor("Ripple")
                self.ttl_data = detector_settings.load_ttl(
                    self.path2RippleDetector, self.recording_start_time
                )
        if not self.ttl_data:
            return False
        print("Loaded ttl data")
        return True

    def load_accelerometer(self, target_freq: int = 50) -> bool:
        if not self.path2LFPdata:
            return False
        """
        Need to figure out which of the channels are AUX if we want to load
        the accelerometer data with minimal user input...
        Annoyingly, there could also be more than one RecordNode which means
        the channels might get represented more than once in the structure.oebin
        file

        Parameters
        ----------
        target_freq : int
            the desired frequency when downsampling the aux data

        Returns
        -------
        bool
            whether the data was loaded or not
        """
        from ephysiopy.openephys2py.OESettings import OEStructure
        from ephysiopy.common.ephys_generic import downsample_aux

        oebin = OEStructure(self.pname)
        aux_chan_nums = []
        aux_bitvolts = 0
        for record_node_key in oebin.data.keys():
            for channel_key in oebin.data[record_node_key].keys():
                # this thing is a 1-item list
                if "continuous" in channel_key:
                    for chan_keys in oebin.data[record_node_key][channel_key][0]:
                        for chan_idx, i_chan in enumerate(
                            oebin.data[record_node_key][channel_key][0]["channels"]
                        ):
                            if "AUX" in i_chan["channel_name"]:
                                aux_chan_nums.append(chan_idx)
                                aux_bitvolts = i_chan["bit_volts"]

        if len(aux_chan_nums) > 0:
            aux_chan_nums = np.unique(np.array(aux_chan_nums))
            if self.path2LFPdata is not None:
                data = memmapBinaryFile(
                    os.path.join(self.path2LFPdata, "continuous.dat"),
                    n_channels=self.channel_count,
                )
                s = slice(min(aux_chan_nums), max(aux_chan_nums) + 1)
                aux_data = data[s, :]
                # now downsample the aux data a lot
                # might take a while so print a message to console
                print(
                    f"Downsampling {aux_data.shape[1]} samples over {aux_data.shape[0]} channels..."
                )
                aux_data = downsample_aux(aux_data, target_freq=target_freq)
                self.aux_data = aux_data
                self.aux_data_fs = target_freq
                self.aux_bitvolts = aux_bitvolts
                return True
        else:
            warnings.warn(
                "No AUX data found in structure.oebin file, so not loaded")
        return False

    def get_waveforms(self, cluster: int | list, channel: int | list, *args, **kwargs):
        pass

    def apply_filter(self, *trial_filter: TrialFilter) -> np.ndarray:
        mask = super().apply_filter(*trial_filter)
        return mask

    def find_files(
        self,
        pname_root: str | Path,
        experiment_name: str = "experiment1",
        rec_name: str = "recording1",
        **kwargs,
    ):
        exp_name = Path(experiment_name)
        PosTracker_match = (
            exp_name / rec_name / "events" / "*Pos_Tracker*/BINARY_group*"
        )
        TrackingPlugin_match = (
            exp_name / rec_name / "events" / "*Tracking_Port*/BINARY_group*"
        )
        TrackMe_match = (
            exp_name / rec_name / "continuous" /
            "TrackMe-[0-9][0-9][0-9].TrackingNode"
        )
        RippleDetector_match = (
            exp_name / rec_name / "events" / "Ripple_Detector*" / "TTL"
        )
        sync_file_match = exp_name / rec_name
        acq_method = ""
        if self.rec_kind == RecordingKind.NEUROPIXELS:
            # the old OE NPX plugins saved two forms of the data,
            # one for AP @30kHz and one for LFP @??Hz
            # the newer plugin saves only the 30kHz data. Also, the
            # 2.0 probes are saved with Probe[A-Z] appended to the end
            # of the folder
            # the older way:
            acq_method = "Neuropix-PXI-[0-9][0-9][0-9]."
            APdata_match = exp_name / rec_name / \
                "continuous" / (acq_method + "0")
            LFPdata_match = exp_name / rec_name / \
                "continuous" / (acq_method + "1")
            # the new way:
            Rawdata_match = (
                exp_name / rec_name / "continuous" /
                (acq_method + "Probe[A-Z]")
            )
        elif self.rec_kind == RecordingKind.FPGA:
            acq_method = "Rhythm_FPGA-[0-9][0-9][0-9]."
            APdata_match = exp_name / rec_name / \
                "continuous" / (acq_method + "0")
            LFPdata_match = exp_name / rec_name / \
                "continuous" / (acq_method + "1")
            Rawdata_match = (
                exp_name / rec_name / "continuous" /
                (acq_method + "Probe[A-Z]")
            )
        else:
            acq_method = "Acquisition_Board-[0-9][0-9][0-9].*"
            APdata_match = exp_name / rec_name / "continuous" / acq_method
            LFPdata_match = exp_name / rec_name / "continuous" / acq_method
            Rawdata_match = (
                exp_name / rec_name / "continuous" /
                (acq_method + "Probe[A-Z]")
            )
        Events_match = (
            # only dealing with a single TTL channel at the moment
            exp_name
            / rec_name
            / "events"
            / acq_method
            / "TTL"
        )

        if pname_root is None:
            pname_root = self.pname_root

        verbose = kwargs.get("verbose", False)

        for d, c, f in os.walk(pname_root):
            for ff in f:
                if "." not in c:  # ignore hidden directories
                    if "data_array.npy" in ff:
                        if PurePath(d).match(str(PosTracker_match)):
                            if self.path2PosData is None:
                                self.path2PosData = os.path.join(d)
                                if verbose:
                                    print(
                                        f"Pos data at: {self.path2PosData}\n")
                            self.path2PosOEBin = Path(d).parents[1]
                        if PurePath(d).match("*pos_data*"):
                            if self.path2PosData is None:
                                self.path2PosData = os.path.join(d)
                                if verbose:
                                    print(
                                        f"Pos data at: {self.path2PosData}\n")
                        if PurePath(d).match(str(TrackingPlugin_match)):
                            if self.path2PosData is None:
                                self.path2PosData = os.path.join(d)
                                if verbose:
                                    print(
                                        f"Pos data at: {self.path2PosData}\n")
                    if "continuous.dat" in ff:
                        if PurePath(d).match(str(APdata_match)):
                            self.path2APdata = os.path.join(d)
                            if verbose:
                                print(
                                    f"Continuous AP data at: {self.path2APdata}\n")
                            self.path2APOEBin = Path(d).parents[1]
                        if PurePath(d).match(str(LFPdata_match)):
                            self.path2LFPdata = os.path.join(d)
                            if verbose:
                                print(
                                    f"Continuous LFP data at: {self.path2LFPdata}\n")
                        if PurePath(d).match(str(Rawdata_match)):
                            self.path2APdata = os.path.join(d)
                            self.path2LFPdata = os.path.join(d)
                        if PurePath(d).match(str(TrackMe_match)):
                            self.path2PosData = os.path.join(d)
                            if verbose:
                                print(
                                    f"TrackMe posdata at: {self.path2PosData}\n")
                    if "sync_messages.txt" in ff:
                        if PurePath(d).match(str(sync_file_match)):
                            sync_file = os.path.join(d, "sync_messages.txt")
                            if fileContainsString(sync_file, "Start Time"):
                                self.sync_message_file = sync_file
                                if verbose:
                                    print(
                                        f"sync_messages file at: {sync_file}\n")
                    if "full_words.npy" in ff:
                        if PurePath(d).match(str(Events_match)):
                            self.path2EventsData = os.path.join(d)
                            if verbose:
                                print(
                                    f"Event data at: {self.path2EventsData}\n")
                        if PurePath(d).match(str(RippleDetector_match)):
                            self.path2RippleDetector = os.path.join(d)
                            if verbose:
                                print(
                                    f"Ripple Detector plugin found at {self.path2RippleDetector}\n"
                                )
                    if ".nwb" in ff:
                        self.path2NWBData = os.path.join(d, ff)
                        if verbose:
                            print(f"nwb data at: {self.path2NWBData}\n")
                    if "spike_templates.npy" in ff:
                        self.path2KiloSortData = os.path.join(d)
                        if verbose:
                            print(
                                f"Found KiloSort data at {self.path2KiloSortData}\n")


class OpenEphysNWB(OpenEphysBase):
    def __init__(self, pname: Path, **kwargs) -> None:
        pname = Path(pname)
        super().__init__(pname, **kwargs)

    def load_neural_data(self, *args, **kwargs) -> None:
        pass

    def load_settings(self, *args, **kwargs):
        return super().load_settings()

    def load_pos_data(
        self, ppm: int = 300, jumpmax: int = 100, *args, **kwargs
    ) -> None:
        with h5py.File(os.path.join(self.path2NWBData), mode="r") as nwbData:
            xy = np.array(nwbData[self.path2PosData + "/data"])
            xy = xy[:, 0:2]
            ts = np.array(nwbData[self.path2PosData]["timestamps"])
            P = PosCalcsGeneric(xy[0, :], xy[1, :],
                                cm=True, ppm=ppm, jumpmax=jumpmax)
            P.xyTS = ts
            P.sample_rate = 1.0 / np.mean(np.diff(ts))
            P.postprocesspos()
            print("Loaded pos data")
            self.PosCalcs = P

    def find_files(
        self,
        experiment_name: str = "experiment_1",
        recording_name: str = "recording0",
    ):
        super().find_files(
            self.pname,
            experiment_name,
            recording_name,
            RecordingKind.NWB,
        )
