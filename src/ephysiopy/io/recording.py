import os
import re
from typing import OrderedDict, override
import warnings
from pathlib import Path, PurePath
import numpy as np
from scipy import signal
from phylib.io.model import TemplateModel

from ephysiopy.axona.axonaIO import IO, Pos
from ephysiopy.axona.tetrode_dict import TetrodeDict
from ephysiopy.common.ephys_generic import (
    EEGCalcsGeneric,
    PosCalcsGeneric,
)
from ephysiopy.openephys2py.OESettings import Settings, SyncMessages
from ephysiopy.openephys2py.KiloSort import KiloSortSession
from ephysiopy.common.utils import (
    TrialFilter,
    memmapBinaryFile,
    fileContainsString,
    RecordingKind,
    Xml2RecordingKind,
)
from ephysiopy.io.bases import TrialInterface
from ephysiopy.openephys2py.raw_data import get_raw_cluster_spikes


class AxonaTrial(TrialInterface):
    def __init__(self, pname: Path, **kwargs) -> None:
        use_volts = kwargs.pop("volts", True)
        pname = Path(pname)
        kwargs["rec_kind"] = RecordingKind.AXONA
        super().__init__(pname, **kwargs)
        self._rec_kind = RecordingKind.AXONA
        self._settings = None
        self.TETRODE = TetrodeDict(str(self.pname.with_suffix("")), volts=use_volts)
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

                new_AxonaTrial.concatenated_trials = [self.pname, other.pname]
                new_AxonaTrial.concatenated = True
                return new_AxonaTrial

        else:
            raise TypeError("Can only add AxonaTrial instances")

    def load_lfp(self, *args, **kwargs):
        from ephysiopy.axona.axonaIO import EEG

        if not self.concatenated:
            if "target_sample_rate" in kwargs.keys():
                lfp = EEG(self.pname, egf=1)
                if lfp is None:  # drop down to eeg (250Hz)
                    lfp = EEG(self.pname)

                target_sample_rate = kwargs.get("target_sample_rate", 250)
                denom = np.gcd(int(target_sample_rate), int(lfp.sample_rate))
                data = lfp.sig
                sig = signal.resample_poly(
                    data.astype(float),
                    target_sample_rate / denom,
                    lfp.sample_rate / denom,
                    0,
                )
                self.EEGCalcs = EEGCalcsGeneric(sig, target_sample_rate)
                return

            if "egf" in args:
                lfp = EEG(self.pname, egf=1)
            else:
                lfp = EEG(self.pname)
            if lfp is not None:
                self.EEGCalcs = EEGCalcsGeneric(lfp.sig, lfp.sample_rate)
        else:
            # concatenated so load the LFP data for each trial and concatenate
            lfp_data = []
            target_sample_rate = kwargs.get("target_sample_rate", 250)
            for trial in self.concatenated_trials:
                lfp = EEG(trial)
                if lfp is not None:
                    denom = np.gcd(int(target_sample_rate), int(lfp.sample_rate))

                    sig = signal.resample_poly(
                        lfp.sig.astype(float),
                        target_sample_rate / denom,
                        lfp.sample_rate / denom,
                        0,
                    )
                    lfp_data.append(sig)
            if lfp_data:
                sig = np.concatenate(lfp_data)
                self.EEGCalcs = EEGCalcsGeneric(sig, target_sample_rate)

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
        pattern = re.compile(str(self.pname.name).replace(".set", ".[0-9].cut"))
        cuts = sorted(
            [Path(f) for f in os.listdir(self.pname.parent) if pattern.match(f)]
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
                        try:
                            clusters.remove(0)
                        except ValueError:
                            pass
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
                    x=AxonaPos.led_pos[:, 0],
                    y=AxonaPos.led_pos[:, 1],
                    cm=True,
                    ppm=ppm,
                    jumpmax=jumpmax,
                    bad_indices=AxonaPos.bad_idx,
                )
                P.sample_rate = AxonaPos.getHeaderVal(AxonaPos.header, "sample_rate")
                P.time = np.ma.MaskedArray(
                    AxonaPos.ts / P.sample_rate
                )  # in seconds now
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
            if cluster is not None:
                if isinstance(cluster, int):
                    return self.TETRODE.get_spike_samples(int(tetrode), int(cluster))

                elif isinstance(cluster, list) and isinstance(tetrode, list):
                    if len(cluster) == 1:
                        tetrode = tetrode[0]
                        cluster = cluster[0]
                        return self.TETRODE.get_spike_samples(
                            int(tetrode), int(cluster)
                        )
                    else:
                        spikes = []
                        for tc in zip(tetrode, cluster):
                            spikes.append(self.TETRODE.get_spike_samples(tc[0], tc[1]))
                        return spikes

            else:
                # return all spike times
                return self.TETRODE.get_all_spike_timestamps(tetrode)

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

    file_path_list = [
        "path2APdata",
        "path2LFPdata",
        "path2APOEBin",
        "path2PosData",
        "path2EventsData",
        "path2KiloSortData",
        "path2NWBData",
        "path2SyncMessages",
    ]

    def __init__(self, pname: Path, **kwargs) -> None:

        pname = Path(pname)
        super().__init__(pname, **kwargs)

        self.path2APdata = []
        self.path2LFPdata = []
        self.path2APOEBin = []
        self.path2PosData = []
        self.path2EventsData = []
        self.path2KiloSortData = []
        self.path2NWBData = []
        self.path2SyncMessages = []

        self.load_settings()

        if self.rec_kind.value == RecordingKind.UNKNOWN.value:
            rec_method = [
                re.search(m, k).string
                for k in self.settings.processors.keys()
                for m in self.record_methods
                if re.search(m, k) is not None
            ][0]
            if "Sources/" in rec_method:
                rec_method = rec_method.lstrip("Sources/")

            self.rec_kind = Xml2RecordingKind[rec_method.rpartition(" ")[0]]

        # get the recording plugin
        rec_plugin_name = list(Xml2RecordingKind.keys())[
            list(Xml2RecordingKind.values()).index(self.rec_kind)
        ]
        self.rec_plugin = self.settings.get_plugin(rec_plugin_name)

        self.cluster_data_id = None
        self.lfp_data_id = None

    def _get_sync_message_info_(self, sync_message_file: str) -> list[SyncMessages]:
        """
        Extract plugin information from the sync_message_file.txt.

        Parameters
        ----------
        sync_message_file - str
            The location of the file on disk

        Returns
        -------
        tuple(float, str, str)
            The start of the recording time for the plugin
        """
        sync_list = []
        sync_info = SyncMessages()
        if sync_message_file is not None:
            with open(sync_message_file, "r") as f:
                sync_strs = f.read()
            sync_lines = sync_strs.split("\n")
            for line in sync_lines:
                if "Start Time" in line:
                    tokens = line.split(":")
                    # get the time in samples...
                    start_time = int(tokens[-1])
                    # ... and the sample rate...
                    sample_rate = int(tokens[0].split("@")[-1].strip().split()[0])
                    # ... convert to seconds...
                    recording_start_time = start_time / float(sample_rate)
                    # ... get the plugin name
                    idx0 = tokens[0].find("for")
                    idx1 = tokens[0].find(") -")
                    pluginName = tokens[0][idx0 + 4 : idx1 + 1]
                    # ... and the stream name
                    idx2 = tokens[0].find("@")
                    stream_name = tokens[0][idx1 + 4 : idx2 - 1]
                    # breakpoint()
                    if sync_info.pluginName != pluginName:
                        sync_info = SyncMessages()
                        sync_info.pluginName = pluginName
                    sync_info.streams.append(stream_name)
                    sync_info.sample_rates.append(sample_rate)
                    sync_info.start_times.append(recording_start_time)
                    sync_list.append(sync_info)

        return list(set(sync_list))

    def _get_start_time_from_stream_name_(self, stream_name: str) -> float:
        """
        Find the start time of the recording from the sync_info that corresponds
        to the events stream that contains the TTL data.

        stream_name - str
            The stream name to search for

        Parameters
        ----------
        stream_name - str
            Name of the stream e.g. "ProbeA-AP"

        Notes
        -----
        The search is case-insensitve and will look through all
        sync_messages.txt files found in the recording folder hierarchy
        """
        if self.path2SyncMessages is not None:
            for s_path in self.path2SyncMessages:
                sync_info = self._get_sync_message_info_(s_path)
                for s in sync_info:
                    for stream in s.streams:
                        if stream.lower().find(stream_name.lower()) != -1:
                            return s.start_times[s.streams.index(stream)]
        return 0.0

    def _get_ttl_times_from_stream_name_(self, stream_name: str) -> np.ndarray:
        """
        Find the TTL times from the sync_info that corresponds
        to the events stream that contains the TTL data.

        Parameters
        ----------
        stream_name - str
            Name of the stream e.g. "ProbeA-AP"
        """
        if self.path2SyncMessages is not None:
            for ev in self.path2EventsData:
                if stream_name in str(ev):
                    ts = np.load(ev / Path("timestamps.npy"))
                    states = np.load(ev / Path("states.npy"))
                    return ts[states == 1]
        return np.array([])

    def get_spike_times(
        self,
        cluster: int | list | None = None,
        tetrode: int | list | None = None,
        **kws,
    ) -> list | np.ndarray:

        if isinstance(cluster, int) and isinstance(tetrode, int):
            if cluster in self.clusterData.cluster_id:
                mask = np.invert(self.clusterData.spike_clusters.mask)
                idx = self.clusterData.spike_clusters == cluster

                return np.ravel(
                    self.clusterData.spike_times[np.ma.logical_and(mask, idx)]
                    / self.clusterData.sample_rate
                )

        elif isinstance(cluster, list) and isinstance(tetrode, list):
            times = []
            for c in cluster:
                if c in self.clusterData.cluster_id:
                    t = np.ravel(
                        self.clusterData.spike_times[
                            self.clusterData.spike_clusters == cluster
                        ]
                        / self.clusterData.sample_rate
                    )
                    times.append(t)
                else:
                    warnings.warn("Cluster not present")
            return times

        return []

    def load_lfp(self, *args, **kwargs):

        def __load_memmap__(l_path: Path | None, n_chans: int):
            if l_path is None:
                return None
            for pname in self.path2LFPdata:
                if l_path in str(pname):
                    lfp = memmapBinaryFile(
                        os.path.join(pname, "continuous.dat"), n_channels=n_chans
                    )
                    self.lfp_data_id = l_path

                    lfp_times = None
                    if Path(self.path2LFPdata[0] / Path("timestamps.npy")).exists():
                        lfp_times = np.load(
                            self.path2LFPdata[0] / Path("timestamps.npy")
                        )

                    return lfp, lfp_times

        lfp_data_id = kwargs.get("stream_name", None)
        channel = kwargs.get("channel", 0)
        target_sample_rate = kwargs.get("target_sample_rate", 250)

        lfp = None
        times = None

        if lfp_data_id is None:
            n_chans = self.rec_plugin.get_streams()[0].channel_count
            lfp, times = __load_memmap__(self.path2LFPdata[0], n_chans)

        elif lfp_data_id != self.lfp_data_id:
            n_chans = self.rec_plugin.get_stream(lfp_data_id).channel_count
            sample_rate = self.rec_plugin.get_stream(lfp_data_id).sample_rate
            lfp, times = __load_memmap__(lfp_data_id, n_chans)

        elif lfp_data_id == self.lfp_data_id:
            if channel == self.lfp_channel:
                return self.EEGCalcs
            else:
                n_chans = self.rec_plugin.get_stream(lfp_data_id).channel_count
                sample_rate = self.rec_plugin.get_stream(lfp_data_id).sample_rate
                lfp, times = __load_memmap__(lfp_data_id, n_chans)

        self.lfp_channel = channel

        # set the target sample rate to 250Hz by default to match
        # Axona EEG data
        sample_rate = self.rec_plugin.get_stream(self.lfp_data_id).sample_rate
        denom = np.gcd(int(target_sample_rate), int(sample_rate))
        data = lfp[channel, :]
        sig = signal.resample_poly(
            data.astype(float),
            target_sample_rate / denom,
            sample_rate / denom,
            0,
        )
        # resample times to match size of sig
        if times is not None:
            factor = np.round(times.shape[0] / sig.shape[0]).astype(int)
            times = times[::factor]

        self.EEGCalcs = EEGCalcsGeneric(sig, target_sample_rate)
        self.EEGCalcs.time = times
        return self.EEGCalcs

    def load_neural_data(self, *args, **kwargs):
        # match to rec_kind and get the channel count from the stream
        # check kwargs for which probe to load
        if "path2APdata" in kwargs.keys():
            self.path2APdata: Path = Path(kwargs["path2APdata"])
        n_channels: int = self.channel_count or kwargs["nChannels"]
        kilo_index = kwargs.get("kilo_index", 0)
        try:
            self.template_model = TemplateModel(
                dir_path=self.path2KiloSortData[kilo_index],
                sample_rate=self.sample_rate,
                dat_path=Path(self.path2KiloSortData[kilo_index]).joinpath(
                    "continuous.dat"
                ),
                n_channels_dat=int(n_channels),
            )
            print("Loaded neural data")
        except Exception:
            warnings.warn("Could not find raw data file")

    def load_settings(self, *args, **kwargs):
        """
        Load the settings.xml file associated with the recording
        """
        if self._settings is None:
            # pname_root gets walked through and over-written with
            # correct location of settings.xml
            self.settings = Settings(self.pname)
            print("Loaded settings data\n")

    def load_cluster_data(self, **kws) -> OrderedDict | None:
        """
        Load the cluster data ideally for a probe specified by
        its stream name/ identity

        kws - dict
            stream_name - str
            e.g. stream_name="ProbeA-AP"

        Notes
        -----
        This is mainly here to maintain function name consistency
        with the other load_* methods in this class
        """
        return self.get_available_clusters_channels(**kws)

    def get_available_clusters_channels(self, **kws) -> OrderedDict | None:
        """
        Get available clusters and their corresponding channels.

        Parameters
        ----------
        kws - dict
            Valid keys:
                "stream_name" - str
                    The stream name that corresponds to a KiloSort
                    dataset. An example would be "ProbeA-AP".
                    If not available then the first stream
                    in self.rec_plugin is taken

        Returns
        -------
        dict
            A dict where keys are channels and values are lists of clusters
        """

        def __load_kilo__(s_name, **kws) -> OrderedDict | None:
            stream_name = kws.get("stream_name", None)
            start_time = 0.0
            if stream_name:
                start_time = self._get_start_time_from_stream_name_(stream_name)
            for ks_path in self.path2KiloSortData:
                if str(s_name) in str(ks_path):
                    print(f"Loading Kilosort data from: {ks_path}")
                    K = KiloSortSession(ks_path)
                    K.load()
                    K.get_all_channels_clusters()
                    K.recording_start_time = start_time
                    self.clusterData = K
                    self.cluster_data_id = s_name
                    return self.clusterData.channels_clusters
            return None

        cluster_data_id = kws.get("stream_name", None)

        # If no stream name is provided, use the first KiloSort data path
        if cluster_data_id is None:
            cluster_data_id = self.path2KiloSortData[0]
            __load_kilo__(str(cluster_data_id), **kws)

        if self.cluster_data_id is None:
            __load_kilo__(str(cluster_data_id), **kws)

        if cluster_data_id == self.cluster_data_id:
            return self.clusterData.channels_clusters
        else:
            __load_kilo__(cluster_data_id, **kws)

    def load_pos_data(
        self, ppm: int = 300, jumpmax: int = 100, *args, **kwargs
    ) -> None:
        # kwargs valid keys = "loadTTLPos" - if present loads the ttl
        # timestamps not the ones in the plugin folder

        cm = kwargs.get("cm", True)

        recording_start_time = 0.0

        if self.path2PosData is not None:
            # figure out which tracking method is being used
            pos_method = [
                "Pos Tracker [0-9][0-9][0-9]",
                "PosTracker [0-9][0-9][0-9]",
                "TrackMe [0-9][0-9][0-9]",
                "TrackingPlugin [0-9][0-9][0-9]",
                "Tracking Port",
                "Trackerizer [0-9][0-9][0-9]",
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

            tracker = self.settings.get_plugin(pos_plugin_name)
            sample_rate = tracker.stream[0].sample_rate

            # Load the positional (x/y) data...
            if isinstance(self.path2PosData, list):
                pos_data = tracker.load(self.path2PosData[0])
            else:
                pos_data = tracker.load(self.path2PosData)

            # ... need to be careful about timnestamps as OE automatically
            # creates a timestamps.npy file based on the claimed sample rate
            # each plugin declares. For position tracking this might well be
            # different to what is declared in the c/c++ files and so the
            # timestamps.npy file is not accurate. We need to link together
            # the "real" timestamp data and the position data - search through
            # the list of TTL event data and load the appropraite one...
            # appropriate one here is the one from the acquisition board so
            # search through the list of events paths and pick it out

            pos_ts = None
            recording_start_time = 0.0

            if "Trackerizer" in pos_plugin_name:
                pos_ts = tracker.load_times(Path(self.path2PosData[0]))

            if "TrackMe" in pos_plugin_name:
                if "loadTTLPos" in kwargs.keys():
                    pos_ts = tracker.load_ttl_times(Path(self.path2EventsData[0]))
                else:
                    if isinstance(self.path2PosData, list):
                        pos_ts = tracker.load_times(Path(self.path2PosData[0]))
                    else:
                        pos_ts = tracker.load_times(Path(self.path2PosData))

                pos_ts = pos_ts[0 : len(pos_data)]

            # sample_rate = float(sample_rate) if sample_rate is not None else 50
            # the timestamps for the Tracker Port plugin are fucked so
            # we have to infer from the shape of the position data
            if "Tracking Port" in pos_plugin_name:
                sample_rate = kwargs.get("sample_rate", 50)
                # pos_ts in seconds
                pos_ts = np.arange(
                    0, pos_data.shape[0] / sample_rate, 1.0 / sample_rate
                )
            # a bit knarly...
            if "TrackMe" not in pos_plugin_name:
                if "Trackerizer" not in pos_plugin_name:
                    time = pos_ts + recording_start_time
                else:
                    time = pos_ts
            else:
                time = pos_ts
            if self.path2SyncMessages is not None:
                recording_start_time = time[0]

            # This is the gateway to all the position processing so if you want
            # to load your own pos data you'll need to create an instance of
            # PosCalcsGeneric yourself and apply it to self
            P = PosCalcsGeneric(
                pos_data[:, 0],
                pos_data[:, 1],
                cm=cm,
                ppm=ppm,
                jumpmax=jumpmax,
                bad_indices=None,
            )
            P.time = np.ma.MaskedArray(time)
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
        """
        Load the TTL data.

        Parameters
        ----------
        kwargs - optional dict with valid keys:

            StimConrol_id - specifies which StimControl to load
                as it's possible there are more than one (one to trigger the
                camera and another for other experimental control e.g. a laser)
            "TTL_channel_number" - which TTL channel to load as high

        args - optional arguments:

        "RippleDetector" - this plugin sends out TTL pulses when it detects
                ripples and has a special method defined for loading that data
                (see ephysiopy.openephys2py.OESettings.RippleDetector)

        Returns
        -------
        dict - keys are processor name (the directory name above the TTL part)
               items are the ttl data

        Notes
        -----
        the attribute self.path2EventsData is a list of directories titled
        "TTL" and contains a full_words.npy file. Some of the files might be
        empty
        """

        for ttl_path in self.path2EventsData:
            if ttl_path.exists():
                ttl_ts = np.load(ttl_path / Path("timestamps.npy"))
                states = np.load(ttl_path / Path("states.npy"))
                if len(ttl_ts) > 0:
                    if "StimControl_id" in kwargs.keys():
                        # TODO: I think this is mostly obsolete
                        # as the TTL data is saved automatically
                        # in whatever RecordNode
                        stim_id = kwargs["StimControl_id"]
                        if stim_id in self.settings.processors.keys():
                            # returned in ms from the plugin so convert to seconds...
                            plugin = self.settings.get_plugin(stim_id)
                            duration = float(plugin.Duration) / 1000.0  # in seconds
                        else:
                            return False

                        if not self.ttl_data:
                            self.ttl_data = {}

                        self.ttl_data["stim_duration"] = duration

                    # recording_start_time = self._get_recording_start_time()
                    if "TTL_channel_number" in kwargs.keys():
                        chan = kwargs["TTL_channel_number"]
                        high_ttl = ttl_ts[states == chan]
                        # get into seconds
                        high_ttl = (high_ttl * 1000.0) - self.recording_start_time
                        self.ttl_data["ttl_timestamps"] = (
                            high_ttl / 1000.0
                        )  # in seconds now
        if "RippleDetector" in args:
            if self.path2RippleDetector:
                detector_settings = self.settings.get_plugin("Ripple")
                self.ttl_data = detector_settings.load_ttl(
                    self.path2RippleDetector[0], self.recording_start_time
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
                    f"""Downsampling {aux_data.shape[1]} samples over {
                        aux_data.shape[0]
                    } channels..."""
                )
                aux_data = downsample_aux(aux_data, target_freq=target_freq)
                self.aux_data = aux_data
                self.aux_data_fs = target_freq
                self.aux_bitvolts = aux_bitvolts
                return True
        else:
            warnings.warn("No AUX data found in structure.oebin file, so not loaded")
        return False

    def get_waveforms(
        self, cluster: int | list, channel: int | list, *args, **kwargs
    ) -> np.ndarray | list:
        """
        Gets the waveforms for the specified cluster(s).
        Ignores the channel input and instead returns the waveforms
        for the four "best" channels for the cluster.
        """
        self.bit_volts = 0.1949999928474426  # hard-coded for now

        if not self.template_model:
            self.load_neural_data()

        if "from_raw" in kwargs.keys() and kwargs["from_raw"]:
            waveforms = get_raw_cluster_spikes(self, cluster, **kwargs)
            # axis 0 and 1 need swapping to get into
            # (n_spikes, n_channel, n_samples) format
            return np.swapaxes(waveforms, 0, 1)

        if isinstance(cluster, int):
            spike_ids = self.template_model.get_cluster_spikes(int(cluster))
            channels = self.template_model.get_cluster_channels(int(cluster))
            channels = channels[0:4]  # get the top 4 channels
            w = self.template_model.get_waveforms(spike_ids, channels) * self.bit_volts
            # swap to (n_spikes, n_channel, n_samples)
            return np.swapaxes(w, -1, 1)
        elif isinstance(cluster, list):
            waveforms = []
            for c in cluster:
                spike_ids = self.template_model.get_cluster_spikes(int(cluster))
                channels = self.template_model.get_cluster_channels(int(cluster))
                channels = channels[0:4]  # get the top 4 channels
                w = (
                    self.template_model.get_waveforms(spike_ids, channels)
                    * self.bit_volts
                )
                waveforms.append(
                    np.swapaxes(w, -1, 1)
                )  # swap to (n_spikes, n_channel, n_samples)
            return waveforms

    def apply_filter(self, *trial_filter: TrialFilter) -> np.ndarray:
        mask = super().apply_filter(*trial_filter)
        return mask

    def find_files(
        self,
        pname_root: str | Path,
        experiment_name: str = "experiment1",
        rec_name: str = "recording1",
        rec_kind: RecordingKind = RecordingKind.NEUROPIXELS,
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
            exp_name / rec_name / "continuous" / "TrackMe-[0-9][0-9][0-9].TrackingNode"
        )
        Trackerizer_match = (
            exp_name / rec_name / "continuous" / "Trackerizer-[0-9][0-9][0-9].Tracking"
        )
        RippleDetector_match = (
            exp_name / rec_name / "events" / "Ripple_Detector*" / "TTL"
        )
        sync_file_match = exp_name / rec_name
        acq_method = ""
        if rec_kind == RecordingKind.NEUROPIXELS:
            # the old OE NPX plugins saved two forms of the data,
            # one for AP @30kHz and one for LFP @??Hz
            # the newer plugin saves only the 30kHz data. Also, the
            # 2.0 probes are saved with Probe[A-Z] appended to the end
            # of the folder
            # the older way:
            acq_method = "Neuropix-PXI-[0-9][0-9][0-9]."
            APdata_match = exp_name / rec_name / "continuous" / (acq_method + "*AP")
            LFPdata_match = exp_name / rec_name / "continuous" / (acq_method + "*LFP")
            # the new way:
            Rawdata_match = (
                exp_name / rec_name / "continuous" / (acq_method + "Probe[A-Z]")
            )
        elif rec_kind == RecordingKind.FPGA:
            acq_method = "Rhythm_FPGA-[0-9][0-9][0-9]."
            APdata_match = exp_name / rec_name / "continuous" / (acq_method + "0")
            LFPdata_match = exp_name / rec_name / "continuous" / (acq_method + "1")
            Rawdata_match = (
                exp_name / rec_name / "continuous" / (acq_method + "Probe[A-Z]")
            )
        else:
            acq_method = "Acquisition_Board-[0-9][0-9][0-9].*"
            APdata_match = exp_name / rec_name / "continuous" / acq_method
            LFPdata_match = exp_name / rec_name / "continuous" / acq_method
            Rawdata_match = (
                exp_name / rec_name / "continuous" / (acq_method + "Probe[A-Z]")
            )
        Events_match = (
            # only dealing with a single TTL channel at the moment
            exp_name / rec_name / "events" / "*/TTL"
        )

        for d, c, f in os.walk(pname_root):
            for ff in f:
                if "." not in c:  # ignore hidden directories
                    if "data_array.npy" in ff:
                        if PurePath(d).match(str(PosTracker_match)):
                            if self.path2PosData is None:
                                self.path2PosData = os.path.join(d)
                            self.path2PosOEBin = Path(d).parents[1]

                        if PurePath(d).match("*pos_data*"):
                            if self.path2PosData is None:
                                self.path2PosData = Path(os.path.join(d))

                        if PurePath(d).match(str(TrackingPlugin_match)):
                            if self.path2PosData is None:
                                self.path2PosData = Path(os.path.join(d))

                    if "continuous.dat" in ff:
                        if PurePath(d).match(str(APdata_match)):
                            self.path2APdata.append(os.path.join(d))
                            self.path2APOEBin.append(Path(d).parents[1])

                        if PurePath(d).match(str(LFPdata_match)):
                            self.path2LFPdata.append(Path(os.path.join(d)))

                        if PurePath(d).match(str(Rawdata_match)):
                            self.path2APdata.append(os.path.join(d))
                            self.path2LFPdata.append(Path(os.path.join(d)))

                        if PurePath(d).match(str(TrackMe_match)):
                            self.path2PosData.append(Path(os.path.join(d)))

                        if PurePath(d).match(str(Trackerizer_match)):
                            self.path2PosData.append(Path(os.path.join(d)))

                    if "sync_messages.txt" in ff:
                        if PurePath(d).match(str(sync_file_match)):
                            sync_file = os.path.join(d, "sync_messages.txt")
                            if fileContainsString(sync_file, "Start Time"):
                                self.path2SyncMessages.append(Path(sync_file))

                    if "full_words.npy" in ff:
                        if PurePath(d).match(str(Events_match)):
                            if len(np.load(Path(d) / Path("full_words.npy"))) > 0:
                                self.path2EventsData.append(Path(os.path.join(d)))

                        if PurePath(d).match(str(RippleDetector_match)):
                            self.path2RippleDetector.append(Path(os.path.join(d)))

                    if ".nwb" in ff:
                        self.path2NWBData.append(Path(os.path.join(d, ff)))

                    if "params.py" in ff:
                        self.path2KiloSortData.append(Path(os.path.join(d)))

        if kwargs.get("verbose", False):
            for attr in self.file_path_list:
                if hasattr(self, attr):
                    a = getattr(self, attr)
                    if a:
                        if isinstance(a, list):
                            print(f"{attr}:")
                            [print(f"  {i}") for i in a]
                        else:
                            print(f"{attr}: {a}")


class OpenEphysFPGA(OpenEphysBase):
    def __init__(self, pname: Path, **kwargs) -> None:

        pname = Path(pname)

        self.rec_kind = RecordingKind.FPGA
        kwargs["rec_kind"] = self.rec_kind

        super().__init__(pname, **kwargs)

        self.find_files(pname, **kwargs)

        self.template_model = None


class OpenEphysAcqBoard(OpenEphysBase):
    def __init__(self, pname: Path, **kwargs) -> None:

        pname = Path(pname)

        self.rec_kind = RecordingKind.ACQUISITIONBOARD
        kwargs["rec_kind"] = self.rec_kind
        super().__init__(pname, **kwargs)

        self.find_files(pname, **kwargs)

        acq_plugin = self.settings.get_plugin("Acquisition Board")
        self.channel_count = acq_plugin.stream[0].channel_count
        self.sample_rate = acq_plugin.stream[0].sample_rate

        self.template_model = None

    @override
    def load_lfp(self, *args, **kwargs):

        lfp_index = kwargs.get("lfp_index", 0)

        if self.path2LFPdata is not None:
            lfp = memmapBinaryFile(
                os.path.join(self.path2LFPdata[lfp_index], "continuous.dat"),
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


class OpenEphysNPX(OpenEphysBase):
    def __init__(self, pname: Path, **kwargs) -> None:
        pname = Path(pname)

        self.rec_kind = RecordingKind.NEUROPIXELS
        kwargs["rec_kind"] = self.rec_kind

        super().__init__(pname, **kwargs)

        self.find_files(pname, **kwargs)

        self.template_model = None
