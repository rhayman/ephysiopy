import os
import re
import warnings
from pathlib import Path, PurePath
import h5py
import numpy as np
from scipy import signal
from phylib.io.model import TemplateModel

from ephysiopy.axona.axonaIO import IO, Pos
from ephysiopy.axona.tetrode_dict import TetrodeDict
from ephysiopy.common.ephys_generic import (
    EEGCalcsGeneric,
    PosCalcsGeneric,
)
from ephysiopy.openephys2py.OESettings import Settings
from ephysiopy.common.utils import (
    TrialFilter,
    memmapBinaryFile,
    fileContainsString,
    Xml2RecordingKind,
    RecordingKind,
)
from ephysiopy.io.bases import TrialInterface


class AxonaTrial(TrialInterface):
    def __init__(self, pname: Path, **kwargs) -> None:
        use_volts = kwargs.pop("volts", True)
        pname = Path(pname)
        super().__init__(pname, **kwargs)
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
                    AxonaPos.led_pos[:, 0],
                    AxonaPos.led_pos[:, 1],
                    cm=True,
                    ppm=ppm,
                    jumpmax=jumpmax,
                )
                P.sample_rate = AxonaPos.getHeaderVal(AxonaPos.header, "sample_rate")
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
                    sample_rate = int(tokens[0].split("@")[-1].strip().split()[0])
                    recording_start_time = start_time / float(sample_rate)
        self.recording_start_time = recording_start_time
        return recording_start_time

    def get_spike_times(
        self, cluster: int | list = None, tetrode: int | list = None, *args, **kwargs
    ) -> list | np.ndarray:
        if not self.template_model:
            self.load_neural_data()
        if isinstance(cluster, int) and isinstance(tetrode, int):
            if cluster in self.template_model.cluster_ids:
                times = self.template_model.spike_times[
                    self.template_model.spike_clusters == cluster
                ]
                return times
            else:
                warnings.warn("Cluster not present")
        elif isinstance(cluster, list) and isinstance(tetrode, list):
            times = []
            for c in cluster:
                if c in self.template_model.cluster_ids:
                    t = self.template_model.spike_times[
                        self.template_model.spike_clusters == cluster
                    ]
                    times.append(t)
                else:
                    warnings.warn("Cluster not present")
            return times

    def load_lfp(self, *args, **kwargs):
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
                sample_rate=self.sample_rate,
                dat_path=Path(self.path2KiloSortData).joinpath("continuous.dat"),
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

        unique_clusters = self.template_model.cluster_ids
        clust_chans = dict([(i, []) for i in range(self.template_model.n_channels)])

        for clust in unique_clusters:
            chan = self.template_model.get_cluster_channels(clust)[0].item()
            clust_chans[chan].append(clust.item())

        clean_dict = {k: v for k, v in clust_chans.items() if v}
        return clean_dict

    def load_cluster_data(self, removeNoiseClusters=True, *args, **kwargs) -> bool:
        warnings.warn("load_cluster_data is deprecated. Use load_neural_data instead.")

    def load_pos_data(
        self, ppm: int = 300, jumpmax: int = 100, *args, **kwargs
    ) -> None:
        # kwargs valid keys = "loadTTLPos" - if present loads the ttl
        # timestamps not the ones in the plugin folder

        # Only sub-class that doesn't use this is OpenEphysNWB
        # which needs updating
        # TODO: Update / overhaul OpenEphysNWB
        # Load the start time from the sync_messages file
        cm = kwargs.get("cm", True)

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
                pos_ts = pos_ts[0 : len(pos_data)]
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
                # returned in ms from the plugin so convert to seconds...
                duration = getattr(self.settings.processors[stim_id], "Duration")
                duration = float(duration) / 1000.0  # in seconds
            else:
                return False
            self.ttl_data["stim_duration"] = duration
        if "TTL_channel_number" in kwargs.keys():
            chan = kwargs["TTL_channel_number"]
            high_ttl = ttl_ts[states == chan]
            # get into seconds
            high_ttl = (high_ttl * 1000.0) - recording_start_time
            self.ttl_data["ttl_timestamps"] = high_ttl / 1000.0  # in seconds now
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

    def get_waveforms(self, cluster: int | list, channel: int | list, *args, **kwargs):
        """
        Gets the waveforms for the specified cluster(s). Ignores the channel input
        and instead returns the waveforms for the four "best" channels for the cluster.
        """
        self.bit_volts = 0.1949999928474426  # hard-coded for now
        if not self.template_model:
            self.load_neural_data()
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
            APdata_match = exp_name / rec_name / "continuous" / (acq_method + "0")
            LFPdata_match = exp_name / rec_name / "continuous" / (acq_method + "1")
            # the new way:
            Rawdata_match = (
                exp_name / rec_name / "continuous" / (acq_method + "Probe[A-Z]")
            )
        elif self.rec_kind == RecordingKind.FPGA:
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
            exp_name / rec_name / "events" / acq_method / "TTL"
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
                                    print(f"Pos data at: {self.path2PosData}\n")
                            self.path2PosOEBin = Path(d).parents[1]
                        if PurePath(d).match("*pos_data*"):
                            if self.path2PosData is None:
                                self.path2PosData = os.path.join(d)
                                if verbose:
                                    print(f"Pos data at: {self.path2PosData}\n")
                        if PurePath(d).match(str(TrackingPlugin_match)):
                            if self.path2PosData is None:
                                self.path2PosData = os.path.join(d)
                                if verbose:
                                    print(f"Pos data at: {self.path2PosData}\n")
                    if "continuous.dat" in ff:
                        if PurePath(d).match(str(APdata_match)):
                            self.path2APdata = os.path.join(d)
                            if verbose:
                                print(f"Continuous AP data at: {self.path2APdata}\n")
                            self.path2APOEBin = Path(d).parents[1]
                        if PurePath(d).match(str(LFPdata_match)):
                            self.path2LFPdata = os.path.join(d)
                            if verbose:
                                print(f"Continuous LFP data at: {self.path2LFPdata}\n")
                        if PurePath(d).match(str(Rawdata_match)):
                            self.path2APdata = os.path.join(d)
                            self.path2LFPdata = os.path.join(d)
                        if PurePath(d).match(str(TrackMe_match)):
                            self.path2PosData = os.path.join(d)
                            if verbose:
                                print(f"TrackMe posdata at: {self.path2PosData}\n")
                    if "sync_messages.txt" in ff:
                        if PurePath(d).match(str(sync_file_match)):
                            sync_file = os.path.join(d, "sync_messages.txt")
                            if fileContainsString(sync_file, "Start Time"):
                                self.sync_message_file = sync_file
                                if verbose:
                                    print(f"sync_messages file at: {sync_file}\n")
                    if "full_words.npy" in ff:
                        if PurePath(d).match(str(Events_match)):
                            self.path2EventsData = os.path.join(d)
                            if verbose:
                                print(f"Event data at: {self.path2EventsData}\n")
                        if PurePath(d).match(str(RippleDetector_match)):
                            self.path2RippleDetector = os.path.join(d)
                            if verbose:
                                print(
                                    f"""Ripple Detector plugin found at {
                                        self.path2RippleDetector
                                    }\n"""
                                )
                    if ".nwb" in ff:
                        self.path2NWBData = os.path.join(d, ff)
                        if verbose:
                            print(f"nwb data at: {self.path2NWBData}\n")
                    if "spike_templates.npy" in ff:
                        self.path2KiloSortData = os.path.join(d)
                        if verbose:
                            print(f"Found KiloSort data at {self.path2KiloSortData}\n")


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
            P = PosCalcsGeneric(xy[0, :], xy[1, :], cm=True, ppm=ppm, jumpmax=jumpmax)
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
