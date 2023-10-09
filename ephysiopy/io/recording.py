import abc
import os
import re
import warnings
from enum import Enum
from pathlib import Path, PurePath
from typing import NoReturn

import h5py
import numpy as np
from phylib.io.model import TemplateModel

from ephysiopy.axona.axonaIO import IO, Pos
from ephysiopy.axona.tetrode_dict import TetrodeDict
from ephysiopy.common.ephys_generic import EEGCalcsGeneric, PosCalcsGeneric
from ephysiopy.openephys2py.KiloSort import KiloSortSession
from ephysiopy.openephys2py.OESettings import Settings
from ephysiopy.visualise.plotting import FigureMaker


def fileContainsString(pname: str, searchStr: str) -> bool:
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


def memmapBinaryFile(path2file: str, n_channels=384, **kwargs) -> np.ndarray:
    """
    Returns a numpy memmap of the int16 data in the
    file path2file, if present
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


def loadTrackingPluginData(pname: Path) -> np.ndarray:
    dt = np.dtype(
        {
            "x": (np.single, 0),
            "y": (np.single, 4),
            "w": (np.single, 8),
            "h": (np.single, 12),
        }
    )
    data_array = np.load(pname)
    new_array = data_array.view(dtype=dt).copy()
    w = new_array["w"][0]
    h = new_array["h"][0]
    x = new_array["x"] * w
    y = new_array["y"] * h
    pos_data = np.array([np.ravel(x), np.ravel(y)]).T
    return pos_data


def loadTrackMePluginData(pname: Path, n_channels: int = 4) -> np.ndarray:
    mmap = memmapBinaryFile(str(pname), n_channels)
    return np.array(mmap[0:2, :]).T


def loadTrackMeTTLTimestamps(pname: Path) -> np.ndarray:
    ts = np.load(os.path.join(pname, "timestamps.npy"))
    states = np.load(os.path.join(pname, "states.npy"))
    return ts[states == 2]


def loadTrackMeTimestamps(pname: Path) -> np.ndarray:
    ts = np.load(os.path.join(pname, "timestamps.npy"))
    return ts - ts[0]


def loadTrackMeFrameCount(pname: Path, n_channels: int = 4) -> np.ndarray:
    data = memmapBinaryFile(str(pname), n_channels)
    # framecount data is always last column in continuous.dat file
    return np.array(data[-1, :]).T


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
    "Neuropix": RecordingKind.NEUROPIXELS
}


class TrialInterface(FigureMaker, metaclass=abc.ABCMeta):
    def __init__(self, pname: Path, **kwargs) -> None:
        assert Path(pname).exists()
        self._pname = pname
        self._settings = None
        self._PosCalcs = None
        self._RateMap = None
        self._sync_message_file = None
        self._clusterData = None  # Kilosort or .cut / .clu file
        self._recording_start_time = None  # float
        self._ttl_data = None  # dict
        self._accelerometer_data = None
        self._path2PosData = None  # Path or str

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

    @abc.abstractmethod
    def load_lfp(self, *args, **kwargs) -> NoReturn:
        """Load the LFP data"""
        raise NotImplementedError

    @abc.abstractmethod
    def load_neural_data(self, *args, **kwargs) -> NoReturn:
        """Load the neural data"""
        raise NotImplementedError

    @abc.abstractmethod
    def load_pos_data(
        self, ppm: int = 300, jumpmax: int = 100, *args, **kwargs
    ) -> NoReturn:
        """
        Load the position data

        Parameters
        -----------
        pname : Path
            Path to base directory containing pos data
        ppm : int
            pixels per metre
        jumpmax : int
            max jump in pixels between positions
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_cluster_data(self, *args, **kwargs) -> bool:
        """Load the cluster data (Kilosort/ Axona cut/ whatever else"""
        raise NotImplementedError

    @abc.abstractmethod
    def load_settings(self, *args, **kwargs) -> NoReturn:
        """Loads the format specific settings file"""
        raise NotImplementedError

    @abc.abstractmethod
    def load_ttl(self, *args, **kwargs) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def get_spike_times(self, cluster: int, tetrode: int, *args, **kwargs):
        """Returns the times of an individual cluster"""
        raise NotImplementedError


class AxonaTrial(TrialInterface):
    def __init__(self, pname: Path, **kwargs) -> None:
        pname = Path(pname)
        super().__init__(pname, **kwargs)
        self._settings = None
        use_volts = kwargs.get("volts", True)
        self.TETRODE = TetrodeDict(
            str(self.pname.with_suffix("")), volts=use_volts)
        self.load_settings()

    def load_lfp(self, *args, **kwargs):
        from ephysiopy.axona.axonaIO import EEG

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

    def load_settings(self, *args, **kwargs):
        if self._settings is None:
            try:
                settings_io = IO()
                self.settings = settings_io.getHeader(str(self.pname))
            except IOError:
                print(".set file not loaded")
                self.settings = None

    def load_pos_data(
        self, pname: Path, ppm: int = 300, jumpmax: int = 100, *args, **kwargs
    ) -> None:
        try:
            AxonaPos = Pos(Path(pname))
            P = PosCalcsGeneric(
                AxonaPos.led_pos[:, 0],
                AxonaPos.led_pos[:, 1],
                cm=True,
                ppm=ppm,
                jumpmax=jumpmax,
            )
            P.xyTS = AxonaPos.ts
            P.sample_rate = AxonaPos.getHeaderVal(
                AxonaPos.header, "sample_rate")
            P.postprocesspos()
            print("Loaded pos data")
            self.PosCalcs = P
        except IOError:
            print("Couldn't load the pos data")

    def load_ttl(self, *args, **kwargs) -> bool:
        from ephysiopy.axona.axonaIO import Stim
        try:
            self.ttl_data = Stim(self.pname)
            # ttl times in Stim are in ms
        except IOError:
            return False
        return True

    def get_spike_times(
            self, cluster: int, tetrode=None,
            *args, **kwargs):
        if tetrode is not None:
            return self.TETRODE.get_spike_samples(tetrode, cluster)


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
        self.find_files(pname)
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
        """
        Get the recording start time from the sync_messages.txt file
        """
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
        return recording_start_time

    def get_spike_times(self, cluster: int, tetrode: int = None, *args, **kwargs):
        ts = self.clusterData.spk_times
        if cluster in self.clusterData.spk_clusters:
            times = ts[self.clusterData.spk_clusters == cluster]
            return times.astype(np.int64) / self.sample_rate
        else:
            warnings.warn("Cluster not present")

    def load_lfp(self, *args, **kwargs):
        """
        Valid kwargs are:
        'target_sample_rate' - int
            the sample rate to downsample to from the original
        """
        from scipy import signal

        if self.path2LFPdata is not None:
            lfp = memmapBinaryFile(
                os.path.join(self.path2LFPdata, "continuous.dat"),
                n_channels=self.channel_count,
            )
            channel = 0
            if "channel" in kwargs.keys():
                channel = kwargs["channel"]
            target_sample_rate = 500
            if "target_sample_rate" in kwargs.keys():
                target_sample_rate = kwargs["target_sample_rate"]
            n_samples = np.shape(lfp[channel, :])[0]
            sig = signal.resample(
                lfp[channel, :], int(n_samples / self.sample_rate) * target_sample_rate
            )
            self.EEGCalcs = EEGCalcsGeneric(sig, target_sample_rate)

    def load_neural_data(self, *args, **kwargs) -> None:
        if "path2APdata" in kwargs.keys():
            self.path2APdata: Path = Path(kwargs["path2APdata"])
        n_channels: int = self.channel_count or kwargs["nChannels"]
        try:
            self.template_model = TemplateModel(
                dir_path=self.path2APdata,
                sample_rate=3e4,
                dat_path=Path(self.path2APdata).joinpath("continuous.dat"),
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
            print("Loaded settings data")

    def load_cluster_data(
            self, removeNoiseClusters=True, *args, **kwargs) -> bool:
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
                "Tracking Port"
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

            if "Tracker" in pos_plugin_name:
                print("Loading Tracker data...")
                pos_data = np.load(os.path.join(self.path2PosData, "data_array.npy"))
            if "Tracking Port" in pos_plugin_name:
                print("Loading Tracking Port data...")
                pos_data = loadTrackingPluginData(
                    os.path.join(self.path2PosData, "data_array.npy"))
            if "TrackingPlugin" in pos_plugin_name:
                print("Loading TrackingPlugin data...")
                pos_data = loadTrackingPluginData(
                    os.path.join(self.path2PosData, "data_array.npy")
                )
            
            pos_ts = np.load(os.path.join(self.path2PosData, "timestamps.npy"))
            # pos_ts in seconds
            pos_ts = np.ravel(pos_ts)
            if "TrackMe" in pos_plugin_name:
                print("Loading TrackMe data...")
                n_pos_chans = int(
                    self.settings.processors[pos_plugin_name].channel_count
                )
                pos_data = loadTrackMePluginData(
                    Path(os.path.join(self.path2PosData, "continuous.dat")),
                    n_channels=n_pos_chans,
                )
                if "loadTTLPos" in kwargs.keys():
                    pos_ts = loadTrackMeTTLTimestamps(Path(self.path2EventsData))
                else:
                    pos_ts = loadTrackMeTimestamps(Path(self.path2PosData))
                pos_ts = pos_ts[0:len(pos_data)]
            sample_rate = self.settings.processors[pos_plugin_name].sample_rate
            sample_rate = float(sample_rate) if sample_rate is not None else 50
            # the timestamps for the Tracker Port plugin are fucked so 
            # we have to infer from the shape of the position data
            if "Tracking Port" in pos_plugin_name:
                sample_rate = kwargs["sample_rate"] or 50
                # pos_ts in seconds
                pos_ts = np.arange(
                    0, pos_data.shape[0]/sample_rate, 1.0/sample_rate)
                print(f"Tracker first and last ts: {pos_ts[0]} & {pos_ts[-1]}")
            if pos_plugin_name != "TrackMe":
                xyTS = pos_ts - recording_start_time
            else:
                xyTS = pos_ts
            if self.sync_message_file is not None:
                recording_start_time = xyTS[0]
            print(f"First and last ts before PosCalcs: {pos_ts[0]} & {pos_ts[-1]}")
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
        """
        Valid kwargs are:
        StimControl_id: str
            This is the string "StimControl [0-9][0-9][0-9]" where the numbers
            are the node id in the openephys signal chain
        TTL_channel_number: int
            The integer value in the "states.npy" file that corresponds to the
            identity of the TTL input on the Digital I/O board on the
            openephys recording system. i.e. if there is input to BNC port 3 on
            the digital I/O board then values of 3 in the states.npy file are
            high TTL values on this input and -3 are low TTL values (I think)

        Returns
        -------
        Nothing but sets some keys/values in a dict on 'self'
        called ttl_data, namely:

        ttl_timestamps: list - the times of high ttl pulses in ms
        stim_duration: int - the duration of the ttl pulse in ms
        """
        if not Path(self.path2EventsData).exists:
            return False
        ttl_ts = np.load(os.path.join(self.path2EventsData, "timestamps.npy"))
        states = np.load(os.path.join(self.path2EventsData, "states.npy"))
        recording_start_time = self._get_recording_start_time()
        self.ttl_data = {}
        if "StimControl_id" in kwargs.keys():
            stim_id = kwargs["StimControl_id"]
            duration = getattr(self.settings.processors[stim_id], "Duration")
            self.ttl_data["stim_duration"] = int(duration)
        if "TTL_channel_number" in kwargs.keys():
            chan = kwargs["TTL_channel_number"]
            high_ttl = ttl_ts[states == chan]
            # get into ms
            high_ttl = (high_ttl * 1000.0) - recording_start_time
            self.ttl_data['ttl_timestamps'] = high_ttl
        if not self.ttl_data:
            return False
        print("Loaded ttl data")
        return True

    def find_files(
        self,
        pname_root: str,
        experiment_name: str = "experiment1",
        rec_name: str = "recording1",
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
            exp_name
            / rec_name
            / "events"
            / acq_method
            / "TTL"
        )

        if pname_root is None:
            pname_root = self.pname_root

        for d, c, f in os.walk(pname_root):
            for ff in f:
                if "." not in c:  # ignore hidden directories
                    if "data_array.npy" in ff:
                        if PurePath(d).match(str(PosTracker_match)):
                            if self.path2PosData is None:
                                self.path2PosData = os.path.join(d)
                                print(f"Pos data at: {self.path2PosData}")
                            self.path2PosOEBin = Path(d).parents[1]
                        if PurePath(d).match("*pos_data*"):
                            if self.path2PosData is None:
                                self.path2PosData = os.path.join(d)
                                print(f"Pos data at: {self.path2PosData}")
                        if PurePath(d).match(str(TrackingPlugin_match)):
                            if self.path2PosData is None:
                                self.path2PosData = os.path.join(d)
                                print(f"Pos data at: {self.path2PosData}")
                    if "continuous.dat" in ff:
                        if PurePath(d).match(str(APdata_match)):
                            self.path2APdata = os.path.join(d)
                            print(f"Continuous AP data at: {self.path2APdata}")
                            self.path2APOEBin = Path(d).parents[1]
                        if PurePath(d).match(str(LFPdata_match)):
                            self.path2LFPdata = os.path.join(d)
                            print(f"Continuous LFP data at: {self.path2LFPdata}")
                        if PurePath(d).match(str(Rawdata_match)):
                            self.path2APdata = os.path.join(d)
                            self.path2LFPdata = os.path.join(d)
                        if PurePath(d).match(str(TrackMe_match)):
                            self.path2PosData = os.path.join(d)
                            print(f"TrackMe posdata at: {self.path2PosData}")
                    if "sync_messages.txt" in ff:
                        if PurePath(d).match(str(sync_file_match)):
                            sync_file = os.path.join(d, "sync_messages.txt")
                            if fileContainsString(sync_file, "Start Time"):
                                self.sync_message_file = sync_file
                                print(f"sync_messages file at: {sync_file}")
                    if "full_words.npy" in ff:
                        if PurePath(d).match(str(Events_match)):
                            self.path2EventsData = os.path.join(d)
                            print(f"Event data at: {self.path2EventsData}")
                    if ".nwb" in ff:
                        self.path2NWBData = os.path.join(d, ff)
                        print(f"nwb data at: {self.path2NWBData}")
                    if "spike_templates.npy" in ff:
                        self.path2KiloSortData = os.path.join(d)
                        print(f"Found KiloSort data at {self.path2KiloSortData}")


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
