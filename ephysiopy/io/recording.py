import abc
import os
import warnings
from enum import Enum
from pathlib import Path, PurePath
from typing import NoReturn

import h5py
import numpy as np
from ephysiopy.common.ephys_generic import PosCalcsGeneric
from ephysiopy.dacq2py.axonaIO import IO, Pos
from ephysiopy.openephys2py.OEKiloPhy import KiloSortSession
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

    if os.path.exists(path2file):
        status = os.stat(path2file)
        n_samples = int(status.st_size / (2.0 * n_channels))
        mmap = np.memmap(
            path2file, np.int16, "r", 0, (n_channels, n_samples), order="F"
        )
        return mmap
    else:
        return np.empty(0)


def loadTrackingPluginData(pname: Path) -> np.array:
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


def loadTrackMePluginData(pname: Path) -> np.ndarray:
    mmap = memmapBinaryFile(str(pname), n_channels=6)
    return np.array(mmap[0:2, :]).T


def loadTrackMeTimestamps(pname: Path) -> np.ndarray:
    ts = np.load(os.path.join(pname, "timestamps.npy"))
    states = np.load(os.path.join(pname, "states.npy"))
    return ts[states > 0]


class RecordingKind(Enum):
    FPGA = 1
    NEUROPIXELS = 2
    ACQUISITIONBOARD = 3
    NWB = 4


Xml2RecordingKind = {
    "Acquisition Board": RecordingKind.ACQUISITIONBOARD,
    "Neuropix-PXI": RecordingKind.NEUROPIXELS,
    "Rhythm_FPGA": RecordingKind.FPGA}


class TrackingKind(Enum):
    POSTRACKER = 1
    TRACKINGPLUGIN = 2


class TrialInterface(FigureMaker, metaclass=abc.ABCMeta):
    def __init__(self, pname: str, **kwargs) -> None:
        assert os.path.exists(pname)
        self._pname = pname
        self._settings = None
        self._PosCalcs = None
        self._pos_data_type = None
        self._sync_message_file = None
        self._clusterData = None  # Kilosort or .cut / .clu file
        self._recording_start_time = None
        self._ttl_data = None
        self._accelerometer_data = None
        self._path2PosData = None

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "load_neural_data")
            and callable(subclass.load_neural_data)
            and hasattr(subclass, "load_pos")
            and callable(subclass.load_pos)
            and hasattr(subclass, "load_cluster_data")
            and callable(subclass.load_cluster_data)
            and hasattr(subclass, "load_settings")
            and callable(subclass.load_settings)
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
    def pos_data_type(self):
        return self._pos_data_type

    @pos_data_type.setter
    def pos_data_type(self, val):
        self._pos_data_type = val

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
    def load_neural_data(self, pname: Path) -> NoReturn:
        """Load the neural data"""
        raise NotImplementedError

    @abc.abstractmethod
    def load_pos_data(
        self, pname: Path, ppm: int = 300, jumpmax: int = 100
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
    def load_cluster_data(self):
        """Load the cluster data (Kilosort/ Axona cut/ whatever else"""
        raise NotImplementedError

    @abc.abstractmethod
    def load_settings(self):
        """Loads the format specific settings file"""
        raise NotImplementedError


class AxonaTrial(TrialInterface):
    def __init__(self, pname: Path, **kwargs) -> None:
        super().__init__(pname, **kwargs)
        self.__settings = None

    @property
    def settings(self) -> None:
        if self.__settings is None:
            try:
                settings_io = IO()
                self._settings = settings_io.getHeader(self.pname)
            except IOError:
                print(".set file not loaded")
                self._settings = None

    @settings.setter
    def settings(self, value) -> None:
        self.__settings = value

    def load_neural_data(self, pname: Path) -> None:
        pass

    def load_cluster_data(self):
        pass

    def load_settings(self):
        return super().load_settings()

    def load_pos_data(self, pname: Path, ppm: int = 300, jumpmax: int = 100) -> None:
        if self.PosCalcs is None:
            try:
                AxonaPos = Pos(self.pname)
                P = PosCalcsGeneric(
                    AxonaPos.led_pos[0, :],
                    AxonaPos.led_pos[1, :],
                    cm=True,
                    ppm=self.ppm,
                )
                P.xyTS = Pos.ts
                P.sample_rate = AxonaPos.getHeaderVal(AxonaPos.header, "sample_rate")
                P.postprocesspos()
                print("Loaded pos data")
                self.PosCalcs = P
            except IOError:
                print("Couldn't load the pos data")


class OpenEphysBase(TrialInterface):
    def __init__(self, pname: Path, **kwargs) -> None:
        super().__init__(pname, **kwargs)
        self.load_settings()
        setattr(self, "sync_messsage_file", None)
        record_methods = ["Acquisition Board", "Neuropix-PXI", "Rhythm_FPGA"]
        record_method = [i for i in self.settings.processors.keys()
                         if i in record_methods][0]
        self.rec_kind = Xml2RecordingKind[record_method]

    def load_neural_data(self, pname: Path) -> None:
        pass

    def load_settings(self):
        if self._settings is None:
            # pname_root gets walked through and over-written with
            # correct location of settings.xml
            self.settings = Settings(self.pname)

    def load_cluster_data(self):
        if self.pname is not None:
            if os.path.exists(self.pname):
                clusterData = KiloSortSession(self.pname)
            if clusterData is not None:
                if clusterData.load():
                    try:
                        clusterData.removeKSNoiseClusters()
                    except Exception:
                        pass
        self.clusterData = clusterData

    def load_pos_data(self, pname: Path, ppm: int = 300, jumpmax: int = 100) -> None:
        # Only sub-class that doesn't use this is OpenEphysNWB
        # which needs updating
        # TODO: Update / overhaul OpenEphysNWB
        # Load the start time from the sync_messages file
        recording_start_time = 0
        if self.sync_message_file is not None:
            with open(self.sync_message_file, "r") as f:
                sync_strs = f.read()
            sync_lines = sync_strs.split("\n")
            for line in sync_lines:
                if "subProcessor: 0" in line:
                    idx = line.find("start time: ")
                    start_val = line[idx + len("start time: ") : -1]
                    tmp = start_val.split("@")
                    recording_start_time = float(tmp[0])  # in samples
        if self.path2PosData is not None:
            pos_data_type = getattr(self, "pos_data_type", "PosTracker")
            if pos_data_type == "PosTracker":
                print("Loading PosTracker data...")
                pos_data = np.load(os.path.join(self.path2PosData, "data_array.npy"))
            if pos_data_type == "TrackingPlugin":
                print("Loading Tracking Plugin data...")
                pos_data = loadTrackingPluginData(
                    os.path.join(self.path2PosData, "data_array.npy")
                )
            pos_ts = np.load(os.path.join(self.path2PosData, "timestamps.npy"))
            pos_ts = np.ravel(pos_ts)
            if pos_data_type == "TrackMe":
                print("Loading TrackMe data...")
                pos_data = loadTrackMePluginData(
                    Path(os.path.join(self.path2PosData, "continuous.dat")))
                pos_ts = loadTrackMeTimestamps(self.path2EventsData)
                pos_ts = pos_ts[0:len(pos_data)]
            pos_timebase = getattr(self, "pos_timebase", 3e4)
            sample_rate = np.floor(1 / np.mean(np.diff(pos_ts) / pos_timebase))
            xyTS = pos_ts - recording_start_time
            # xyTS = xyTS / pos_timebase  # convert to seconds
            if self.sync_message_file is not None:
                recording_start_time = xyTS[0]

            P = PosCalcsGeneric(
                pos_data[:, 0],
                pos_data[:, 1],
                cm=True,
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

    def find_files(
        self,
        pname_root: str,
        experiment_name: str = "experiment1",
        recording_name: str = "recording1",
    ):
        exp_name = Path(experiment_name)
        PosTracker_match = (
            exp_name / recording_name / "events" / "*Pos_Tracker*/BINARY_group*"
        )
        TrackingPlugin_match = (
            exp_name / recording_name / "events" / "*Tracking_Port*/BINARY_group*"
        )
        TrackMe_match = (
            exp_name / recording_name / "continuous" / "TrackMe-[0-9][0-9][0-9].TrackingNode"
        )
        sync_file_match = exp_name / recording_name
        acquisition_method = ""
        match self.rec_kind:
            case RecordingKind.NEUROPIXELS:
                acquisition_method = "Neuropix-PXI-[0-9][0-9][0-9]."
                APdata_match = exp_name / recording_name / "continuous" / (acquisition_method + "0")
                LFPdata_match = exp_name / recording_name / "continuous" / (acquisition_method + "1")
            case RecordingKind.FPGA:
                acquisition_method = "Rhythm_FPGA-[0-9][0-9][0-9]."
                APdata_match = exp_name / recording_name / "continuous" / (acquisition_method + "0")
                LFPdata_match = exp_name / recording_name / "continuous" / (acquisition_method + "1")
            case _:
                acquisition_method = "Acquisition_Board-[0-9][0-9][0-9].*"
                APdata_match = exp_name / recording_name / "continuous" / acquisition_method
                LFPdata_match = exp_name / recording_name / "continuous" / acquisition_method
        Events_match = (
            # only dealing with a single TTL channel at the moment
            exp_name / recording_name / "events" / acquisition_method / "TTL"
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
                                setattr(self, "pos_data_type", "PosTracker")
                                print(f"Found pos data at: {self.path2PosData}")
                            self.path2PosOEBin = Path(d).parents[1]
                        if PurePath(d).match("*pos_data*"):
                            if self.path2PosData is None:
                                self.path2PosData = os.path.join(d)
                                print(f"Found pos data at: {self.path2PosData}")
                        if PurePath(d).match(str(TrackingPlugin_match)):
                            if self.path2PosData is None:
                                self.path2PosData = os.path.join(d)
                                setattr(self, "pos_data_type", "TrackingPlugin")
                                print(f"Found pos data at: {self.path2PosData}")
                    if "continuous.dat" in ff:
                        if PurePath(d).match(str(APdata_match)):
                            self.path2APdata = os.path.join(d)
                            print(f"Found continuous data at: {self.path2APdata}")
                            self.path2APOEBin = Path(d).parents[1]
                        if PurePath(d).match(str(LFPdata_match)):
                            self.path2LFPdata = os.path.join(d)
                            print(f"Found continuous data at: {self.path2LFPdata}")
                        if PurePath(d).match(str(TrackMe_match)):
                            self.path2PosData = os.path.join(d)
                            setattr(self, "pos_data_type", "TrackMe")
                            print(f"Found TrackMe posdata at: {self.path2PosData}")
                    if "sync_messages.txt" in ff:
                        if PurePath(d).match(str(sync_file_match)):
                            sync_file = os.path.join(d, "sync_messages.txt")
                            if fileContainsString(sync_file, "Processor"):
                                self.sync_message_file = sync_file
                                print(f"Found sync_messages file at: {sync_file}")
                    if "full_words.npy" in ff:
                        if PurePath(d).match(str(Events_match)):
                            self.path2EventsData = os.path.join(d)
                            print(f"Found event data at: {self.path2EventsData}")
                    if ".nwb" in ff:
                        self.path2NWBData = os.path.join(d, ff)
                        print(f"Found nwb data at: {self.path2NWBData}")


class OpenEphysNWB(OpenEphysBase):
    def __init__(self, pname: Path, **kwargs) -> None:
        super().__init__(pname, **kwargs)

    def load_neural_data(self, pname: Path) -> None:
        pass

    def load_settings(self):
        return super().load_settings()

    def load_pos_data(self, pname: Path, ppm: int = 300, jumpmax: int = 100) -> None:
        assert pname.exists()
        with h5py.File(os.path.join(self.path2NWBData), mode="r") as nwbData:
            xy = np.array(nwbData[self.path2PosData + "/data"])
            xy = xy[:, 0:2]
            ts = np.array(nwbData[self.path2PosData]["timestamps"])
            P = PosCalcsGeneric(
                xy[0, :],
                xy[1, :],
                cm=True,
                ppm=self.ppm,
            )
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
