"""
Classes for parsing information contained in the settings.xml
file that is saved when recording with the openephys system.
"""

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

import os
import json
from pathlib import Path
from abc import ABC
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from ephysiopy.common.utils import memmapBinaryFile

"""
Some conversion classes used in the dataclasses below. When the attributes for the
dataclasses are added (via recursion in the addValues2Class and recurseNode functions at
the end of this file) they are added as strings. If we know that they should be ints or 
floats or whatever we can do that conversion by using descriptor-typed fields. See 

https://docs.python.org/3/library/dataclasses.html#descriptor-typed-fields

for more details
"""


class IntConversion:
    def __init__(self, *, default):
        self._default = default

    def __set_name__(self, owner, name):
        self._name = "_" + name

    def __get__(self, obj, type):
        if obj is None:
            return self._default

        return getattr(obj, self._name, self._default)

    def __set__(self, obj, value):
        setattr(obj, self._name, int(value))


class FloatConversion:
    def __init__(self, *, default):
        self._default = default

    def __set_name__(self, owner, name):
        self._name = "_" + name

    def __get__(self, obj, type):
        if obj is None:
            return self._default

        return getattr(obj, self._name, self._default)

    def __set__(self, obj, value):
        setattr(obj, self._name, float(value))


@dataclass
class Channel(object):
    """
    Documents the information attached to each channel
    """

    name: str = field(default_factory=str)
    _number: IntConversion = IntConversion(default=0)
    _gain: FloatConversion = FloatConversion(default=0)
    _param: bool = field(default_factory=bool)
    _record: bool = field(default=False)
    _audio: bool = field(default=False)
    _lowcut: IntConversion = IntConversion(default=0)
    _highcut: IntConversion = IntConversion(default=0)

    @property
    def number(self) -> int:
        return self._number

    @number.setter
    def number(self, value: str) -> None:
        self._number = int(value)

    @property
    def gain(self) -> float:
        return self._gain

    @gain.setter
    def gain(self, value: str) -> None:
        self._gain = float(value)

    @property
    def param(self) -> bool:
        return self._param

    @param.setter
    def param(self, value: str) -> None:
        if value == "1":
            self._param = True
        else:
            self._param = False

    @property
    def record(self) -> bool:
        return self._record

    @record.setter
    def record(self, value: str) -> None:
        if value == "1":
            self._record = True
        else:
            self._record = False

    @property
    def audio(self) -> bool:
        return self._audio

    @audio.setter
    def audio(self, value: str) -> None:
        if value == "1":
            self._audio = True
        else:
            self._audio = False

    @property
    def lowcut(self) -> int:
        return self._lowcut

    @lowcut.setter
    def lowcut(self, value: str) -> None:
        self._lowcut = int(value)

    @property
    def highcut(self) -> int:
        return self._highcut

    @highcut.setter
    def highcut(self, value: str) -> None:
        self._highcut = int(value)


@dataclass
class Stream:
    """
    Documents an OE DatasSream
    """

    name: str = field(default_factory=str)
    description: str = field(default_factory=str)
    sample_rate: IntConversion = IntConversion(default=0)
    channel_count: IntConversion = IntConversion(default=0)


@dataclass
class OEPlugin(ABC):
    """
    Documents an OE plugin
    """

    name: str = field(default_factory=str)
    insertionPoint: IntConversion = IntConversion(default=0)
    pluginName: str = field(default_factory=str)
    type: IntConversion = IntConversion(default=0)
    index: IntConversion = IntConversion(default=0)
    libraryName: str = field(default_factory=str)
    libraryVersion: IntConversion = IntConversion(default=0)
    processorType: IntConversion = IntConversion(default=0)
    nodeId: IntConversion = IntConversion(default=0)
    channel_count: IntConversion = IntConversion(default=0)
    stream: Stream = field(default_factory=Stream)
    sample_rate: IntConversion = IntConversion(default=0)


@dataclass
class RecordNode(OEPlugin):
    """
    Documents the RecordNode plugin
    """

    path: str = field(default_factory=str)
    engine: str = field(default_factory=str)
    recordEvents: IntConversion = IntConversion(default=0)
    recordSpikes: IntConversion = IntConversion(default=0)
    isMainStream: IntConversion = IntConversion(default=0)
    sync_line: IntConversion = IntConversion(default=0)
    source_node_id: IntConversion = IntConversion(default=0)
    recording_state: str = field(default_factory=str)


@dataclass
class RhythmFPGA(OEPlugin):
    """
    Documents the Rhythm FPGA plugin
    """

    channel_info: list[Channel] = field(default_factory=list)


# Identical to the RhythmFPGA class above for now
@dataclass
class NeuropixPXI(OEPlugin):
    """
    Documents the Neuropixels-PXI plugin
    """

    channel_info: list[Channel] = field(default_factory=list)


@dataclass
class AcquisitionBoard(OEPlugin):
    """
    Documents the Acquisition Board plugin
    """

    LowCut: IntConversion = IntConversion(default=0)
    HighCut: IntConversion = IntConversion(default=0)


@dataclass
class BandpassFilter(OEPlugin):
    """
    Documents the Bandpass Filter plugin
    """

    name = "Bandpass Filter"
    pluginName = "Bandpass Filter"
    pluginType = 1
    libraryName = "Bandpass Filter"

    channels: list[int] = field(default_factory=list)
    low_cut: FloatConversion = FloatConversion(default=0)
    high_cut: FloatConversion = FloatConversion(default=0)


@dataclass
class TrackingPort(OEPlugin):
    """
    Documents the Tracking Port plugin which uses Bonsai input
    and Tracking Visual plugin for visualisation within OE
    """

    def load(self, path2data: Path) -> np.ndarray:
        print("Loading Tracking Port data...")
        dt = np.dtype(
            [
                ("x", np.single),
                ("y", np.single),
                ("w", np.single),
                ("h", np.single),
            ]
        )
        data_array = np.load(path2data / Path("data_array.npy"))
        new_array = data_array.view(dtype=dt).copy()
        w = new_array["w"][0]
        h = new_array["h"][0]
        x = new_array["x"] * w
        y = new_array["y"] * h
        pos_data = np.array([np.ravel(x), np.ravel(y)]).T
        return pos_data

    def load_times(self, path2data: Path) -> np.ndarray:
        return np.load(path2data / Path("timestamps.npy"))


@dataclass
class PosTracker(OEPlugin):
    """
    Documents the PosTracker plugin
    """

    Brightness: IntConversion = IntConversion(default=20)
    Contrast: IntConversion = IntConversion(default=20)
    Exposure: IntConversion = IntConversion(default=20)
    LeftBorder: IntConversion = IntConversion(default=0)
    RightBorder: IntConversion = IntConversion(default=800)
    TopBorder: IntConversion = IntConversion(default=0)
    BottomBorder: IntConversion = IntConversion(default=600)
    AutoExposure: bool = field(default=False)
    OverlayPath: bool = field(default=False)
    sample_rate: IntConversion = IntConversion(default=30)

    """
    Custom methods for loading numpy arrays containing position data
    and associated timestamps
    """

    def load(self, path2data: Path) -> np.ndarray:
        print("Loading Tracker data...")
        return np.load(path2data / Path("data_array.npy"))

    def load_times(self, path2data: Path) -> np.ndarray:
        return np.load(path2data / Path("timestamps.npy"))


@dataclass
class TrackMe(OEPlugin):
    """
    Documents the TrackMe plugin
    """

    def load(self, path2data: Path) -> np.ndarray:
        print("Loading TrackMe data...")
        mmap = memmapBinaryFile(path2data / Path("data_array.npy"), self.channel_count)
        return np.array(mmap[0:2, :]).T

    def load_times(self, path2data: Path) -> np.ndarray:
        ts = np.load(path2data / Path("timestamps.npy"))
        return ts - ts[0]

    def load_frame_count(self, path2data: Path) -> np.ndarray:
        data = memmapBinaryFile(path2data / Path("data_array.npy"), self.channel_count)
        # framecount data is always last column in continuous.dat file
        return np.array(data[-1, :]).T

    def load_ttl_times(self, path2data: Path) -> np.ndarray:
        ts = np.load(path2data / Path("timestamps.npy"))
        states = np.load(path2data / Path("states.npy"))
        return ts[states > 0]


@dataclass
class StimControl(OEPlugin):
    """
    Documents the StimControl plugin
    """

    Device: IntConversion = IntConversion(default=0)
    Duration: IntConversion = IntConversion(default=0)
    Interval: IntConversion = IntConversion(default=0)
    Gate: IntConversion = IntConversion(default=0)
    Output: IntConversion = IntConversion(default=0)
    Start: IntConversion = IntConversion(default=0)
    Stop: IntConversion = IntConversion(default=0)
    Trigger: IntConversion = IntConversion(default=0)


@dataclass
class SpikeSorter(OEPlugin):
    pass


@dataclass
class SpikeViewer(OEPlugin):
    pass


@dataclass
class Electrode(object):
    """
    Documents the ELECTRODE entries in the settings.xml file
    """

    nChannels: IntConversion = IntConversion(default=0)
    id: IntConversion = IntConversion(default=0)
    subChannels: list[int] = field(default_factory=list)
    subChannelsThresh: list[int] = field(default_factory=list)
    subChannelsActive: list[int] = field(default_factory=list)
    prePeakSamples: IntConversion = IntConversion(default=8)
    postPeakSamples: IntConversion = IntConversion(default=32)


"""
The RippleDetector plugin emits TTL events so it has a custom
method defined for loading and processing that
"""


@dataclass
class RippleDetector(OEPlugin):
    """
    Documents the Ripple Detector plugin
    """

    Ripple_Input: IntConversion = IntConversion(default=-1)
    Ripple_Out: IntConversion = IntConversion(default=-1)
    Ripple_save: IntConversion = IntConversion(default=-1)
    ripple_std: FloatConversion = FloatConversion(default=-1)
    time_thresh: FloatConversion = FloatConversion(default=-1)
    refr_time: FloatConversion = FloatConversion(default=-1)
    rms_samples: FloatConversion = FloatConversion(default=-1)
    ttl_duration: IntConversion = IntConversion(default=-1)
    ttl_percent: IntConversion = IntConversion(default=-1)
    mov_detect: IntConversion = IntConversion(default=-1)
    mov_input: IntConversion = IntConversion(default=-1)
    mov_out: IntConversion = IntConversion(default=-1)
    mov_std: FloatConversion = FloatConversion(default=-1)
    min_time_st: FloatConversion = FloatConversion(default=-1)
    min_time_mov: FloatConversion = FloatConversion(default=-1)

    def load_ttl(self, path2TTL: Path, trial_start_time: float) -> dict:
        timestamps = np.load(path2TTL / Path("timestamps.npy")) - trial_start_time
        states = np.load(path2TTL / Path("states.npy"))
        out = dict()
        out_ttl = int(self.Ripple_Out)
        save_ttl = int(self.Ripple_save)
        # check for identical times - should fix the plugin really...
        bad_indices = []
        for i in range(len(states) - 2):
            i_pair = states[i : i + 2]
            if np.all(i_pair == np.array([save_ttl, out_ttl])):
                if np.diff(timestamps[i : i + 2]) == 0:
                    bad_indices.append(i)
        mask = np.ones_like(states, dtype=bool)
        mask[bad_indices] = False
        timestamps = timestamps[mask]
        states = states[mask]

        out["ttl_timestamps"] = timestamps[states == out_ttl]
        out["ttl_timestamps_off"] = timestamps[states == out_ttl * -1]
        out["no_laser_ttls"] = timestamps[states == save_ttl]
        if len(out["ttl_timestamps_off"]) == len(out["ttl_timestamps"]):
            out["stim_duration"] = np.nanmean(
                out["ttl_timestamps_off"] - out["ttl_timestamps"]
            )
        else:
            out["stim_duration"] = None
        return out


class AbstractProcessorFactory:
    def create_pos_tracker(self):
        return PosTracker()

    def create_rhythm_fpga(self):
        return RhythmFPGA()

    def create_neuropix_pxi(self):
        return NeuropixPXI()

    def create_acquisition_board(self):
        return AcquisitionBoard()

    def create_spike_sorter(self):
        return SpikeSorter()

    def create_track_me(self):
        return TrackMe()

    def create_record_node(self):
        return RecordNode()

    def create_stim_control(self):
        return StimControl()

    def create_oe_plugin(self):
        return OEPlugin()

    def create_ripple_detector(self):
        return RippleDetector()

    def create_bandpass_filter(self):
        return BandpassFilter()


class ProcessorFactory:
    factory = AbstractProcessorFactory()

    def create_processor(self, proc_name: str):
        if "Pos Tracker" in proc_name or "PosTracker" in proc_name:
            return self.factory.create_pos_tracker()
        elif "Rhythm" in proc_name:
            return self.factory.create_rhythm_fpga()
        elif "Neuropix-PXI" in proc_name:
            return self.factory.create_neuropix_pxi()
        elif "Acquisition Board" in proc_name:
            return self.factory.create_acquisition_board()
        elif "Spike Sorter" in proc_name:
            return self.factory.create_spike_sorter()
        elif "TrackMe" in proc_name:
            return self.factory.create_track_me()
        elif "Record Node" in proc_name:
            return self.factory.create_record_node()
        elif "StimControl" in proc_name:
            return self.factory.create_stim_control()
        elif "Ripple Detector" in proc_name:
            return self.factory.create_ripple_detector()
        elif "Bandpass Filter" in proc_name:
            return self.factory.create_bandpass_filter()
        else:
            return self.factory.create_oe_plugin()


def recurseNode(node: ET.Element, func: Callable, cls: dataclass):
    """
    Recursive function that applies func to each node
    """
    if node is not None:
        func(node, cls)
        for item in node:
            recurseNode(item, func, cls)


def addValues2Class(node: ET.Element, cls: dataclass):
    for i in node.items():
        if hasattr(cls, i[0]):
            setattr(cls, i[0], i[1])
    if hasattr(cls, "channel_info") and node.tag == "CHANNEL":
        if cls.channel_info is None:
            cls.channel_info = list()
        chan = Channel()
        recurseNode(node, addValues2Class, chan)
        cls.channel_info.append(chan)
    if hasattr(cls, "stream") and node.tag == "STREAM":
        if cls.stream is None:
            cls.stream = Stream()
        recurseNode(node, addValues2Class, cls.stream)


class OEStructure(object):
    """
    Loads up the structure.oebin file for openephys flat binary
    format recordings
    """

    def __init__(self, fname: Path):
        self.filename = []
        if isinstance(fname, str):
            fname = Path(fname)
        files = dict.fromkeys(self.find_oebin(fname))

        for f in files.keys():
            files[f] = self.read_oebin(f)

        self.data = files

    def find_oebin(self, pname: Path) -> list:
        path2oebins = []
        for d, c, f in os.walk(pname):
            for ff in f:
                if "." not in c:  # ignore hidden directories
                    if ff == "structure.oebin":
                        path2oebins.append(Path(d))
        return path2oebins

    def read_oebin(self, fname: Path) -> dict:
        self.filename.append(fname)
        fname = fname / Path("structure.oebin")
        with open(fname, "r") as f:
            data = json.load(f)
        return data


class Settings(object):
    """
    Groups together the other classes in this module and does the actual
    parsing of the settings.xml file

    Parameters
    ----------
    pname : str
        The pathname to the top-level directory, typically in form
        YYYY-MM-DD_HH-MM-SS
    """

    def __init__(self, pname: str | Path):
        self.filename = None

        for d, _, f in os.walk(pname):
            for ff in f:
                if "settings.xml" in ff:
                    self.filename = os.path.join(d, "settings.xml")
        self.tree = None
        self.processors = OrderedDict()
        self.record_nodes = OrderedDict()
        self.load()
        self.parse()

    def load(self):
        """
        Creates a handle to the basic xml document
        """
        if self.filename is not None:
            self.tree = ET.parse(self.filename).getroot()

    def parse(self):
        """
        Parses the basic information about the processors in the
        open-ephys signal chain and as described in the settings.xml
        file(s)
        """
        if self.tree is None:
            self.load()
        processor_factory = ProcessorFactory()
        if self.tree is not None:
            for elem in self.tree.iter("PROCESSOR"):
                i_proc = elem.get("name")
                if i_proc is not None:
                    if "/" in i_proc:
                        i_proc = i_proc.split("/")[-1]
                    new_processor = processor_factory.create_processor(i_proc)
                    recurseNode(elem, addValues2Class, new_processor)
                    if i_proc == "Record Node":
                        if new_processor.nodeId is not None:
                            self.record_nodes[
                                i_proc + " " + str(new_processor.nodeId)
                            ] = new_processor
                        else:
                            self.record_nodes[i_proc] = new_processor
                    else:
                        if new_processor.nodeId is not None:
                            self.processors[
                                i_proc + " " + str(new_processor.nodeId)
                            ] = new_processor
                        else:
                            self.processors[i_proc] = new_processor

    def get_processor(self, key: str):
        """
        Returns the information about the requested processor or an
        empty OEPlugin instance if it's not available
        """
        processor = [self.processors[k] for k in self.processors.keys() if key in k]
        if processor:
            if len(processor) == 1:
                return processor[0]
            else:
                return processor
        return OEPlugin()
