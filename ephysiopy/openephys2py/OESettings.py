"""
Classes for parsing information contained in the settings.xml
file that is saved when recording with the openephys system.
"""

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

import xml
import os
import json
from pathlib import Path
from abc import ABC
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Channel(object):
    """
    Documents the information attached to each channel
    """

    name: str = field(default_factory=str)
    _number: int = field(default_factory=int)
    _gain: float = field(default_factory=float)
    _param: bool = field(default_factory=bool)
    _record: bool = field(default=False)
    _audio: bool = field(default=False)
    _lowcut: int = field(default_factory=int)
    _highcut: int = field(default_factory=int)

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
    sample_rate: int = field(default_factory=int)
    channel_count: int = field(default_factory=int)


@dataclass
class OEPlugin(ABC):
    """
    Documents an OE plugin
    """

    name: str = field(default_factory=str)
    insertionPoint: int = field(default_factory=int)
    pluginName: str = field(default_factory=str)
    type: int = field(default_factory=int)
    index: int = field(default_factory=int)
    libraryName: str = field(default_factory=str)
    libraryVersion: int = field(default_factory=int)
    processorType: int = field(default_factory=int)
    nodeId: int = field(default_factory=int)
    channel_count: int = field(default_factory=int)
    stream: Stream = field(default_factory=Stream)
    sample_rate: int = field(default_factory=int)


@dataclass
class RecordNode(OEPlugin):
    """
    Documents the RecordNode plugin
    """

    path: str = field(default_factory=str)
    engine: str = field(default_factory=str)
    recordEvents: int = field(default_factory=int)
    recordSpikes: int = field(default_factory=int)
    isMainStream: int = field(default_factory=int)
    sync_line: int = field(default_factory=int)
    source_node_id: int = field(default_factory=int)
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

    LowCut: int = field(default_factory=int)
    HighCut: int = field(default_factory=int)


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
    low_cut: float = field(default_factory=float)
    high_cut: float = field(default_factory=float)


@dataclass
class PosTracker(OEPlugin):
    """
    Documents the PosTracker plugin
    """

    Brightness: int = field(default=20)
    Contrast: int = field(default=20)
    Exposure: int = field(default=20)
    LeftBorder: int = field(default=0)
    RightBorder: int = field(default=800)
    TopBorder: int = field(default=0)
    BottomBorder: int = field(default=600)
    AutoExposure: bool = field(default=False)
    OverlayPath: bool = field(default=False)
    sample_rate: int = field(default=30)


@dataclass
class TrackMe(OEPlugin):
    """
    Documents the TrackMe plugin
    """

    pass


@dataclass
class StimControl(OEPlugin):
    """
    Documents the StimControl plugin
    """

    Device: int = field(default_factory=int)
    Duration: int = field(default_factory=int)
    Interval: int = field(default_factory=int)
    Gate: int = field(default_factory=int)
    Output: int = field(default_factory=int)
    Start: int = field(default_factory=int)
    Stop: int = field(default_factory=int)
    Trigger: int = field(default_factory=int)


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

    nChannels: int = field(default_factory=int)
    id: int = field(default_factory=int)
    subChannels: list[int] = field(default_factory=list)
    subChannelsThresh: list[int] = field(default_factory=list)
    subChannelsActive: list[int] = field(default_factory=list)
    prePeakSamples: int = field(default=8)
    postPeakSamples: int = field(default=32)


@dataclass
class RippleDetector(OEPlugin):
    """
    Documents the Ripple Detector plugin
    """

    Ripple_Input: int = field(default_factory=int)
    Ripple_Out: int = field(default_factory=int)
    ripple_std: float = field(default_factory=float)
    time_thresh: float = field(default_factory=float)
    refr_time: float = field(default_factory=float)
    rms_samples: float = field(default_factory=float)
    mov_detect: int = field(default_factory=int)
    mov_input: int = field(default_factory=int)
    mov_out: int = field(default_factory=int)
    mov_std: float = field(default_factory=float)
    min_time_st: float = field(default_factory=float)
    min_time_mov: float = field(default_factory=float)


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
    else:
        return


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

    def __init__(self, pname: str):
        self.filename = None
        import os

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
