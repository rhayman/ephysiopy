"""
Classes for parsing information contained in the settings.xml
file that is saved when recording with the openephys system.
"""
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

import xml
from abc import ABC
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, List


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
    _lowcut: int = field(default=None)
    _highcut: int = field(default=None)

    @property
    def number(self) -> int:
        return self._number

    @number.setter
    def number(self, value: str) -> None:
        self._number = int(value)

    @property
    def gain(self) -> int:
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
class Stream():
    """
    Documents an OE DatasSream
    """

    name: str = field(default=None)
    description: str = field(default=None)
    sample_rate: int = field(default=None)
    channel_count: int = field(default=None)


@dataclass
class OEPlugin(ABC):
    """
    Documents an OE plugin
    """

    name: str = field(default=None)
    insertionPoint: int = field(default=None)
    pluginName: str = field(default=None)
    type: int = field(default=None)
    index: int = field(default=None)
    libraryName: str = field(default=None)
    libraryVersion: int = field(default=None)
    processorType: int = field(default=None)
    nodeId: int = field(default=None)
    channel_count: int = field(default=None)
    stream: Stream = field(default=None)
    sample_rate: int = field(default=None)


@dataclass
class RecordNode(OEPlugin):
    """
    Documents the RecordNode plugin
    """

    path: str = field(default=None)
    engine: str = field(default=None)
    recordEvents: int = field(default=None)
    recordSpikes: int = field(default=None)
    isMainStream: int = field(default=None)
    sync_line: int = field(default=None)
    source_node_id: int = field(default=None)
    recording_state: str = field(default=None)


@dataclass
class RhythmFPGA(OEPlugin):
    """
    Documents the Rhythm FPGA plugin
    """

    channel_info: List[Channel] = field(default=None)


# Identical to the RhythmFPGA class above for now
@dataclass
class NeuropixPXI(OEPlugin):
    """
    Documents the Neuropixels-PXI plugin
    """

    channel_info: List[Channel] = field(default=None)


@dataclass
class AcquisitionBoard(OEPlugin):
    """
    Documents the Acquisition Board plugin
    """

    LowCut: int = field(default=None)
    HighCut: int = field(default=None)


@dataclass
class BandpassFilter(OEPlugin):
    """
    Documents the Bandpass Filter plugin
    """

    name = "Bandpass Filter"
    pluginName = "Bandpass Filter"
    pluginType = 1
    libraryName = "Bandpass Filter"

    channels: List[int] = field(default_factory=List)
    lowcut: List[int] = field(default_factory=List)
    highcut: List[int] = field(default_factory=List)


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
    Device: int = field(default=None)
    Duration: int = field(default=None)
    Interval: int = field(default=None)
    Gate: int = field(default=None)
    Output: int = field(default=None)
    Start: int = field(default=None)
    Stop: int = field(default=None)
    Trigger: int = field(default=None)


@dataclass
class SpikeSorter(OEPlugin):
    pass


@dataclass
class Electrode(object):
    """
    Documents the ELECTRODE entries in the settings.xml file
    """

    nChannels: int = field(default_factory=int)
    id: int = field(default_factory=int)
    subChannels: List[int] = field(default_factory=List)
    subChannelsThresh: List[int] = field(default_factory=List)
    subChannelsActive: List[int] = field(default_factory=List)
    prePeakSamples: int = field(default=8)
    postPeakSamples: int = field(default=32)


class AbstractProcessorFactory():
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


class ProcessorFactory():
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
        else:
            return self.factory.create_oe_plugin()


def recurseNode(
                node: xml.etree.ElementTree.Element,
                func: Callable,
                cls: dataclass):
    """
    Recursive function that applies func to each node
    """
    if node is not None:
        func(node, cls)
        for item in node:
            recurseNode(item, func, cls)
    else:
        return


def addValues2Class(node: xml.etree.ElementTree.Element, cls: dataclass):
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

    def __init__(self, fname: str):
        self.filename = []
        self.data = []
        import json

        self.filename.append(fname)
        with open(fname, "r") as f:
            self.data.append(json.load(f))


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
                if "/" in i_proc:
                    i_proc = i_proc.split("/")[-1]
                new_processor = processor_factory.create_processor(i_proc)
                recurseNode(elem, addValues2Class, new_processor)
                if i_proc == "Record Node":
                    if new_processor.nodeId is not None:
                        self.record_nodes[i_proc + " " + new_processor.nodeId] = new_processor
                    else:
                        self.record_nodes[i_proc] = new_processor
                else:
                    if new_processor.nodeId is not None:
                        self.processors[i_proc + " " + new_processor.nodeId] = new_processor
                    else:
                        self.processors[i_proc] = new_processor
