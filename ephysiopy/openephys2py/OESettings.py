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
    sample_rate: int = field(default=None)


# Identical to the RhythmFPGA class above for now
@dataclass
class NeuropixPXI(OEPlugin):
    """
    Documents the Neuropixels-PXI plugin
    """

    channel_info: List[Channel] = field(default=None)
    sample_rate: int = field(default=None)


@dataclass
class AcquisitionBoard(OEPlugin):
    """
    Documents the Acquisition Board plugin
    """

    sample_rate: int = field(default=None)
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
    lowcuts: List[int] = field(default_factory=List)
    highcuts: List[int] = field(default_factory=List)


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


def recurseNode(node: xml.etree.ElementTree.Element, func: Callable, cls: dataclass):
    """
    Recursive function that applies func to each node
    """
    if node is not None:
        func(node, cls)
        for item in node:
            recurseNode(item, func, cls)
    else:
        return


def addValuesToDataClass(node: xml.etree.ElementTree.Element, cls: dataclass):
    for i in node.items():
        if hasattr(cls, i[0]):
            setattr(cls, i[0], i[1])
    if hasattr(cls, "channel_info") and node.tag == "CHANNEL":
        if cls.channel_info is None:
            cls.channel_info = list()
        chan = Channel()
        recurseNode(node, addValuesToDataClass, chan)
        cls.channel_info.append(chan)
    if hasattr(cls, "stream") and node.tag == "STREAM":
        if cls.stream is None:
            cls.stream = Stream()
        recurseNode(node, addValuesToDataClass, cls.stream)


class OEStructure(object):
    """
    Loads up the structure.oebin file for openephys flat binary
    format recordings
    """

    def __init__(self, pname: str):
        self.filename = []
        self.data = []
        import os
        import json

        for d, _, f in os.walk(pname):
            for ff in f:
                if "structure.oebin" in ff:
                    self.filename.append(os.path.join(d, ff))
                    with open(os.path.join(d, ff), "r") as f:
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
        """
        It's not uncommon to have > 1 of the same type of processor, i.e.
        2 x bandpass filter to look at LFP and APs. This deals with that...
        """
        self.possible_processors = OrderedDict([
            ("Pos Tracker", PosTracker()),
            ("Rhythm FPGA", RhythmFPGA()),
            ("Neuropix-PXI", NeuropixPXI()),
            ("Acquisition Board", AcquisitionBoard()),
            ("Spike Sorter", SpikeSorter()),
            ("TrackMe", TrackMe()),
            ("Record Node", RecordNode())
        ])
        self.processors = OrderedDict()
        self.record_nodes = OrderedDict()
        self.tracker_params = {}
        self.stimcontrol_params = {}
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
        # quick hack to deal with flat binary format that has no settings.xml
        if self.tree is not None:
            for elem in self.tree.iter("PROCESSOR"):
                this_proc = elem.get("name")
                if this_proc == "Record Node":  # special as could be > 1
                    rec_node = RecordNode()
                    recurseNode(elem, addValuesToDataClass, rec_node)
                    self.record_nodes[this_proc + " " + rec_node.nodeId] = rec_node
                elif this_proc in self.possible_processors.keys():
                    self.processors[this_proc] = self.possible_processors[this_proc]
                    recurseNode(elem, addValuesToDataClass, self.processors[this_proc])
                else:
                    self.processors[this_proc] = OEPlugin()
                    recurseNode(elem, addValuesToDataClass, self.processors[this_proc])
                
    def parseStimControl(self):
        """
        Parses information attached to the StimControl module I wrote
        """
        if len(self.processors) == 0:
            self.parse()
        children = self.processors["StimControl"][0].iter()
        for child in children:
            if "Parameters" in child.tag:
                self.stimcontrol_params = child.attrib
        # convert string values to ints
        self.stimcontrol_params = dict(
            [k, int(v)] for k, v in self.stimcontrol_params.items()
        )
