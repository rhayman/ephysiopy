"""
Classes for parsing information contained in the settings.xml
file that is saved when recording with the openephys system.
"""
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from abc import ABC
from dataclasses import dataclass, field
from typing import Callable, List
import xml

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
        if value == '1':
            self._param = True
        else:
            self._param = False

    @property
    def record(self) -> bool:
        return self._record

    @record.setter
    def record(self, value: str) -> None:
        if value == '1':
            self._record = True
        else:
            self._record = False

    @property
    def audio(self) -> bool:
        return self._audio

    @audio.setter
    def audio(self, value: str) -> None:
        if value == '1':
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
class OEPlugin(ABC):
    """
    Documents an OE plugin
    """
    name: str = field(default=None)
    insertionPoint: int = field(default=None)
    pluginName: str = field(default=None)
    pluginType: int = field(default=None)
    pluginIndex: int = field(default=None)
    libraryName: str = field(default=None)
    libraryVersion: int = field(default=None)
    NodeId: int = field(default=None)
    Type: str = field(default=None)
    isSource: bool = field(default=True)
    isSink: bool = field(default=False)


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
class BandpassFilter(OEPlugin):
    """
    Documents the Bandpass Filter plugin
    """
    name = "Filters/Bandpass Filter"
    pluginName = "Bandpass Filter"
    pluginType = 1
    libraryName = "Bandpass Filter"
    isSource = False
    isSink = False

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
    '''
    Recursive function that applies func to each node
    '''
    if node != None:
        func(node, cls)
        for item in node:
            recurseNode(item, func, cls)
    else:
        return


def addValuesToDataClass(node: xml.etree.ElementTree.Element, cls: dataclass):
    for i in node.items():
        if hasattr(cls, i[0]):
            setattr(cls, i[0], i[1])
    if hasattr(cls, 'channel_info') and node.tag == 'CHANNEL':
        if cls.channel_info is None:
            cls.channel_info = list()
        chan = Channel()
        recurseNode(node, addValuesToDataClass, chan)
        cls.channel_info.append(chan)

                
class OEStructure(object):
    """
    Loads up the structure.oebin file for openephys flat binary
    format recordings
    
    self.data is a dict
    """
    def __init__(self, pname: str):
        self.filename = None
        self.data = None
        import os
        for d, _, f in os.walk(pname):
            for ff in f:
                if 'structure.oebin' in ff:
                    self.filename = os.path.join(d, "structure.oebin")
        if self.filename is not None:
            import json
            with open(self.filename, 'r') as f:
                self.data = json.load(f)
        

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
                if 'settings.xml' in ff:
                    self.filename = os.path.join(d, "settings.xml")
        self.tree = None
        """
        It's not uncommon to have > 1 of the same type of processor, i.e.
        2 x bandpass filter to look at LFP and APs. This deals with that...
        """
        self.processors = OrderedDict()
        self.electrodes = OrderedDict()
        self.tracker_params = {}
        self.stimcontrol_params = {}
        self.bandpass_params = OrderedDict()

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
            for elem in self.tree.iter('PROCESSOR'):
                this_proc = elem.get('name')
                if this_proc == 'Sources/Pos Tracker':
                    pos_tracker = PosTracker()
                    recurseNode(elem, addValuesToDataClass, pos_tracker)
                    self.processors[this_proc] = pos_tracker
                if this_proc == 'Sources/Rhythm FPGA':
                    fpga = RhythmFPGA()
                    recurseNode(elem, addValuesToDataClass, fpga)
                    self.processors[this_proc] = fpga
                if this_proc == 'Sources/Neuropix-PXI':
                    npx = NeuropixPXI()
                    recurseNode(elem, addValuesToDataClass, npx)
                    self.processors[this_proc] = npx
                if this_proc == 'Filters/Spike Sorter':
                    spike_sorter = SpikeSorter()
                    recurseNode(elem, addValuesToDataClass, spike_sorter)
                

    # def parseSpikeSorter(self):
    #     """
    #     Parses data attached to each ELECTRODE object in the xml tree
    #     """
    #     if len(self.processors) == 0:
    #         self.parse()
    #     electrode_info = OrderedDict()
    #     for child in self.processors['Filters/Spike Sorter'][0].iter():
    #         if 'SpikeSorter' in child.tag:
    #             for grandchild in child.iter():
    #                 if 'ELECTRODE' == grandchild.tag:
    #                     for this_electrode in grandchild.iter('ELECTRODE'):
    #                         info_obj = Electrode()
    #                         info_obj.name = this_electrode.get('name')
    #                         info_obj.nChannels = this_electrode.get(
    #                             'numChannels')
    #                         info_obj.prePeakSamples = this_electrode.get(
    #                             'prePeakSamples')
    #                         info_obj.postPeakSamples = this_electrode.get
    #                         ('postPeakSamples')
    #                         info_obj.id = this_electrode.get('electrodeID')
    #                         subchan = []
    #                         subchanThresh = []
    #                         subchanActive = []
    #                         for ggrandkid in grandchild.iter():
    #                             if 'SUBCHANNEL' == ggrandkid.tag:
    #                                 for schan in ggrandkid.iter('SUBCHANNEL'):
    #                                     subchan.append(schan.get('ch'))
    #                                     subchanThresh.append(
    #                                         schan.get('thresh'))
    #                                     subchanActive.append(
    #                                         schan.get('isActive'))

    #                         info_obj.subChannels = subchan
    #                         info_obj.subChannelsThresh = subchanThresh
    #                         info_obj.subChannelsActive = subchanActive
    #                         electrode_info[info_obj.id] = info_obj
    #     self.electrodes = electrode_info

    def parseStimControl(self):
        """
        Parses information attached to the StimControl module I wrote
        """
        if len(self.processors) == 0:
            self.parse()
        children = self.processors['Sinks/StimControl'][0].iter()
        for child in children:
            if 'Parameters' in child.tag:
                self.stimcontrol_params = child.attrib
        # convert string values to ints
        self.stimcontrol_params = dict(
            [k, int(v)] for k, v in self.stimcontrol_params.items())

    

    # def parseBandpassFilters(self):
    #     """
    #     Parse the bandpass filter information
    #     """
    #     if len(self.processors) == 0:
    #         self.parse()
    #     bandpass_info = OrderedDict()
    #     for bpf in self.processors['Filters/Bandpass Filter']:
    #         if 'PROCESSOR' in bpf.tag:
    #             this_bpf = BandpassFilter()
    #             this_bpf.nodeId = bpf.get('NodeId')
    #             channels = []
    #             for child in bpf.iter('CHANNEL'):
    #                 this_chan = ChannelInfo()
    #                 this_chan.number = child.get('number')
    #                 this_chan.name = child.get('name')
    #                 for state in child.iter('SELECTIONSTATE'):
    #                     this_chan.param = state.get('param')
    #                     this_chan.record = state.get('record')
    #                 for params in child.iter('PARAMETERS'):
    #                     this_chan.lowcut = params.get('lowcut')
    #                     this_chan.highcut = params.get('highcut')
    #                 channels.append(this_chan)
    #                 bandpass_info[this_bpf.nodeId] = channels
    #     self.bandpass_params = bandpass_info
