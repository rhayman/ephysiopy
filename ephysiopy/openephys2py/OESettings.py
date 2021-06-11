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
from typing import List


@dataclass
class Channel(object):
    """
    Documents the information attached to each channel
    """
    name: str = field(default_factory=str)
    number: int = field(default_factory=int)
    gain: float = field(default_factory=float)
    param: bool = field(default_factory=bool)
    record: bool = field(default=False)
    audio: bool = field(default=False)
    lowcut: int = field(default=None)
    highcut: int = field(default=None)


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
    channel_info: List[Channel] = field(default=list)
    sample_rate: int = field(default=None)
        

# Identical to the RhythmFPGA class above for now
@dataclass
class NeuropixPXI(OEPlugin):
    """
    Documents the Neuropixels-PXI plugin
    """
    channel_info: List[Channel] = field(default=list)
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
                    for child in elem.attrib.items():
                        if hasattr(pos_tracker, child[0]):
                            setattr(pos_tracker, child[0], child[1])
                    for params in elem.iter('Parameters'):
                        for param in params.items():
                            if hasattr(pos_tracker, param[0]):
                                setattr(pos_tracker, param[0], param[1])
                    self.processors[this_proc] = pos_tracker
                if this_proc == 'Sources/Rhythm FPGA':
                    fpga = RhythmFPGA()
                    for child in elem.attrib.items():
                        if hasattr(fpga, child[0]):
                            setattr(fpga, child[0], child[1])
                    self.processors[this_proc] = fpga
                if this_proc == 'Sources/Neuropix-PXI':
                    npx = NeuropixPXI()
                    for child in elem.attrib.items():
                        if hasattr(npx, child[0]):
                            setattr(npx, child[0], child[1])
                    self.processors[this_proc] = npx
                

    def parseSpikeSorter(self):
        """
        Parses data attached to each ELECTRODE object in the xml tree
        """
        if len(self.processors) == 0:
            self.parse()
        electrode_info = OrderedDict()
        for child in self.processors['Filters/Spike Sorter'][0].iter():
            if 'SpikeSorter' in child.tag:
                for grandchild in child.iter():
                    if 'ELECTRODE' == grandchild.tag:
                        for this_electrode in grandchild.iter('ELECTRODE'):
                            info_obj = Electrode()
                            info_obj.name = this_electrode.get('name')
                            info_obj.nChannels = this_electrode.get(
                                'numChannels')
                            info_obj.prePeakSamples = this_electrode.get(
                                'prePeakSamples')
                            info_obj.postPeakSamples = this_electrode.get
                            ('postPeakSamples')
                            info_obj.id = this_electrode.get('electrodeID')
                            subchan = []
                            subchanThresh = []
                            subchanActive = []
                            for ggrandkid in grandchild.iter():
                                if 'SUBCHANNEL' == ggrandkid.tag:
                                    for schan in ggrandkid.iter('SUBCHANNEL'):
                                        subchan.append(schan.get('ch'))
                                        subchanThresh.append(
                                            schan.get('thresh'))
                                        subchanActive.append(
                                            schan.get('isActive'))

                            info_obj.subChannels = subchan
                            info_obj.subChannelsThresh = subchanThresh
                            info_obj.subChannelsActive = subchanActive
                            electrode_info[info_obj.id] = info_obj
        self.electrodes = electrode_info

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

    def parseProcessor(self, proc_type: str = 'Sources/Rhythm FPGA'):
        """
        Parses data attached to each processor

        Parameters
        ----------
        proc_type: str
            Legal values are anything in self.processors
            Examples: 'Sources/Rhythm FPGA', 'Sources/Neuropix-PXI',
            'Sinks/Probe Viewer', 'Sources/Pos Tracker'

        """
        if len(self.processors) == 0:
            self.parse()
        channel_info = OrderedDict()
        channel = OrderedDict()
        if self.processors[proc_type]:
            for chan_info in self.processors[proc_type].iter():
                if 'CHANNEL_INFO' in chan_info.tag:
                    for this_chan in chan_info.iter('CHANNEL'):
                        info_obj = ChannelInfo()
                        info_obj.number = this_chan.get('number')
                        info_obj.name = this_chan.get('name')
                        info_obj.gain = this_chan.get('gain')
                        channel_info[info_obj.number] = info_obj
                if 'CHANNEL' in chan_info.tag:
                    for chan_state in chan_info.iter('CHANNEL'):
                        info_obj = ChannelInfo()
                        info_obj.number = chan_state.get('number')
                        info_obj.name = chan_state.get('name')
                        for state in chan_state.iter('SELECTIONSTATE'):
                            info_obj.param = state.get('param')
                            info_obj.record = state.get('record')
                            info_obj.audio = state.get('audio')
                        channel[info_obj.number] = info_obj
        self.channel_info = channel_info
        self.channel = channel

    def parseBandpassFilters(self):
        """
        Parse the bandpass filter information
        """
        if len(self.processors) == 0:
            self.parse()
        bandpass_info = OrderedDict()
        for bpf in self.processors['Filters/Bandpass Filter']:
            if 'PROCESSOR' in bpf.tag:
                this_bpf = BandpassFilter()
                this_bpf.nodeId = bpf.get('NodeId')
                channels = []
                for child in bpf.iter('CHANNEL'):
                    this_chan = ChannelInfo()
                    this_chan.number = child.get('number')
                    this_chan.name = child.get('name')
                    for state in child.iter('SELECTIONSTATE'):
                        this_chan.param = state.get('param')
                        this_chan.record = state.get('record')
                    for params in child.iter('PARAMETERS'):
                        this_chan.lowcut = params.get('lowcut')
                        this_chan.highcut = params.get('highcut')
                    channels.append(this_chan)
                    bandpass_info[this_bpf.nodeId] = channels
        self.bandpass_params = bandpass_info
