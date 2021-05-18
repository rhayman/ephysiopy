"""
Classes for parsing information contained in the settings.xml
file that is saved when recording with the openephys system.
"""
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import List


@dataclass
class ChannelInfo(object):
    """
    Documents the information attached to each channel
    """
    name: str = None
    number: int = -1
    gain: float = 0
    param: bool = False
    record: bool = False
    audio: bool = False
    lowcut: int = 0
    highcut: int = 0


@dataclass
class BandpassFilter(object):
    """
    Documents the Bandpass Filter plugin
    """
    nodeId: int
    channels: List[int]
    lowcuts: List[int]
    highcuts: List[int]


@dataclass
class Electrode(object):
    """
    Documents the ELECTRODE entries in the settings.xml file
    """
    nChannels: int
    id: int
    subChannels: List[int]
    subChannelsThresh: List[int]
    subChannelsActive: List[int]
    prePeakSamples: int = 8
    postPeakSamples: int = 32


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
        self.fpga_nodeId = None
        """
        It's not uncommon to have > 1 of the same type of processor, i.e.
        2 x bandpass filter to look at LFP and APs. This deals with that...
        """
        self.processors = defaultdict()
        self.electrodes = OrderedDict()
        self.tracker_params = {}
        self.stimcontrol_params = {}
        self.bandpass_params = OrderedDict()

    def load(self):
        """
        Creates a handle to the basic xml document
        """
        if self.filename is not None:
            self.tree = ET.ElementTree(file=self.filename)

    def parse(self):
        """
        Parses the basic information attached to the FPGA module
        """
        if self.tree is None:
            self.load()
        # quick hack to deal with flat binary format that has no settings.xml
        if self.tree is not None:
            for elem in self.tree.iter(tag='PROCESSOR'):
                self.processors[elem.attrib['name']] = elem
        # Check if FPGA present - value needed for navigating .nwb file
        if 'Sources/Rhythm FPGA' in self.processors:
            children = self.processors['Sources/Rhythm FPGA'][0].iter()
            for child in children:
                if 'PROCESSOR' in child.tag:
                    self.fpga_nodeId = child.attrib['NodeId']
        else:
            self.fpga_nodeId = None

    def parsePos(self):
        """
        Parses the information attached to the PosTracker plugin I wrote
        """
        if len(self.processors) == 0:
            self.parse()

        if len(self.processors) > 0:  # hack for no settings.xml file
            if self.processors['Sources/Pos Tracker']:
                children = self.processors['Sources/Pos Tracker'][0].iter()
                for child in children:
                    if 'Parameters' in child.tag:
                        self.tracker_params = child.attrib
                # convert string values to ints
                self.tracker_params = dict(
                    [k, int(v)] for k, v in self.tracker_params.items())
            else:
                self.tracker_params = {
                    'Brightness': 20, 'Contrast': 20, 'Exposure': 20,
                    'AutoExposure': 0, 'OverlayPath': 1,
                    'LeftBorder': 0, 'RightBorder': 800, 'TopBorder': 0,
                    'BottomBorder': 600}
        else:
            self.tracker_params = {
                    'Brightness': 20, 'Contrast': 20, 'Exposure': 20,
                    'AutoExposure': 0, 'OverlayPath': 1,
                    'LeftBorder': 0, 'RightBorder': 800, 'TopBorder': 0,
                    'BottomBorder': 600}

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
