"""
Classes for parsing information contained in the settings.xml
file that is saved when recording with the openephys system.
"""
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict


class ChannelInfo(object):
    """
    Documents the information attached to each channel
    """
    def __init__(self):
        self._name = None
        self._number = -1
        self._gain = 0
        self._param = False
        self._record = False
        self._audio = False
        self._lowcut = 0
        self._highcut = 0

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        self._name = val

    @property
    def number(self):
        return self._number

    @number.setter
    def number(self, val):
        self._number = int(val)

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, val):
        self._gain = float(val)

    @property
    def param(self):
        return self._param

    @param.setter
    def param(self, val):
        self._param = bool(val)

    @property
    def record(self):
        return self._record

    @record.setter
    def record(self, val):
        self._record = bool(val)

    @property
    def audio(self):
        return self._audio

    @audio.setter
    def audio(self, val):
        self._audio = bool(val)

    @property
    def lowcut(self):
        return self._lowcut

    @lowcut.setter
    def lowcut(self, val):
        self._lowcut = int(val)

    @property
    def highcut(self):
        return self._highcut

    @highcut.setter
    def highcut(self, val):
        self._highcut = int(val)


class BandpassFilter(object):
    """
    Documents the Bandpass Filter plugin
    """
    def __init__(self):
        self._nodeId = None
        self._channels = None  # becomes a list of int
        self._lowcuts = None  # becomes list of int
        self._highcuts = None  # ditto

    @property
    def nodeId(self):
        return self._nodeId

    @nodeId.setter
    def nodeId(self, val):
        self._nodeId = val

    @property
    def channels(self):
        return self._channels

    @channels.setter
    def channels(self, val):
        self._channels = [int(v) for v in val]

    @property
    def lowcuts(self):
        return self._lowcuts

    @lowcuts.setter
    def lowcuts(self, val):
        self._lowcuts = [int(v) for v in val]

    @property
    def highcuts(self):
        return self._highcuts

    @highcuts.setter
    def highcuts(self, val):
        self._highcuts = [int(v) for v in val]


class Electrode(object):
    """
    Documents the ELECTRODE entries in the settings.xml file
    """
    def __init__(self):
        self._nChannels = None  # int
        self._prePeakSamples = 8
        self._postPeakSamples = 32
        self._id = None
        self._subChannels = None  # becomes a list of int
        self._subChannelsThresh = None  # becomes a list of float
        self._subChannelsActive = None  # becomes a list of bool

    @property
    def nChannels(self):
        return self._nChannels

    @nChannels.setter
    def nChannels(self, val):
        self._nChannels = int(val)

    @property
    def prePeakSamples(self):
        return self._prePeakSamples

    @prePeakSamples.setter
    def prePeakSamples(self, val):
        self._prePeakSamples = int(val)

    @property
    def postPeakSamples(self):
        return self._postPeakSamples

    @postPeakSamples.setter
    def postPeakSamples(self, val):
        self._postPeakSamples = int(val)

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, val):
        self._id = val

    @property
    def subChannels(self):
        return self._subChannels

    @subChannels.setter
    def subChannels(self, val):
        self._subChannels = [int(v) for v in val]

    @property
    def subChannelsThresh(self):
        return self._subChannelsThresh

    @subChannelsThresh.setter
    def subChannelsThresh(self, val):
        self._subChannelsThresh = [float(v) for v in val]

    @property
    def subChannelsActive(self):
        return self._subChannelsActive

    @subChannelsActive.setter
    def subChannelsActive(self, val):
        self._subChannelsActive = [bool(int(v)) for v in val]


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
        self.processors = defaultdict(list)
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
                self.processors[elem.attrib['name']].append(elem)
        # in a try / except because might be Neuropixels probe so
        # no Rhythm FPGA
        try:
            fpga_items = dict(
                self.processors['Sources/Rhythm FPGA'][0].items())
            self.fpga_nodeId = fpga_items['NodeId']
        except Exception:
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

    def parseChannels(self):
        """
        Parses data attached to each channel
        """
        if len(self.processors) == 0:
            self.parse()
        channel_info = OrderedDict()
        for chan_info in self.processors['Sources/Rhythm FPGA'][0].iter():
            if 'CHANNEL_INFO' in chan_info.tag:
                for this_chan in chan_info.iter('CHANNEL'):
                    info_obj = ChannelInfo()
                    info_obj.number = this_chan.get('number')
                    info_obj.name = this_chan.get('name')
                    info_obj.gain = this_chan.get('gain')
                    channel_info[info_obj.number] = info_obj
            if 'CHANNEL' in chan_info.tag:
                for chan_state in chan_info.iter('CHANNEL'):
                    num = int(chan_state.get('number'))
                    for i in channel_info.keys():
                        if i == num:
                            info_obj = channel_info[i]
                            for state in chan_state.iter('SELECTIONSTATE'):
                                info_obj.param = state.get('param')
                                info_obj.record = state.get('record')
                            channel_info[i] = info_obj
        self.channel_info = channel_info

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
