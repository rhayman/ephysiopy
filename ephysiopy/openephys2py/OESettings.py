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
    """
    Descriptor class for converting attribute values to integers.

    Parameters
    ----------
    default : int
        The default value to return if the attribute is not set.

    Methods
    -------
    __set_name__(owner, name)
        Sets the internal name for the attribute.
    __get__(obj, type)
        Retrieves the attribute value, returning the default if not set.
    __set__(obj, value)
        Sets the attribute value, converting it to an integer.
    """

    def __init__(self, *, default):
        """
        Initialize the IntConversion descriptor.

        Parameters
        ----------
        default : int
            The default value to return if the attribute is not set.
        """
        self._default = default

    def __set_name__(self, owner, name):
        """
        Set the internal name for the attribute.

        Parameters
        ----------
        owner : type
            The owner class where the descriptor is defined.
        name : str
            The name of the attribute.
        """
        self._name = "_" + name

    def __get__(self, obj, type):
        """
        Retrieve the attribute value.

        Parameters
        ----------
        obj : object
            The instance of the owner class.
        type : type
            The owner class type.

        Returns
        -------
        int
            The attribute value or the default value if not set.
        """
        if obj is None:
            return self._default
        return getattr(obj, self._name, self._default)

    def __set__(self, obj, value):
        """
        Set the attribute value, converting it to an integer.

        Parameters
        ----------
        obj : object
            The instance of the owner class.
        value : any
            The value to set, which will be converted to an integer.
        """
        if value == "50.0":
            breakpoint()
        setattr(obj, self._name, int(value))


class FloatConversion:
    """
    Descriptor class for converting attribute values to floats.

    Parameters
    ----------
    default : float
        The default value to return if the attribute is not set.

    Methods
    -------
    __set_name__(owner, name)
        Sets the internal name for the attribute.
    __get__(obj, type)
        Retrieves the attribute value, returning the default if not set.
    __set__(obj, value)
        Sets the attribute value, converting it to a float.
    """

    def __init__(self, *, default):
        """
        Initialize the FloatConversion descriptor.

        Parameters
        ----------
        default : float
            The default value to return if the attribute is not set.
        """
        self._default = default

    def __set_name__(self, owner, name):
        """
        Set the internal name for the attribute.

        Parameters
        ----------
        owner : type
            The owner class where the descriptor is defined.
        name : str
            The name of the attribute.
        """
        self._name = "_" + name

    def __get__(self, obj, type):
        """
        Retrieve the attribute value.

        Parameters
        ----------
        obj : object
            The instance of the owner class.
        type : type
            The owner class type.

        Returns
        -------
        float
            The attribute value or the default value if not set.
        """
        if obj is None:
            return self._default
        return getattr(obj, self._name, self._default)

    def __set__(self, obj, value):
        """
        Set the attribute value, converting it to a float.

        Parameters
        ----------
        obj : object
            The instance of the owner class.
        value : any
            The value to set, which will be converted to a float.
        """
        setattr(obj, self._name, float(value))


@dataclass
class Channel:
    """
    Documents the information attached to each channel.

    Attributes
    ----------
    name : str
        The name of the channel.
    number : int
        The channel number, converted from a string.
    gain : float
        The gain value, converted from a string.
    param : bool
        A boolean parameter, converted from a string ("1" for True, otherwise False).
    record : bool
        A boolean indicating if the channel is recorded, converted from a string ("1" for True, otherwise False).
    audio : bool
        A boolean indicating if the channel is audio, converted from a string ("1" for True, otherwise False).
    lowcut : float
        The low cut frequency, converted from a string.
    highcut : float
        The high cut frequency, converted from a string.
    """

    name: str = field(default_factory=str)
    _number: IntConversion = IntConversion(default=0)
    _gain: FloatConversion = FloatConversion(default=0)
    _param: bool = field(default_factory=bool)
    _record: bool = field(default=False)
    _audio: bool = field(default=False)
    _lowcut: FloatConversion = FloatConversion(default=0)
    _highcut: FloatConversion = FloatConversion(default=0)

    @property
    def number(self) -> int:
        """
        Get the channel number.

        Returns
        -------
        int
            The channel number.
        """
        return self._number

    @number.setter
    def number(self, value: str) -> None:
        """
        Set the channel number.

        Parameters
        ----------
        value : str
            The channel number as a string.
        """
        self._number = int(value)

    @property
    def gain(self) -> float:
        """
        Get the gain value.

        Returns
        -------
        float
            The gain value.
        """
        return self._gain

    @gain.setter
    def gain(self, value: str) -> None:
        """
        Set the gain value.

        Parameters
        ----------
        value : str
            The gain value as a string.
        """
        self._gain = float(value)

    @property
    def param(self) -> bool:
        """
        Get the boolean parameter.

        Returns
        -------
        bool
            The boolean parameter.
        """
        return self._param

    @param.setter
    def param(self, value: str) -> None:
        """
        Set the boolean parameter.

        Parameters
        ----------
        value : str
            The boolean parameter as a string ("1" for True, otherwise False).
        """
        self._param = value == "1"

    @property
    def record(self) -> bool:
        """
        Get the record status.

        Returns
        -------
        bool
            The record status.
        """
        return self._record

    @record.setter
    def record(self, value: str) -> None:
        """
        Set the record status.

        Parameters
        ----------
        value : str
            The record status as a string ("1" for True, otherwise False).
        """
        self._record = value == "1"

    @property
    def audio(self) -> bool:
        """
        Get the audio status.

        Returns
        -------
        bool
            The audio status.
        """
        return self._audio

    @audio.setter
    def audio(self, value: str) -> None:
        """
        Set the audio status.

        Parameters
        ----------
        value : str
            The audio status as a string ("1" for True, otherwise False).
        """
        self._audio = value == "1"

    @property
    def lowcut(self) -> float:
        """
        Get the low cut frequency.

        Returns
        -------
        float
            The low cut frequency.
        """
        return self._lowcut

    @lowcut.setter
    def lowcut(self, value: str) -> None:
        """
        Set the low cut frequency.

        Parameters
        ----------
        value : str
            The low cut frequency as a string.
        """
        self._lowcut = float(value)

    @property
    def highcut(self) -> float:
        """
        Get the high cut frequency.

        Returns
        -------
        float
            The high cut frequency.
        """
        return self._highcut

    @highcut.setter
    def highcut(self, value: str) -> None:
        """
        Set the high cut frequency.

        Parameters
        ----------
        value : str
            The high cut frequency as a string.
        """
        self._highcut = float(value)


@dataclass
class Stream:
    """
    Documents an OE DataStream.

    Attributes
    ----------
    name : str
        The name of the data stream.
    description : str
        A description of the data stream.
    sample_rate : FloatConversion
        The sample rate of the data stream, converted from a string.
    channel_count : IntConversion
        The number of channels in the data stream, converted from a string.
    """

    name: str = field(default_factory=str)
    description: str = field(default_factory=str)
    sample_rate: FloatConversion = FloatConversion(default=0)
    channel_count: IntConversion = IntConversion(default=0)


@dataclass
class OEPlugin(ABC):
    """
    Documents an OE plugin.

    Attributes
    ----------
    name : str
        The name of the plugin.
    insertionPoint : IntConversion
        The insertion point of the plugin, converted from a string.
    pluginName : str
        The name of the plugin.
    type : IntConversion
        The type of the plugin, converted from a string.
    index : IntConversion
        The index of the plugin, converted from a string.
    libraryName : str
        The name of the library.
    libraryVersion : str
        The version of the library.
    processorType : IntConversion
        The type of processor, converted from a string.
    nodeId : IntConversion
        The node ID, converted from a string.
    channel_count : IntConversion
        The number of channels, converted from a string.
    stream : Stream
        The data stream associated with the plugin.
    sample_rate : FloatConversion
        The sample rate, converted from a string.
    """

    name: str = field(default_factory=str)
    insertionPoint: IntConversion = IntConversion(default=0)
    pluginName: str = field(default_factory=str)
    type: IntConversion = IntConversion(default=0)
    index: IntConversion = IntConversion(default=0)
    libraryName: str = field(default_factory=str)
    libraryVersion: str = field(default_factory=str)
    processorType: IntConversion = IntConversion(default=0)
    nodeId: IntConversion = IntConversion(default=0)
    channel_count: IntConversion = IntConversion(default=0)
    stream: Stream = field(default_factory=Stream)
    sample_rate: FloatConversion = FloatConversion(default=0)


@dataclass
class RecordNode(OEPlugin):
    """
    Documents the RecordNode plugin.

    Attributes
    ----------
    path : str
        The file path associated with the RecordNode.
    engine : str
        The engine used by the RecordNode.
    recordEvents : IntConversion
        Indicates if events are recorded, converted from a string.
    recordSpikes : IntConversion
        Indicates if spikes are recorded, converted from a string.
    isMainStream : IntConversion
        Indicates if this is the main stream, converted from a string.
    sync_line : IntConversion
        The sync line, converted from a string.
    source_node_id : IntConversion
        The source node ID, converted from a string.
    recording_state : str
        The recording state of the RecordNode.
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
    Documents the Rhythm FPGA plugin.

    Attributes
    ----------
    channel_info : list of Channel
        A list containing information about each channel.
    """

    channel_info: list[Channel] = field(default_factory=list)


# Identical to the RhythmFPGA class above for now
@dataclass
class NeuropixPXI(OEPlugin):
    """
    Documents the Neuropixels-PXI plugin.

    Attributes
    ----------
    channel_info : list of Channel
        A list containing information about each channel.
    """

    channel_info: list[Channel] = field(default_factory=list)


@dataclass
class AcquisitionBoard(OEPlugin):
    """
    Documents the Acquisition Board plugin

    Attributes
    ----------
    LowCut : FloatConversion
        The low cut-off frequency for the acquisition board.
    HighCut : FloatConversion
        The high cut-off frequency for the acquisition board.
    """

    LowCut: FloatConversion = FloatConversion(default=0)
    HighCut: FloatConversion = FloatConversion(default=0)


@dataclass
class BandpassFilter(OEPlugin):
    """
    Documents the Bandpass Filter plugin

    Attributes
    ----------
    name : str
        The name of the plugin.
    pluginName : str
        The display name of the plugin.
    pluginType : int
        The type identifier for the plugin.
    libraryName : str
        The library name of the plugin.
    channels : list of int
        The list of channels to which the filter is applied.
    low_cut : FloatConversion
        The low cut-off frequency for the bandpass filter.
    high_cut : FloatConversion
        The high cut-off frequency for the bandpass filter.
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
        """
        Load Tracking Port data from a specified path.

        Parameters
        ----------
        path2data : Path
            The path to the directory containing the data file.

        Returns
        -------
        np.ndarray
            A 2D numpy array with the position data.
        """
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
        """
        Load timestamps from a specified path.

        Parameters
        ----------
        path2data : Path
            The path to the directory containing the timestamps file.

        Returns
        -------
        np.ndarray
            A numpy array containing the timestamps.
        """
        return np.load(path2data / Path("timestamps.npy"))


@dataclass
class PosTracker(OEPlugin):
    """
    Documents the PosTracker plugin.

    Attributes
    ----------
    Brightness : IntConversion
        Brightness setting for the tracker, default is 20.
    Contrast : IntConversion
        Contrast setting for the tracker, default is 20.
    Exposure : IntConversion
        Exposure setting for the tracker, default is 20.
    LeftBorder : IntConversion
        Left border setting for the tracker, default is 0.
    RightBorder : IntConversion
        Right border setting for the tracker, default is 800.
    TopBorder : IntConversion
        Top border setting for the tracker, default is 0.
    BottomBorder : IntConversion
        Bottom border setting for the tracker, default is 600.
    AutoExposure : bool
        Auto exposure setting for the tracker, default is False.
    OverlayPath : bool
        Overlay path setting for the tracker, default is False.
    sample_rate : IntConversion
        Sample rate setting for the tracker, default is 30.

    Methods
    -------
    load(path2data: Path) -> np.ndarray
        Load Tracking Port data from a specified path.
    load_times(path2data: Path) -> np.ndarray
        Load timestamps from a specified path.
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

    def load(self, path2data: Path) -> np.ndarray:
        """
        Load Tracking Port data from a specified path.

        Parameters
        ----------
        path2data : Path
            The path to the directory containing the data file.

        Returns
        -------
        np.ndarray
            A 2D numpy array with the position data.
        """
        print("Loading Tracker data...")
        return np.load(path2data / Path("data_array.npy"))

    def load_times(self, path2data: Path) -> np.ndarray:
        """
        Load timestamps from a specified path.

        Parameters
        ----------
        path2data : Path
            The path to the directory containing the timestamps file.

        Returns
        -------
        np.ndarray
            A numpy array containing the timestamps.
        """
        return np.load(path2data / Path("timestamps.npy"))


@dataclass
class TrackMe(OEPlugin):
    """
    Documents the TrackMe plugin.

    Methods
    -------
    load(path2data: Path) -> np.ndarray
        Load TrackMe data from a specified path.
    load_times(path2data: Path) -> np.ndarray
        Load timestamps from a specified path.
    load_frame_count(path2data: Path) -> np.ndarray
        Load frame count data from a specified path.
    load_ttl_times(path2data: Path) -> np.ndarray
        Load TTL times from a specified path.
    """

    def load(self, path2data: Path) -> np.ndarray:
        """
        Load TrackMe data from a specified path.

        Parameters
        ----------
        path2data : Path
            The path to the directory containing the data file.

        Returns
        -------
        np.ndarray
            A 2D numpy array with the TrackMe data.
        """
        print("Loading TrackMe data...")
        mmap = memmapBinaryFile(path2data / Path("continuous.dat"), self.channel_count)
        return np.array(mmap[0:2, :]).T

    def load_times(self, path2data: Path) -> np.ndarray:
        """
        Load timestamps from a specified path.

        Parameters
        ----------
        path2data : Path
            The path to the directory containing the timestamps file.

        Returns
        -------
        np.ndarray
            A numpy array containing the timestamps.
        """
        ts = np.load(path2data / Path("timestamps.npy"))
        return ts - ts[0]

    def load_frame_count(self, path2data: Path) -> np.ndarray:
        """
        Load frame count data from a specified path.

        Parameters
        ----------
        path2data : Path
            The path to the directory containing the data file.

        Returns
        -------
        np.ndarray
            A numpy array containing the frame count data.
        """
        data = memmapBinaryFile(path2data / Path("data_array.npy"), self.channel_count)
        # framecount data is always last column in continuous.dat file
        return np.array(data[-1, :]).T

    def load_ttl_times(self, path2data: Path) -> np.ndarray:
        """
        Load TTL times from a specified path.

        Parameters
        ----------
        path2data : Path
            The path to the directory containing the timestamps and states files.

        Returns
        -------
        np.ndarray
            A numpy array containing the TTL times.
        """
        ts = np.load(path2data / Path("timestamps.npy"))
        states = np.load(path2data / Path("states.npy"))
        return ts[states > 0]


@dataclass
class StimControl(OEPlugin):
    """
    Documents the StimControl plugin.

    Attributes
    ----------
    Device : IntConversion
        Device setting for the StimControl, default is 0.
    Duration : IntConversion
        Duration setting for the StimControl, default is 0.
    Interval : IntConversion
        Interval setting for the StimControl, default is 0.
    Gate : IntConversion
        Gate setting for the StimControl, default is 0.
    Output : IntConversion
        Output setting for the StimControl, default is 0.
    Start : IntConversion
        Start setting for the StimControl, default is 0.
    Stop : IntConversion
        Stop setting for the StimControl, default is 0.
    Trigger : IntConversion
        Trigger setting for the StimControl, default is 0.
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
    Documents the ELECTRODE entries in the settings.xml file.

    Attributes
    ----------
    nChannels : IntConversion
        Number of channels for the electrode, default is 0.
    id : IntConversion
        ID of the electrode, default is 0.
    subChannels : list[int]
        List of sub-channel indices, default is an empty list.
    subChannelsThresh : list[int]
        List of sub-channel thresholds, default is an empty list.
    subChannelsActive : list[int]
        List of active sub-channels, default is an empty list.
    prePeakSamples : IntConversion
        Number of samples before the peak, default is 8.
    postPeakSamples : IntConversion
        Number of samples after the peak, default is 32.
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
    Documents the Ripple Detector plugin.

    Attributes
    ----------
    Ripple_Input : IntConversion
        Input setting for the Ripple Detector, default is -1.
    Ripple_Out : IntConversion
        Output setting for the Ripple Detector, default is -1.
    Ripple_save : IntConversion
        Save setting for the Ripple Detector, default is -1.
    ripple_std : FloatConversion
        Standard deviation setting for the Ripple Detector, default is -1.
    time_thresh : FloatConversion
        Time threshold setting for the Ripple Detector, default is -1.
    refr_time : FloatConversion
        Refractory time setting for the Ripple Detector, default is -1.
    rms_samples : FloatConversion
        RMS samples setting for the Ripple Detector, default is -1.
    ttl_duration : FloatConversion
        TTL duration setting for the Ripple Detector, default is -1.
    ttl_percent : FloatConversion
        TTL percent setting for the Ripple Detector, default is -1.
    mov_detect : IntConversion
        Movement detection setting for the Ripple Detector, default is -1.
    mov_input : IntConversion
        Movement input setting for the Ripple Detector, default is -1.
    mov_out : IntConversion
        Movement output setting for the Ripple Detector, default is -1.
    mov_std : FloatConversion
        Movement standard deviation setting for the Ripple Detector, default is -1.
    min_time_st : FloatConversion
        Minimum time setting for the Ripple Detector, default is -1.
    min_time_mov : FloatConversion
        Minimum movement time setting for the Ripple Detector, default is -1.

    Methods
    -------
    load_ttl(path2TTL: Path, trial_start_time: float) -> dict
        Load TTL data from a specified path and trial start time.
    """

    Ripple_Input: IntConversion = IntConversion(default=-1)
    Ripple_Out: IntConversion = IntConversion(default=-1)
    Ripple_save: IntConversion = IntConversion(default=-1)
    ripple_std: FloatConversion = FloatConversion(default=-1)
    time_thresh: FloatConversion = FloatConversion(default=-1)
    refr_time: FloatConversion = FloatConversion(default=-1)
    rms_samples: FloatConversion = FloatConversion(default=-1)
    ttl_duration: FloatConversion = FloatConversion(default=-1)
    ttl_percent: FloatConversion = FloatConversion(default=-1)
    mov_detect: IntConversion = IntConversion(default=-1)
    mov_input: IntConversion = IntConversion(default=-1)
    mov_out: IntConversion = IntConversion(default=-1)
    mov_std: FloatConversion = FloatConversion(default=-1)
    min_time_st: FloatConversion = FloatConversion(default=-1)
    min_time_mov: FloatConversion = FloatConversion(default=-1)

    def load_ttl(self, path2TTL: Path, trial_start_time: float) -> dict:
        """
        Load TTL data from a specified path and trial start time.

        Parameters
        ----------
        path2TTL : Path
            The path to the directory containing the TTL data files.
        trial_start_time : float
            The start time of the trial.

        Returns
        -------
        dict
            A dictionary containing the TTL timestamps and other related data.
        """
        timestamps = np.load(path2TTL / Path("timestamps.npy")) - trial_start_time
        states = np.load(path2TTL / Path("states.npy"))
        out = dict()
        out_ttl = self.Ripple_Out
        save_ttl = self.Ripple_save
        # check for identical times - should fix the plugin really...
        # for some reason the plugin occasionally spits out negative TTL states
        # for the ripple out line without a corresponding positive state. This
        # picks out those bad indices and removes them from the states and
        # timestamps vectors
        indices = np.ravel(np.argwhere(states == (0 - out_ttl)))
        bad_indices = np.ravel(
            indices[np.argwhere(states[indices] + states[indices - 1])]
        )
        states = np.delete(states, bad_indices)
        timestamps = np.delete(timestamps, bad_indices)

        out["ttl_timestamps"] = timestamps[states == out_ttl]
        out["ttl_timestamps_off"] = timestamps[states == out_ttl * -1]
        all_ons = timestamps[states == save_ttl]
        out["no_laser_ttls"] = np.lib.setdiff1d(all_ons, out["ttl_timestamps"])
        if len(out["ttl_timestamps_off"]) == len(out["ttl_timestamps"]):
            out["stim_duration"] = np.nanmean(
                out["ttl_timestamps_off"] - out["ttl_timestamps"]
            )
        else:
            out["stim_duration"] = None
        return out


class AbstractProcessorFactory:
    """
    Factory class for creating various processor objects.

    Methods
    -------
    create_pos_tracker() -> PosTracker
        Create a PosTracker object.
    create_rhythm_fpga() -> RhythmFPGA
        Create a RhythmFPGA object.
    create_neuropix_pxi() -> NeuropixPXI
        Create a NeuropixPXI object.
    create_acquisition_board() -> AcquisitionBoard
        Create an AcquisitionBoard object.
    create_spike_sorter() -> SpikeSorter
        Create a SpikeSorter object.
    create_track_me() -> TrackMe
        Create a TrackMe object.
    create_record_node() -> RecordNode
        Create a RecordNode object.
    create_stim_control() -> StimControl
        Create a StimControl object.
    create_oe_plugin() -> OEPlugin
        Create an OEPlugin object.
    create_ripple_detector() -> RippleDetector
        Create a RippleDetector object.
    create_bandpass_filter() -> BandpassFilter
        Create a BandpassFilter object.
    """

    def create_pos_tracker(self):
        """
        Create a PosTracker object.

        Returns
        -------
        PosTracker
            A new PosTracker object.
        """
        return PosTracker()

    def create_rhythm_fpga(self):
        """
        Create a RhythmFPGA object.

        Returns
        -------
        RhythmFPGA
            A new RhythmFPGA object.
        """
        return RhythmFPGA()

    def create_neuropix_pxi(self):
        """
        Create a NeuropixPXI object.

        Returns
        -------
        NeuropixPXI
            A new NeuropixPXI object.
        """
        return NeuropixPXI()

    def create_acquisition_board(self):
        """
        Create an AcquisitionBoard object.

        Returns
        -------
        AcquisitionBoard
            A new AcquisitionBoard object.
        """
        return AcquisitionBoard()

    def create_spike_sorter(self):
        """
        Create a SpikeSorter object.

        Returns
        -------
        SpikeSorter
            A new SpikeSorter object.
        """
        return SpikeSorter()

    def create_track_me(self):
        """
        Create a TrackMe object.

        Returns
        -------
        TrackMe
            A new TrackMe object.
        """
        return TrackMe()

    def create_record_node(self):
        """
        Create a RecordNode object.

        Returns
        -------
        RecordNode
            A new RecordNode object.
        """
        return RecordNode()

    def create_stim_control(self):
        """
        Create a StimControl object.

        Returns
        -------
        StimControl
            A new StimControl object.
        """
        return StimControl()

    def create_oe_plugin(self):
        """
        Create an OEPlugin object.

        Returns
        -------
        OEPlugin
            A new OEPlugin object.
        """
        return OEPlugin()

    def create_ripple_detector(self):
        """
        Create a RippleDetector object.

        Returns
        -------
        RippleDetector
            A new RippleDetector object.
        """
        return RippleDetector()

    def create_bandpass_filter(self):
        """
        Create a BandpassFilter object.

        Returns
        -------
        BandpassFilter
            A new BandpassFilter object.
        """
        return BandpassFilter()


class ProcessorFactory:
    """
    Factory class for creating various processor objects based on the processor name.

    Attributes
    ----------
    factory : AbstractProcessorFactory
        An instance of AbstractProcessorFactory used to create processor objects.

    Methods
    -------
    create_processor(proc_name: str)
        Create a processor object based on the processor name.
    """

    factory = AbstractProcessorFactory()

    def create_processor(self, proc_name: str):
        """
        Create a processor object based on the processor name.

        Parameters
        ----------
        proc_name : str
            The name of the processor to create.

        Returns
        -------
        object
            The created processor object.
        """
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
    Recursive function that applies a function to each node.

    Parameters
    ----------
    node : ET.Element
        The current XML element node.
    func : Callable
        The function to apply to each node.
    cls : dataclass
        The dataclass instance to pass to the function.
    """
    if node is not None:
        func(node, cls)
        for item in node:
            recurseNode(item, func, cls)


def addValues2Class(node: ET.Element, cls: dataclass):
    """
    Add values from an XML node to a dataclass instance.

    Parameters
    ----------
    node : ET.Element
        The XML element node containing the values.
    cls : dataclass
        The dataclass instance to which the values will be added.
    """
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
    Loads up the structure.oebin file for Open Ephys flat binary format recordings.

    Parameters
    ----------
    fname : Path
        The path to the directory containing the structure.oebin file.

    Attributes
    ----------
    filename : list
        List of filenames found.
    data : dict
        Dictionary containing the data read from the structure.oebin files.

    Methods
    -------
    find_oebin(pname: Path) -> list
        Find all structure.oebin files in the specified path.
    read_oebin(fname: Path) -> dict
        Read the structure.oebin file and return its contents as a dictionary.
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
        """
        Find all structure.oebin files in the specified path.

        Parameters
        ----------
        pname : Path
            The path to search for structure.oebin files.

        Returns
        -------
        list
            A list of paths to the found structure.oebin files.
        """
        path2oebins = []
        for d, c, f in os.walk(pname):
            for ff in f:
                if "." not in c:  # ignore hidden directories
                    if ff == "structure.oebin":
                        path2oebins.append(Path(d))
        return path2oebins

    def read_oebin(self, fname: Path) -> dict:
        """
        Read the structure.oebin file and return its contents as a dictionary.

        Parameters
        ----------
        fname : Path
            The path to the structure.oebin file.

        Returns
        -------
        dict
            The contents of the structure.oebin file.
        """
        self.filename.append(fname)
        fname = fname / Path("structure.oebin")
        with open(fname, "r") as f:
            data = json.load(f)
        return data


class Settings(object):
    """
    Groups together the other classes in this module and does the actual
    parsing of the settings.xml file.

    Parameters
    ----------
    pname : str or Path
        The pathname to the top-level directory, typically in form
        YYYY-MM-DD_HH-MM-SS.

    Attributes
    ----------
    filename : str or None
        The path to the settings.xml file.
    tree : ElementTree or None
        The parsed XML tree of the settings.xml file.
    processors : OrderedDict
        Dictionary of processor objects.
    record_nodes : OrderedDict
        Dictionary of record node objects.

    Methods
    -------
    load()
        Creates a handle to the basic XML document.
    parse()
        Parses the basic information about the processors in the
        open-ephys signal chain as described in the settings.xml file(s).
    get_processor(key: str)
        Returns the information about the requested processor or an
        empty OEPlugin instance if it's not available.
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
        Creates a handle to the basic XML document.
        """
        if self.filename is not None:
            self.tree = ET.parse(self.filename).getroot()

    def parse(self):
        """
        Parses the basic information about the processors in the
        open-ephys signal chain as described in the settings.xml file(s).
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
        empty OEPlugin instance if it's not available.

        Parameters
        ----------
        key : str
            The key of the processor to retrieve.

        Returns
        -------
        object
            The requested processor object or an empty OEPlugin instance.
        """
        processor = [self.processors[k] for k in self.processors.keys() if key in k]
        if processor:
            if len(processor) == 1:
                return processor[0]
            else:
                return processor
        return OEPlugin()
