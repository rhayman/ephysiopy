import os
import warnings
from enum import Enum
from pathlib import Path, PurePath

import matplotlib.pylab as plt
import numpy as np
from ephysiopy.common.ephys_generic import PosCalcsGeneric
from ephysiopy.openephys2py.OESettings import Settings
from ephysiopy.visualise.plotting import FigureMaker


def fileExists(pname, fname) -> bool:
    return os.path.exists(os.path.join(pname, fname))


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


class RecordingKind(Enum):
    FPGA = 1
    NEUROPIXELS = 2


class KiloSortSession(object):
    """
    Loads and processes data from a Kilosort session.

    A kilosort session results in a load of .npy files, a .csv or .tsv file.
    The .npy files contain things like spike times, cluster indices and so on.
    Importantly	the .csv (or .tsv) file contains the cluster identities of
    the SAVED part of the phy template-gui (ie when you click "Save" from the
    Clustering menu): this file consists of a header ('cluster_id' and 'group')
    where 'cluster_id' is obvious (relates to identity in spk_clusters.npy),
    the 'group' is a string that contains things like 'noise' or 'unsorted' or
    whatever as the phy user can define their own labels.

    Parameters
    ----------
    fname_root : str
        The top-level directory. If the Kilosort session was run directly on
        data from an openephys recording session then fname_root is typically
        in form of YYYY-MM-DD_HH-MM-SS
    """

    def __init__(self, fname_root):
        """
        Walk through the path to find the location of the files in case this
        has been called in another way i.e. binary format a la Neuropixels
        """
        self.fname_root = fname_root
        import os

        for d, c, f in os.walk(fname_root):
            for ff in f:
                if "." not in c:  # ignore hidden directories
                    if "spike_times.npy" in ff:
                        self.fname_root = d
        self.cluster_id = None
        self.spk_clusters = None
        self.spk_times = None
        self.good_clusters = []

    def load(self):
        """
        Load all the relevant files

        There is a distinction between clusters assigned during the automatic
        spike sorting process (here KiloSort2) and the manually curated
        distillation of the automatic process conducted by the user with
        a program such as phy.

        * The file cluster_KSLabel.tsv is output from KiloSort.
            All this information is also contained in the cluster_info.tsv
            file! Not sure about the .csv version (from original KiloSort?)
        * The files cluster_group.tsv or cluster_groups.csv contain
            "group labels" from phy ('good', 'MUA', 'noise' etc).
            One of these (cluster_groups.csv or cluster_group.tsv)
            is from kilosort and the other from kilosort2
        """
        import os

        import pandas as pd

        dtype = {"names": ("cluster_id", "group"), "formats": ("i4", "S10")}
        # One of these (cluster_groups.csv or cluster_group.tsv) is from
        # kilosort and the other from kilosort2
        # and is updated by the user when doing cluster assignment in phy
        # See comments above this class definition for a bit more info
        if fileExists(self.fname_root, "cluster_groups.csv"):
            self.cluster_id, self.group = np.loadtxt(
                os.path.join(self.fname_root, "cluster_groups.csv"),
                unpack=True,
                skiprows=1,
                dtype=dtype,
            )
        if fileExists(self.fname_root, "cluster_group.tsv"):
            self.cluster_id, self.group = np.loadtxt(
                os.path.join(self.fname_root, "cluster_group.tsv"),
                unpack=True,
                skiprows=1,
                dtype=dtype,
            )

        """
        Output some information to the user if self.cluster_id is still None
        it implies that data has not been sorted / curated
        """
        # if self.cluster_id is None:
        #     print(f"Searching {os.path.join(self.fname_root)} and...")
        #     warnings.warn("No cluster_groups.tsv or cluster_group.csv file
        # was found.\
        #         Have you manually curated the data (e.g with phy?")

        # HWPD 20200527
        # load cluster_info file and add X co-ordinate to it
        if fileExists(self.fname_root, "cluster_info.tsv"):
            self.cluster_info = pd.read_csv(
                os.path.join(self.fname_root, "cluster_info.tsv"), "\t"
            )
            if fileExists(self.fname_root, "channel_positions.npy") and fileExists(
                self.fname_root, "channel_map.npy"
            ):
                chXZ = np.load(os.path.join(self.fname_root, "channel_positions.npy"))
                chMap = np.load(os.path.join(self.fname_root, "channel_map.npy"))
                chID = np.asarray(
                    [np.argmax(chMap == x) for x in self.cluster_info.ch.values]
                )
                self.cluster_info["chanX"] = chXZ[chID, 0]
                self.cluster_info["chanY"] = chXZ[chID, 1]

        dtype = {"names": ("cluster_id", "KSLabel"), "formats": ("i4", "S10")}
        # 'Raw' labels from a kilosort session
        if fileExists(self.fname_root, "cluster_KSLabel.tsv"):
            self.ks_cluster_id, self.ks_group = np.loadtxt(
                os.path.join(self.fname_root, "cluster_KSLabel.tsv"),
                unpack=True,
                skiprows=1,
                dtype=dtype,
            )
        if fileExists(self.fname_root, "spike_clusters.npy"):
            self.spk_clusters = np.squeeze(
                np.load(os.path.join(self.fname_root, "spike_clusters.npy"))
            )
        if fileExists(self.fname_root, "spike_times.npy"):
            self.spk_times = np.squeeze(
                np.load(os.path.join(self.fname_root, "spike_times.npy"))
            )
            return True
        warnings.warn(
            "No spike times or clusters were found \
            (spike_times.npy or spike_clusters.npy).\
                You should run KiloSort"
        )
        return False

    def removeNoiseClusters(self):
        """
        Removes clusters with labels 'noise' and 'mua' in self.group
        """
        if self.cluster_id is not None:
            self.good_clusters = []
            for id_group in zip(self.cluster_id, self.group):
                if (
                    "noise" not in id_group[1].decode()
                    and "mua" not in id_group[1].decode()
                ):
                    self.good_clusters.append(id_group[0])

    def removeKSNoiseClusters(self):
        """
        Removes "noise" and "mua" clusters from the kilosort labelled stuff
        """
        for cluster_id, kslabel in zip(self.ks_cluster_id, self.ks_group):
            if "good" in kslabel.decode():
                self.good_clusters.append(cluster_id)


class OpenEphysBase(FigureMaker):
    """
    Base class for openephys anaylsis with data recorded in either
    the NWB or binary format

    Parameters
    ----------
    pname_root : str
        The top-level directory, typically in form of YYYY-MM-DD_HH-MM-SS

    Notes
    ----
    This isn't really an Abstract Base Class (as with c++) as Python doesn't
    really have this concept but it forms the backbone for two other classes
    (OpenEphysNPX & OpenEphysNWB)
    """

    def __init__(self, pname_root: str, **kwargs):
        super().__init__()
        # top-level directory, typically of form YYYY-MM-DD_HH-MM-SS
        assert os.path.exists(pname_root)
        self.pname_root = pname_root
        self.settings = None
        self.kilodata = None
        self.rawData = None
        self.xy = None
        self.xyTS = None
        self.sync_message_file = None
        self.ap_sample_rate = 30000
        self.recording_start_time = 0
        self.ts = None
        self.ttl_data = None
        self.ttl_timestamps = None
        # a list of np.arrays, nominally containing tetrode data in
        # format nspikes x 4 x 40
        self.spikeData = None
        self.accelerometerData = None
        # This will become an instance of OESettings.Settings
        self.settings = None
        if "jumpmax" in kwargs:
            self.jumpmax = kwargs["jumpmax"]
        else:
            self.jumpmax = 100
        self._ppm = getattr(self, "ppm", 300)

    @property
    def ppm(self):
        return self._ppm

    @ppm.setter
    def ppm(self, value):
        if hasattr(self, "PosCalcs"):
            self._ppm = value
            P = getattr(self, "PosCalcs")
            P.ppm = value
            sample_rate = getattr(P, "sample_rate")
            tracker_params = P.tracker_params
            tracker_params["SampleRate"] = sample_rate
            P.postprocesspos(tracker_params)
            setattr(self, "xy", P.xy)
            setattr(self, "dir", P.dir)
            setattr(self, "speed", P.speed)

    def find_files(
        self,
        pname_root: str,
        experiment_name: str = "experiment1",
        recording_name: str = "recording1",
        recording_kind: RecordingKind = RecordingKind.NEUROPIXELS,
    ):
        exp_name = Path(experiment_name)
        PosTracker_match = (
            exp_name / recording_name / "events" / "*Pos_Tracker*/BINARY_group*"
        )
        TrackingPlugin_match = (
            exp_name / recording_name / "events" / "*Tracking_Port*/BINARY_group*"
        )
        sync_file_match = exp_name / recording_name
        rec_kind = ""
        if recording_kind == RecordingKind.NEUROPIXELS:
            rec_kind = "Neuropix-PXI-[0-9][0-9][0-9]."
        elif recording_kind == RecordingKind.FPGA:
            rec_kind = "Rhythm_FPGA-[0-9][0-9][0-9]."
        APdata_match = exp_name / recording_name / "continuous" / (rec_kind + "0")
        LFPdata_match = exp_name / recording_name / "continuous" / (rec_kind + "1")

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
                    if "sync_messages.txt" in ff:
                        if PurePath(d).match(str(sync_file_match)):
                            sync_file = os.path.join(d, "sync_messages.txt")
                            if fileContainsString(sync_file, "Processor"):
                                self.sync_message_file = sync_file
                                print(f"Found sync_messages file at: {sync_file}")

    def load(self, *args, **kwargs):
        # Overridden by sub-classes
        pass

    def loadPos(self, *args, **kwargs):
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
                    recording_start_time = float(tmp[0])
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
            pos_timebase = getattr(self, "pos_timebase", 3e4)
            sample_rate = np.floor(1 / np.mean(np.diff(pos_ts) / pos_timebase))
            self.xyTS = pos_ts - recording_start_time
            pos_timebase = getattr(self, "pos_timebase", 3e4)
            self.xyTS = self.xyTS / pos_timebase  # convert to seconds
            if self.sync_message_file is not None:
                recording_start_time = self.xyTS[0]
            self.pos_sample_rate = sample_rate
            self.orig_x = pos_data[:, 0]
            self.orig_y = pos_data[:, 1]

            P = PosCalcsGeneric(
                pos_data[:, 0],
                pos_data[:, 1],
                cm=True,
                ppm=self.ppm,
                jumpmax=self.jumpmax,
            )
            P.postprocesspos({"SampleRate": sample_rate})
            setattr(self, "PosCalcs", P)
            self.xy = P.xy
            self.dir = P.dir
            self.speed = P.speed
        else:
            warnings.warn(
                "Could not find the pos data. \
                Make sure there is a pos_data folder with data_array.npy \
                and timestamps.npy in"
            )
        self.recording_start_time = recording_start_time

    def loadKilo(self, **kwargs):
        import os

        if "pname" in kwargs:
            pname = kwargs["pname"]
        else:
            pname = self.pname_root
        # Loads a kilosort session
        kilodata = None
        if pname is not None:
            if os.path.exists(pname):
                # pname_root gets walked through and over-written with
                # correct location of kilosort data
                kilodata = KiloSortSession(pname)
        if kilodata is not None:
            if kilodata.load():
                try:
                    kilodata.removeKSNoiseClusters()
                except Exception:
                    pass
        self.kilodata = kilodata

    def __loadSettings__(self):
        # Loads the settings.xml data
        if self.settings is None:
            # pname_root gets walked through and over-written with
            # correct location of settings.xml
            settings = Settings(self.pname_root)
            settings.parse()
            self.settings = settings

    def __loaddata__(self, **kwargs):
        self.load(self.pname_root, **kwargs)  # some knarly hack

    def __calcTrialLengthFromBinarySize__(
        self, path2file: str, n_channels=384, sample_rate=30000
    ):
        """
        Returns the time taken to run the trial (in seconds) based on the size
        of the binary file on disk
        """
        import os

        status = os.stat(path2file)
        return status.st_size / (2.0 * n_channels * sample_rate)

    def memmapBinaryFile(self, path2file: str, n_channels=384, **kwargs):
        """
        Returns a numpy memmap of the 30Khz sampled high frequency data in the
        file 'continuous.dat', if present
        """
        import os

        if os.path.exists(path2file):
            status = os.stat(path2file)
            n_samples = status.st_size / (2.0 * n_channels)
            mmap = np.memmap(
                path2file, np.int16, "r", 0, (n_channels, n_samples), order="F"
            )
            return mmap

    def exportPos(self):
        xy = self.plotPos(show=False)
        out = np.hstack([xy, self.xyTS[:, np.newaxis]])
        np.savetxt("position.txt", out, delimiter=",", fmt=["%3.3i", "%3.3i", "%3.3f"])

    def save_ttl(self, out_fname):
        """
        Saves the ttl data to text file out_fname
        """
        if (len(self.ttl_data) > 0) and (len(self.ttl_timestamps) > 0):
            data = np.array([self.ttl_data, self.ttl_timestamps])
            if data.shape[0] == 2:
                data = data.T
            np.savetxt(out_fname, data, delimiter="\t")

    def filterPosition(self, filter_dict: dict):
        if hasattr(self, "PosCalcs"):
            P = getattr(self, "PosCalcs")
            P.filterPos(filter_dict)
            setattr(self, "xy", P.xy)
            setattr(self, "dir", P.dir)
            setattr(self, "speed", P.speed)

    def getClusterSpikeTimes(self, cluster: int):
        """
        Returns the spike times in seconds of the given cluster
        """
        times = self.kilodata.spk_times.T
        return (
            times[self.kilodata.spk_clusters == cluster].astype(np.int64)
            / self.ap_sample_rate
        )

    def plotSummary(self, cluster: int, **kwargs):
        ts = self.getClusterSpikeTimes(cluster)
        fig = self.makeSummaryPlot(ts, **kwargs)
        plt.show()
        return fig

    def plotSpikesOnPath(self, cluster: int = None, **kwargs):
        ts = None
        if cluster is not None:
            ts = self.getClusterSpikeTimes(cluster)  # in samples
        ax = self.makeSpikePathPlot(ts, **kwargs)
        plt.show()
        return ax

    def plotRateMap(self, cluster: int, **kwargs):
        ts = self.getClusterSpikeTimes(cluster)  # in samples
        ax = self.makeRateMap(ts)
        plt.show()
        return ax

    def plotHDMap(self, cluster: int, **kwargs):
        ts = self.getClusterSpikeTimes(cluster)  # in samples
        ax = self.makeHDPlot(ts, **kwargs)
        plt.show()
        return ax

    def plotSAC(self, cluster: int, **kwargs):
        ts = self.getClusterSpikeTimes(cluster)  # in samples
        ax = self.makeSAC(ts, **kwargs)
        plt.show()
        return ax

    def plotSpeedVsRate(self, cluster: int, **kwargs):
        ts = self.getClusterSpikeTimes(cluster)  # in samples
        ax = self.makeSpeedVsRatePlot(ts, **kwargs)
        plt.show()
        return ax

    def plotSpeedVsHeadDirection(self, cluster: int, **kwargs):
        ts = self.getClusterSpikeTimes(cluster)  # in samples
        ax = self.makeSpeedVsHeadDirectionPlot(ts, **kwargs)
        plt.show()
        return ax

    def plotEEGPower(self, channel=0):
        """
        Plots LFP power

        Parameters
        ----------
        channel : int
            The channel from which to plot the power

        See Also
        -----
        ephysiopy.common.ephys_generic.EEGCalcsGeneric.plotPowerSpectrum()
        """
        from ephysiopy.common.ephys_generic import EEGCalcsGeneric

        if self.rawData is None:
            print("Loading raw data...")
            self.load(loadraw=True)
        from scipy import signal

        n_samples = np.shape(self.rawData[:, channel])[0]
        s = signal.resample(self.rawData[:, channel], int(n_samples / 3e4) * 500)
        E = EEGCalcsGeneric(s, 500)
        power_res = E.calcEEGPowerSpectrum()
        ax = self.makePowerSpectrum(
            power_res[0],
            power_res[1],
            power_res[2],
            power_res[3],
            power_res[4],
        )
        plt.show()
        return ax

    def plotXCorr(self, cluster: int, **kwargs):
        ts = self.getClusterSpikeTimes(cluster)
        ax = self.makeXCorr(ts)
        plt.show()
        return ax

    def plotPSTH(self, **kwargs):
        """Plots the peri-stimulus time histogram for all the 'good' clusters

        Given some data has been recorded in the ttl channel, this method plots
        the PSTH for each 'good' cluster and just keeps spitting out figure
        windows
        """
        self.__loadSettings__()
        self.settings.parseStimControl()
        if self.kilodata is None:
            self.loadKilo(**kwargs)
        from ephysiopy.common.spikecalcs import SpikeCalcsGeneric

        # in seconds
        spk_times = (self.kilodata.spk_times.T[0] / 3e4) + self.ts[0]
        S = SpikeCalcsGeneric(spk_times)
        # this is because some of the trials have two weird events
        # logged at about 2-3 minutes in...
        S.event_ts = self.ttl_timestamps[2::2]
        S.spk_clusters = self.kilodata.spk_clusters
        S.stim_width = 0.01  # in seconds
        for x in self.kilodata.good_clusters:
            print(next(S.plotPSTH(x)))

    def plotEventEEG(self):
        from ephysiopy.common.ephys_generic import EEGCalcsGeneric

        if self.rawData is None:
            print("Loading raw data...")
            self.load(loadraw=True)
        E = EEGCalcsGeneric(self.rawData[:, 0], 3e4)
        # this is because some of the trials have two weird events
        # logged at about 2-3 minutes in...
        event_ts = self.ttl_timestamps[2::2]
        E.plotEventEEG(event_ts)


class OpenEphysNPX(OpenEphysBase):
    """
    The main class for dealing with data recorded using Neuropixels probes
    under openephys.
    """

    def __init__(self, pname_root: str):
        super().__init__(pname_root)
        self.path2PosData = None
        self.path2APdata = None
        self.path2LFPdata = None
        self.path2syncmessages = None
        self.path2APOEBin = None
        self.path2PosOEBin = None

    def load(
        self,
        experiment_name="experiment1",
        recording_name="recording1",
        **kwargs,
    ):
        """
        Loads data recorded in the OE 'flat' binary format.

        Parameters
        ----------
        pname_root : str
            The top level directory, typically in form of YYYY-MM-DD_HH-MM-SS

        recording_name : str
            The directory immediately beneath pname_root

        See Also
        --------
        See open-ephys wiki pages
        """
        self.isBinary = True
        import os

        self.sync_message_file = None
        self.recording_start_time = None
        ap_sample_rate = getattr(self, "ap_sample_rate", 30000)

        super().find_files(
            self.pname_root, experiment_name, recording_name, RecordingKind.NEUROPIXELS
        )
        super().loadPos()

        n_channels = getattr(self, "n_channels", 384)
        trial_length = 0  # make sure a trial_length has a value
        if self.path2APdata is not None:
            if fileExists(self.path2APdata, "continuous.dat"):
                trial_length = self.__calcTrialLengthFromBinarySize__(
                    os.path.join(self.path2APdata, "continuous.dat"),
                    n_channels,
                    ap_sample_rate,
                )

        # this way of creating timestamps will be fine for single probes
        # but will need to be modified if using multiple probes and/ or
        # different timestamp syncing method
        # OE's 'synchronised_timestamps.npy' should now take care of this
        self.ts = np.arange(
            self.recording_start_time,
            trial_length + self.recording_start_time,
            1.0 / ap_sample_rate,
        )

    def plotSpectrogramByDepth(
        self, nchannels=384, nseconds=100, maxFreq=125, **kwargs
    ):
        """
        Plots a heat map spectrogram of the LFP for each channel.

        Line plots of power per frequency band and power on a subset
        of channels are also displayed to the right and above the main plot.

        Parameters
        ----------
        nchannels : int
            The number of channels on the probe
        nseconds : int, optional
            How long in seconds from the start of the trial to do the
            spectrogram for (for speed).
            Default 100
        maxFreq : int
            The maximum frequency in Hz to plot the spectrogram out to.
            Maximum 1250.
            Default 125
        kwargs: legal values are: 'frequencies' and 'channels'; both are lists
                that denote which frequencies to show the mean power of
                along the length of the probe and,
                which channels to show the frequency spectra of.

        Notes
        -----
        Should also allow kwargs to specify exactly which channels and / or
        frequency bands to do the line plots for
        """
        import os

        lfp_file = os.path.join(self.path2LFPdata, "continuous.dat")
        status = os.stat(lfp_file)
        nsamples = int(status.st_size / 2 / nchannels)
        mmap = np.memmap(lfp_file, np.int16, "r", 0, (nchannels, nsamples), order="F")
        # Load the channel map
        #  Assumes this is in the AP data location and that kilosort was run
        # channel_map = np.squeeze(np.load(os.path.join(
        #     self.path2APdata, 'channel_map.npy')))
        channel_map = np.arange(nchannels)
        lfp_sample_rate = 2500
        data = np.array(mmap[channel_map, 0 : nseconds * lfp_sample_rate])
        from ephysiopy.common.ephys_generic import EEGCalcsGeneric

        E = EEGCalcsGeneric(data[0, :], lfp_sample_rate)
        E.calcEEGPowerSpectrum()
        # Select a subset of the full amount of data to display later
        spec_data = np.zeros(shape=(data.shape[0], len(E.sm_power[0::50])))
        for chan in range(data.shape[0]):
            E = EEGCalcsGeneric(data[chan, :], lfp_sample_rate)
            E.calcEEGPowerSpectrum()
            spec_data[chan, :] = E.sm_power[0::50]

        x, y = np.meshgrid(E.freqs[0::50], channel_map)
        import matplotlib.colors as colors
        from matplotlib.pyplot import cm
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        _, spectoAx = plt.subplots()
        if "cmap" in kwargs:
            cmap = kwargs["cmap"]
        else:
            cmap = "bone"
        spectoAx.pcolormesh(
            x,
            y,
            spec_data,
            edgecolors="face",
            cmap=cmap,
            norm=colors.LogNorm(),
            shading="nearest",
        )
        if "minFreq" in kwargs:
            minFreq = kwargs["minFreq"]
        else:
            minFreq = 0
        spectoAx.set_xlim(minFreq, maxFreq)
        spectoAx.set_ylim(channel_map[0], channel_map[-1])
        spectoAx.set_xlabel("Frequency (Hz)")
        spectoAx.set_ylabel("Channel")
        divider = make_axes_locatable(spectoAx)
        channel_spectoAx = divider.append_axes("top", 1.2, pad=0.1, sharex=spectoAx)
        meanfreq_powerAx = divider.append_axes("right", 1.2, pad=0.1, sharey=spectoAx)
        plt.setp(
            channel_spectoAx.get_xticklabels() + meanfreq_powerAx.get_yticklabels(),
            visible=False,
        )

        mn_power = np.mean(spec_data, 0)
        if "channels" in kwargs:
            channels = kwargs["channels"]
            cols = iter(cm.rainbow(np.linspace(0, 1, (len(kwargs["channels"])))))
        else:
            channels = np.arange(0, spec_data.shape[0], 60)
            cols = iter(cm.rainbow(np.linspace(0, 1, (nchannels // 60) + 1)))
        for i in channels:
            c = next(cols)
            channel_spectoAx.plot(
                E.freqs[0::50],
                10 * np.log10(spec_data[i, :] / mn_power),
                c=c,
                label=str(i),
            )

        channel_spectoAx.set_ylabel("Channel power(dB)")
        channel_spectoAx.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            mode="expand",
            fontsize="x-small",
            ncol=4,
        )

        if "frequencies" in kwargs:
            lower_freqs = kwargs["frequencies"]
            upper_freqs = lower_freqs[1::]
            inc = np.diff([lower_freqs[-2], lower_freqs[-1]])
            upper_freqs = np.append(upper_freqs, lower_freqs[-1] + inc)
        else:
            freq_inc = 6
            lower_freqs = np.arange(1, maxFreq - freq_inc, freq_inc)
            upper_freqs = np.arange(1 + freq_inc, maxFreq, freq_inc)

        cols = iter(cm.nipy_spectral(np.linspace(0, 1, len(upper_freqs))))
        mn_power = np.mean(spec_data, 1)
        print(f"spec_data shape = {np.shape(spec_data)}")
        print(f"mn_power shape = {np.shape(mn_power)}")
        for freqs in zip(lower_freqs, upper_freqs):
            freq_mask = np.logical_and(
                E.freqs[0::50] > freqs[0], E.freqs[0::50] < freqs[1]
            )
            mean_power = 10 * np.log10(np.mean(spec_data[:, freq_mask], 1) / mn_power)
            c = next(cols)
            meanfreq_powerAx.plot(
                mean_power,
                channel_map,
                c=c,
                label=str(freqs[0]) + " - " + str(freqs[1]),
            )
        meanfreq_powerAx.set_xlabel("Mean freq. band power(dB)")
        meanfreq_powerAx.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            mode="expand",
            fontsize="x-small",
            ncol=1,
        )
        if "saveas" in kwargs:
            saveas = kwargs["saveas"]
            plt.savefig(saveas)
        plt.show()


class OpenEphysNWB(OpenEphysBase):
    """
    Parameters
    ------------
    pname_root : str
        The top level directory, typically in form of YYYY-MM-DD_HH-MM-SS
    """

    def __init__(self, pname_root, **kwargs):
        super().__init__(pname_root)
        self.nwbData = None  # handle to the open nwb file (HDF5 file object)
        self.rawData = None  # np.array holding the raw, continuous recording
        self.recording_name = None  # the recording name ('recording0' etc)
        self.isBinary = False
        self.xy = None

    def load(
        self,
        pname_root: None,
        session_name=None,
        recording_name=None,
        loadraw=False,
        loadspikes=False,
        savedat=False,
    ):
        """
        Loads xy pos from binary part of the hdf5 file and data resulting from
        a Kilosort session (see KiloSortSession class above)

        Parameters
        ----------
        pname_root : str
            The top level directory, typically the one named
            YYYY-MM-DD_HH-MM-SS
            NB In the nwb format this directory contains the experiment_1.nwb
            and settings.xml files
        session_name : str
            Defaults to experiment_1.nwb
        recording_name : str
            Defaults to recording0
        loadraw : bool
            Defaults to False; if True will load and save the
            raw part of the data
        savedat : bool
            Defaults to False; if True will extract the electrode
            data part of the hdf file and save as 'experiment_1.dat'
            NB only works if loadraw is True. Also note that this
            currently saves 64 channels worth of data (ie ignores
            the 6 accelerometer channels)
        """

        import os

        import h5py

        if pname_root is None:
            pname_root = self.pname_root
        if session_name is None:
            session_name = "experiment_1.nwb"
        self.nwbData = h5py.File(os.path.join(pname_root, session_name), mode="r")
        # Position data...
        if self.recording_name is None:
            if recording_name is None:
                recording_name = "recording1"
            self.recording_name = recording_name
        try:
            self.xy = np.array(
                self.nwbData["acquisition"]["timeseries"][self.recording_name][
                    "events"
                ]["binary1"]["data"]
            )

            self.xyTS = np.array(
                self.nwbData["acquisition"]["timeseries"][self.recording_name][
                    "events"
                ]["binary1"]["timestamps"]
            )
            self.xy = self.xy[:, 0:2]
        except Exception:
            self.xy = None
            self.xyTS = None
        try:
            # TTL data...
            self.ttl_data = np.array(
                self.nwbData["acquisition"]["timeseries"][self.recording_name][
                    "events"
                ]["ttl1"]["data"]
            )
            self.ttl_timestamps = np.array(
                self.nwbData["acquisition"]["timeseries"][self.recording_name][
                    "events"
                ]["ttl1"]["timestamps"]
            )
        except Exception:
            self.ttl_data = None
            self.ttl_timestamps = None

        # ...everything else
        try:
            self.__loadSettings__()
            fpgaId = self.settings.processors["Sources/Rhythm FPGA"].NodeId
            fpgaNode = "processor" + str(fpgaId) + "_" + str(fpgaId)
            self.ts = np.array(
                self.nwbData["acquisition"]["timeseries"][self.recording_name][
                    "continuous"
                ][fpgaNode]["timestamps"]
            )
            if loadraw is True:
                print("Attempting to reference ALL the raw data...")
                self.rawData = self.nwbData["acquisition"][
                    "time\
                    series"
                ][self.recording_name][
                    "\
                        continuous"
                ][
                    fpgaNode
                ][
                    "data"
                ]
                print("Referenced the raw data! Yay!\nParsing channels...")
                self.settings.parseProcessor()  # get the neural data channels
                print("Channels parsed\nAccessing accelerometer data...")
                self.accelerometerData = self.rawData[:, 64:]
                print(
                    "Accessed the accelerometer data\
                    \nAttempting to access the raw data..."
                )
                self.rawData = self.rawData[:, 0:64]
                print("Got the raw data!")
                if savedat is True:
                    data2save = self.rawData[:, 0:64]
                    data2save.tofile(os.path.join(pname_root, "experiment_1.dat"))
            if loadspikes is True:
                if self.nwbData["acquisition"]["timeseries"][self.recording_name][
                    "\
                        spikes"
                ]:
                    # Create a dictionary containing keys 'electrode1',
                    # 'electrode2' etc and None for values
                    electrode_dict = dict.fromkeys(
                        self.nwbData["acquisition"][
                            "\
                            timeseries"
                        ][self.recording_name]["spikes"].keys()
                    )
                    # Each entry in the electrode dict is a dict
                    #  containing keys 'timestamps' and 'data'...
                    for i_electrode in electrode_dict.keys():
                        data_and_ts_dict = {"timestamps": None, "data": None}
                        data_and_ts_dict["timestamps"] = np.array(
                            self.nwbData["acquisition"][
                                "\
                                timeseries"
                            ][self.recording_name][
                                "\
                                    spikes"
                            ][
                                i_electrode
                            ][
                                "timestamps"
                            ]
                        )
                        data_and_ts_dict["data"] = np.array(
                            self.nwbData["acquisition"][
                                "\
                                timeseries"
                            ][self.recording_name][
                                "\
                                    spikes"
                            ][
                                i_electrode
                            ][
                                "data"
                            ]
                        )
                        electrode_dict[i_electrode] = data_and_ts_dict
                self.spikeData = electrode_dict
        except Exception:
            self.ts = self.xy


class OpenEphysBinary(OpenEphysBase):
    """
    The main class for dealing with data recorded using openephys
    and the Rhythm-FPGA module .
    """

    def __init__(self, pname_root: str):
        super().__init__(pname_root)
        self.path2PosData = None
        self.path2APdata = None
        self.path2LFPdata = None
        self.rawData = None

    def load(
        self,
        experiment_name="experiment1",
        recording_name="recording1",
        loadraw=False,
        n_channels=64,
    ):
        """
        Loads data recorded in the OE 'flat' binary format.

        Parameters
        ----------
        pname_root : str
            The top level directory, typically in form of YYYY-MM-DD_HH-MM-SS

        recording_name : str
            The directory immediately beneath pname_root

        See Also
        --------
        See open-ephys wiki
        """
        self.isBinary = True
        import os

        super().find_files(
            self.pname_root, experiment_name, recording_name, RecordingKind.FPGA
        )
        super().loadPos()

        n_channels = getattr(self, "n_channels", 384)
        trial_length = 0  # make sure a trial_length has a value
        ap_sample_rate = getattr(self, "ap_sample_rate", 30000)
        if self.path2APdata is not None:
            if fileExists(self.path2APdata, "continuous.dat"):
                trial_length = self.__calcTrialLengthFromBinarySize__(
                    os.path.join(self.path2APdata, "continuous.dat"),
                    n_channels,
                    ap_sample_rate,
                )

        if loadraw is True:
            if self.path2APdata is not None:
                if fileExists(self.path2APdata, "continuous.dat"):
                    status = os.stat(os.path.join(self.path2APdata, "continuous.dat"))
                    n_samples = int(status.st_size / 2 / n_channels)
                    mmap = np.memmap(
                        os.path.join(self.path2APdata, "continuous.dat"),
                        np.int16,
                        "r",
                        0,
                        (n_channels, n_samples),
                        "C",
                    )
                    self.rawData = np.array(mmap, dtype=np.float64)

        # this way of creating timestamps will be fine for single probes
        # but will need to be modified if using multiple probes and/ or
        # different timestamp syncing method
        # OE's 'synchronised_timestamps.npy' should now take care of this
        self.ts = np.arange(
            self.recording_start_time,
            trial_length + self.recording_start_time,
            1.0 / ap_sample_rate,
        )
