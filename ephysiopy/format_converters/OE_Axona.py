from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy import signal
from tqdm import tqdm

from ephysiopy.axona import axonaIO
from ephysiopy.axona.file_headers import CutHeader, TetrodeHeader
from ephysiopy.io.recording import OpenEphysBase
from ephysiopy.openephys2py import OESettings


class OE2Axona(object):
    """
    Converts openephys data into Axona files

    Example workflow:

    You have recorded some openephys data using the binary
    format leading to a directory structure something like this:

    M4643_2023-07-21_11-52-02
    ├── Record Node 101
    │ ├── experiment1
    │ │ └── recording1
    │ │     ├── continuous
    │ │     │ └── Acquisition_Board-100.Rhythm Data
    │ │     │     ├── amplitudes.npy
    │ │     │     ├── channel_map.npy
    │ │     │     ├── channel_positions.npy
    │ │     │     ├── cluster_Amplitude.tsv
    │ │     │     ├── cluster_ContamPct.tsv
    │ │     │     ├── cluster_KSLabel.tsv
    │ │     │     ├── continuous.dat
    │ │     │     ├── params.py
    │ │     │     ├── pc_feature_ind.npy
    │ │     │     ├── pc_features.npy
    │ │     │     ├── phy.log
    │ │     │     ├── rez.mat
    │ │     │     ├── similar_templates.npy
    │ │     │     ├── spike_clusters.npy
    │ │     │     ├── spike_templates.npy
    │ │     │     ├── spike_times.npy
    │ │     │     ├── template_feature_ind.npy
    │ │     │     ├── template_features.npy
    │ │     │     ├── templates_ind.npy
    │ │     │     ├── templates.npy
    │ │     │     ├── whitening_mat_inv.npy
    │ │     │     └── whitening_mat.npy
    │ │     ├── events
    │ │     │ ├── Acquisition_Board-100.Rhythm Data
    │ │     │ │ └── TTL
    │ │     │ │     ├── full_words.npy
    │ │     │ │     ├── sample_numbers.npy
    │ │     │ │     ├── states.npy
    │ │     │ │     └── timestamps.npy
    │ │     │ └── MessageCenter
    │ │     │     ├── sample_numbers.npy
    │ │     │     ├── text.npy
    │ │     │     └── timestamps.npy
    │ │     ├── structure.oebin
    │ │     └── sync_messages.txt
    │ └── settings.xml
    └── Record Node 104
        ├── experiment1
        │ └── recording1
        │     ├── continuous
        │     │ └── TrackMe-103.TrackingNode
        │     │     ├── continuous.dat
        │     │     ├── sample_numbers.npy
        │     │     └── timestamps.npy
        │     ├── events
        │     │ ├── MessageCenter
        │     │ │ ├── sample_numbers.npy
        │     │ │ ├── text.npy
        │     │ │ └── timestamps.npy
        │     │ └── TrackMe-103.TrackingNode
        │     │     └── TTL
        │     │         ├── full_words.npy
        │     │         ├── sample_numbers.npy
        │     │         ├── states.npy
        │     │         └── timestamps.npy
        │     ├── structure.oebin
        │     └── sync_messages.txt
        └── settings.xml

    The binary data file is called "continuous.dat" in the
    continuous/Acquisition_Board-100.Rhythm Data folder. There
    is also a collection of files resulting from a KiloSort session
    in that directory.

    Run the conversion code like so:

    >>> from ephysiopy.format_converters.OE_Axona import OE2Axona
    >>> from pathlib import Path
    >>> nChannels = 64
    >>> apData = Path("M4643_2023-07-21_11-52-02/Record Node 101/experiment1/recording1/continuous/Acquisition_Board-100.Rhythm Data")
    >>> OE = OE2Axona(Path("M4643_2023-07-21_11-52-02"), path2APData=apData, channels=nChannels)
    >>> OE.getOEData()

    The last command will attempt to load position data and also load up
    something called a TemplateModel (from the package phylib) which
    should grab a handle to the neural data. If that doesn't throw
    out errors then try:

    >>> OE.exportPos()

    There are a few arguments you can provide the exportPos() function - see
    the docstring for it below. Basically, it calls a function called
    convertPosData(xy, xyts) where xy is the xy data with shape nsamples x 2
    and xyts is a vector of timestamps. So if the call to exportPos() fails, you
    could try calling convertPosData() directly which returns axona formatted
    position data. If the variable returned from convertPosData() is called axona_pos_data
    then you can call the function:

    writePos2AxonaFormat(pos_header, axona_pos_data)

    Providing the pos_header to it - see the last half of the exportPos function
    for how to create and modify the pos_header as that will need to have
    user-specific information added to it.

    >>> OE.convertTemplateDataToAxonaTetrode()

    This is the main function for creating the tetrode files. It has an optional
    argument called max_n_waves which is used to limit the maximum number of spikes
    that make up a cluster. This defaults to 2000 which means that if a cluster has
    12000 spikes, it will have 2000 spikes randomly drawn from those 12000 (without
    replacement), that will then be saved to a tetrode file. This is mostly a time-saving
    device as if you have 250 clusters and many consist of 10,000's of spikes,
    processing that data will take a long time.

    >>> OE.exportLFP()

    This will save either a .eeg or .egf file depending on the arguments. Check the
    docstring for how to change what channel is chosen for the LFP etc.

    >>> OE.exportSetFile()

    This should save the .set file with all the metadata for the trial.

    """

    def __init__(
        self,
        pname: Path,
        path2APData: Path = None,
        pos_sample_rate: int = 50,
        channels: int = 0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        pname : Path
            The base directory of the openephys recording.
            e.g. '/home/robin/Data/M4643_2023-07-21_11-52-02'
        path2APData : Path, optional
            Path to AP data. Defaults to None.
        pos_sample_rate : int, optional
            Position sample rate. Defaults to 50.
        channels : int, optional
            Number of channels. Defaults to 0.
        **kwargs
            Variable length argument list.

        """
        pname = Path(pname)
        assert pname.exists()
        self.pname: Path = pname
        self.path2APdata: Path = path2APData
        self.pos_sample_rate: int = pos_sample_rate
        # 'experiment_1.nwb'
        self.experiment_name: Path = self.pname or Path(kwargs["experiment_name"])
        self.recording_name = None  # will become 'recording1' etc
        self.OE_data = None  # becomes instance of io.recording.OpenEphysBase
        self._settings = None  # will become an instance of OESettings.Settings
        # Create a basename for Axona file names
        # e.g.'/home/robin/Data/experiment_1'
        # that we can append '.pos' or '.eeg' or whatever onto
        self.axona_root_name = self.experiment_name
        # need to instantiated now for later
        self.AxonaData = axonaIO.IO(self.axona_root_name.name + ".pos")
        # THIS IS TEMPORARY AND WILL BE MORE USER-SPECIFIABLE IN THE FUTURE
        # it is used to scale the spikes
        self.hp_gain = 500
        self.lp_gain = 15000
        self.bitvolts = 0.195
        # if left as None some default values for the next 3 params are loaded
        #  from top-level __init__.py
        # these are only used in self.__filterLFP__
        self.fs = None
        # if lfp_channel is set to None then the .set file will reflect that
        #  no EEG was recorded
        # this should mean that you can load data into Tint without a .eeg file
        self.lfp_channel = 1 or kwargs["lfp_channel"]
        self.lfp_lowcut = None
        self.lfp_highcut = None
        # set the tetrodes to record from
        # defaults to 1 through 4 - see self.makeSetData below
        self.tetrodes = ["1", "2", "3", "4"]
        self.channel_count = channels

    def resample(self, data, src_rate=30, dst_rate=50, axis=0):
        """
        Resamples data using FFT.

        Parameters
        ----------
        data : array_like
            The input data to be resampled.
        src_rate : int, optional
            The original sampling rate of the data. Defaults to 30.
        dst_rate : int, optional
            The desired sampling rate of the resampled data. Defaults to 50.
        axis : int, optional
            The axis along which to resample. Defaults to 0.

        Returns
        -------
        new_data : ndarray
            The resampled data.

        """
        denom = np.gcd(dst_rate, src_rate)
        new_data = signal.resample_poly(data, dst_rate / denom, src_rate / denom, axis)
        return new_data

    @property
    def settings(self):
        """
        Loads the settings data from the settings.xml file
        """
        if self._settings is None:
            self._settings = OESettings.Settings(self.pname)
        return self._settings

    @settings.setter
    def settings(self, value):
        self._settings = value

    def getOEData(self) -> OpenEphysBase:
        """
        Loads the nwb file names in filename_root and returns a dict
        containing some of the nwb data relevant for converting to Axona file formats.

        Parameters
        ----------
        filename_root : str
            Fully qualified name of the nwb file.
        recording_name : str
            The name of the recording in the nwb file. Note that
            the default has changed in different versions of OE from 'recording0'
            to 'recording1'.

        Returns
        -------
        OpenEphysBase
            An instance of OpenEphysBase containing the loaded data.
        """
        OE_data = OpenEphysBase(self.pname)
        try:
            OE_data.load_pos_data(sample_rate=self.pos_sample_rate)
            # It's likely that spikes have been collected after the last
            # position sample
            # due to buffering issues I can't be bothered to resolve.
            # Get the last pos
            # timestamps here and check that spikes don't go beyond
            #  this when writing data
            # out later
            # Also the pos and spike data timestamps almost never start at
            #  0 as the user
            # usually acquires data for a while before recording.
            # Grab the first timestamp
            # here with a view to subtracting this from everything (including
            # the spike data)
            # and figuring out what to keep later
            first_pos_ts = OE_data.PosCalcs.xyTS[0]
            last_pos_ts = OE_data.PosCalcs.xyTS[-1]
            self.first_pos_ts = first_pos_ts
            self.last_pos_ts = last_pos_ts
        except Exception:
            OE_data.load_neural_data()  # will create TemplateModel instance
            self.first_pos_ts = 0
            self.last_pos_ts = self.OE_data.template_model.duration
        print(f"First pos ts: {self.first_pos_ts}")
        print(f"Last pos ts: {self.last_pos_ts}")
        self.OE_data = OE_data
        if self.path2APdata is None:
            self.path2APdata = self.OE_data.path2APdata
        # extract number of channels from settings
        for item in self.settings.record_nodes.items():
            if "Rhythm Data" in item[1].name:
                self.channel_count = int(item[1].channel_count)
        return OE_data

    def exportSetFile(self, **kwargs):
        """
        Wrapper for makeSetData below
        """
        print("Exporting set file data...")
        self.makeSetData(**kwargs)
        print("Done exporting set file.")

    def exportPos(self, ppm=300, jumpmax=100, as_text=False, **kwargs):
        """
        Exports position data to either text or Axona format.

        Parameters
        ----------
        ppm : int, optional
            Pixels per meter. Defaults to 300.
        jumpmax : int,def
            Maximum allowed jump in position data. Defaults to 100.
        as_text : bool, optional
            If True, exports position data to text format. Defaults to False.
        **kwargs
            Additional keyword arguments.


        """
        self.settings.parse()
        if not self.OE_data:
            self.getOEData()
        if not self.OE_data.PosCalcs:
            self.OE_data.load_pos_data(sample_rate=self.pos_sample_rate)
        print("Post-processing position data...")
        self.OE_data.PosCalcs.jumpmax = jumpmax
        self.OE_data.PosCalcs.tracker_params["AxonaBadValue"] = 1023
        self.OE_data.PosCalcs.postprocesspos(self.OE_data.PosCalcs.tracker_params)
        xy = self.OE_data.PosCalcs.xy.T
        xyTS = self.OE_data.PosCalcs.xyTS  # in seconds
        xyTS = xyTS * self.pos_sample_rate
        # extract some values from PosCalcs or overrides given
        # when calling this method
        ppm = self.OE_data.PosCalcs.ppm or ppm
        sample_rate = self.OE_data.PosCalcs.sample_rate or kwargs["sample_rate"]
        if as_text is True:
            print("Beginning export of position data to text format...")
            pos_file_name = self.axona_root_name + ".txt"
            np.savetxt(pos_file_name, xy, fmt="%1.u")
            print("Completed export of position data")
            return
        # Do the upsampling of both xy and the timestamps
        print("Beginning export of position data to Axona format...")
        axona_pos_data = self.convertPosData(xy, xyTS)
        # make sure pos data length is same as duration * num_samples
        axona_pos_data = axona_pos_data[
            0 : int(self.last_pos_ts - self.first_pos_ts) * self.pos_sample_rate
        ]
        # Create an empty header for the pos data
        from ephysiopy.axona.file_headers import PosHeader

        pos_header = PosHeader()
        tracker_params = self.OE_data.PosCalcs.tracker_params
        min_xy = np.floor(np.min(xy, 0)).astype(int).data
        max_xy = np.ceil(np.max(xy, 0)).astype(int).data
        pos_header.pos["min_x"] = pos_header.pos["window_min_x"] = (
            str(tracker_params["LeftBorder"])
            if "LeftBorder" in tracker_params.keys()
            else str(min_xy[0])
        )
        pos_header.pos["min_y"] = pos_header.pos["window_min_y"] = (
            str(tracker_params["TopBorder"])
            if "TopBorder" in tracker_params.keys()
            else str(min_xy[1])
        )
        pos_header.pos["max_x"] = pos_header.pos["window_max_x"] = (
            str(tracker_params["RightBorder"])
            if "RightBorder" in tracker_params.keys()
            else str(max_xy[0])
        )
        pos_header.pos["max_y"] = pos_header.pos["window_max_y"] = (
            str(tracker_params["BottomBorder"])
            if "BottomBorder" in tracker_params.keys()
            else str(max_xy[1])
        )
        pos_header.common["duration"] = str(int(self.last_pos_ts - self.first_pos_ts))
        pos_header.pos["pixels_per_metre"] = str(ppm)
        pos_header.pos["num_pos_samples"] = str(len(axona_pos_data))
        pos_header.pos["pixels_per_metre"] = str(ppm)
        pos_header.pos["sample_rate"] = str(sample_rate)

        self.writePos2AxonaFormat(pos_header, axona_pos_data)
        print("Exported position data to Axona format")

    def exportSpikes(self):
        """
        Exports spiking data.

        Notes
        -----
        Converts spiking data from the Open Ephys format to the Axona format.
        """
        print("Beginning conversion of spiking data...")
        self.convertSpikeData(
            self.OE_data.nwbData["acquisition"][
                "\
                timeseries"
            ][self.recording_name]["spikes"]
        )
        print("Completed exporting spiking data")

    def exportLFP(
        self, channel: int = 0, lfp_type: str = "eeg", gain: int = 5000, **kwargs
    ):
        """
        Exports LFP data to file.

        Parameters
        ----------
        channel : int, optional
            The channel number. Default is 0.
        lfp_type : str, optional
            The type of LFP data. Legal values are 'egf' or 'eeg'. Default is 'eeg'.
        gain : int, optional
            Multiplier for the LFP data. Default is 5000.

        Notes
        -----
        Converts and exports LFP data from the Open Ephys format to the Axona format.
            gain (int): Multiplier for the LFP data.
        """
        print("Beginning conversion and exporting of LFP data...")
        if not self.settings.processors:
            self.settings.parse()
        from ephysiopy.io.recording import memmapBinaryFile

        try:
            data = memmapBinaryFile(
                Path(self.path2APdata).joinpath("continuous.dat"),
                n_channels=self.channel_count,
            )
            self.makeLFPData(data[channel, :], eeg_type=lfp_type, gain=gain)
            print("Completed exporting LFP data to " + lfp_type + " format")
        except Exception as e:
            print(f"Couldn't load raw data:\n{e}")

    def convertPosData(self, xy: np.array, xy_ts: np.array) -> np.array:
        """
        Performs the conversion of the array parts of the data.

        Parameters
        ----------
        xy : np.array
            The x and y coordinates.
        xy_ts : np.array
            The timestamps for the x and y coordinates.

        Returns
        -------
        np.array
            The converted position data.

        Notes
        -----
        Upsamples the data to the Axona position sampling rate (50Hz) and inserts
        columns into the position array to match the Axona format.
        """
        n_new_pts = int(
            np.floor((self.last_pos_ts - self.first_pos_ts) * self.pos_sample_rate)
        )
        t = xy_ts - self.first_pos_ts
        new_ts = np.linspace(t[0], t[-1], n_new_pts)
        new_x = np.interp(new_ts, t, xy[:, 0])
        new_y = np.interp(new_ts, t, xy[:, 1])
        new_x[np.isnan(new_x)] = 1023
        new_y[np.isnan(new_y)] = 1023
        # Expand the pos bit of the data to make it look like Axona data
        new_pos = np.vstack([new_x, new_y]).T
        new_pos = np.c_[
            new_pos,
            np.ones_like(new_pos) * 1023,
            np.zeros_like(new_pos),
            np.zeros_like(new_pos),
        ]
        new_pos[:, 4] = 40  # just made this value up - it's numpix i think
        new_pos[:, 6] = 40  # same
        # Squeeze this data into Axona pos format array
        dt = self.AxonaData.axona_files[".pos"]
        new_data = np.zeros(n_new_pts, dtype=dt)
        # Timestamps in Axona are pos_samples (monotonic, linear integer)
        new_data["ts"] = new_ts
        new_data["pos"] = new_pos
        return new_data

    def convertTemplateDataToAxonaTetrode(self, max_n_waves=2000, **kwargs):
        """
        Converts the data held in a TemplateModel instance into tetrode format Axona data files.

        Parameters
        ----------
        max_n_waves : int, default=2000
            The maximum number of waveforms to process.

        Notes
        -----
        For each cluster, the channel with the peak amplitude is identified, and the
        data is converted to the Axona tetrode format. If a channel from a tetrode is
        missing, the spikes for that channel are zeroed when saved to the Axona format.

        Examples
        --------
        If cluster 3 has a peak channel of 1 then get_cluster_channels() might look like:
        [ 1,  2,  0,  6, 10, 11,  4,  12,  7,  5,  8,  9]
        Here the cluster has the best signal on 1, then 2, 0 etc, but note that channel 3
        isn't in the list. In this case the data for channel 3 will be zeroed
        when saved to Axona format.

        References
        ----------
        .. [1] https://phy.readthedocs.io/en/latest/api/#phyappstemplatetemplatemodel
        """
        # First lets get the datatype for tetrode files as this will be the
        # same for all tetrodes...
        dt = self.AxonaData.axona_files[".1"]
        # Load the TemplateModel
        if "path2APdata" in kwargs.keys():
            self.OE_data.load_neural_data(**kwargs)
        else:
            self.OE_data.load_neural_data()
        model = self.OE_data.template_model
        clusts = model.cluster_ids
        # have to pre-process the channels / clusters to determine
        # which tetrodes clusters belong to - this is based on
        # the 'best' channel for a given cluster
        clusters_channels = OrderedDict(dict.fromkeys(clusts, np.ndarray))
        for c in clusts:
            clusters_channels[c] = model.get_cluster_channels(c)
        tetrodes_clusters = OrderedDict(
            dict.fromkeys(range(0, int(self.channel_count / 4)), [])
        )
        for t in tetrodes_clusters.items():
            this_tetrodes_clusters = []
            for c in clusters_channels.items():
                if int(c[1][0] / 4) == t[0]:
                    this_tetrodes_clusters.append(c[0])
            tetrodes_clusters[t[0]] = this_tetrodes_clusters
        # limit the number of spikes to max_n_waves in the
        # interests of speed. Randomly select spikes across
        # the period they fire
        rng = np.random.default_rng()

        for i, i_tet_item in enumerate(tetrodes_clusters.items()):
            this_tetrode = i_tet_item[0]
            times_to_sort = []
            new_clusters = []
            new_waves = []
            for clust in tqdm(i_tet_item[1], desc="Tetrode " + str(i + 1)):
                clust_chans = model.get_cluster_channels(clust)
                idx = np.logical_and(
                    clust_chans >= this_tetrode, clust_chans < this_tetrode + 4
                )
                # clust_chans is an ordered list of the channels
                # the cluster was most active on. idx has True
                # where there is overlap between that and the
                # currently active tetrode channel numbers (0:4, 4:8
                # or whatever)
                spike_idx = model.get_cluster_spikes(clust)
                # limit the number of spikes to max_n_waves in the
                # interests of speed. Randomly select spikes across
                # the period they fire
                total_n_waves = len(spike_idx)
                max_num_waves = (
                    max_n_waves if max_n_waves < total_n_waves else total_n_waves
                )
                # grab spike times (in seconds) so the random sampling of
                # spikes matches their times
                times = model.spike_times[model.spike_clusters == clust]
                spike_idx_times_subset = rng.choice(
                    (spike_idx, times), max_num_waves, axis=1, replace=False
                )
                # spike_idx_times_subset is unsorted as it's just been drawn
                # from a random distribution, so sort it now
                spike_idx_times_subset = np.sort(spike_idx_times_subset, 1)
                # split out into spikes and times
                spike_idx_subset = spike_idx_times_subset[0, :].astype(int)
                times = spike_idx_times_subset[1, :]
                waves = model.get_waveforms(spike_idx_subset, clust_chans[idx])
                # Given a spike at time T, Axona takes T-200us and T+800us
                # from the buffer to make up a waveform. From OE
                # take 30 samples which corresponds to a 1ms sample
                # if the data is sampled at 30kHz. Interpolate this so the
                # length is 50 samples as with Axona
                waves = waves[:, 30:60, :]
                # waves go from int16 to float as a result of the resampling
                waves = self.resample(waves.astype(float), axis=1)
                # multiply by bitvolts to get microvolts
                waves = waves * self.bitvolts
                # scale appropriately for Axona and invert as
                # OE seems to be inverted wrt Axona
                waves = waves / (self.hp_gain / 4 / 128.0) * (-1)
                # check the shape of waves to make sure it has 4
                # channels, if not add some to make it so and make
                # sure they are in the correct order for the tetrode
                ordered_chans = np.argsort(clust_chans[idx])
                if waves.shape[-1] != 4:
                    z = np.zeros(shape=(waves.shape[0], waves.shape[1], 4))
                    z[:, :, ordered_chans] = waves
                    waves = z
                else:
                    waves = waves[:, :, ordered_chans]
                # Axona format tetrode waveforms are nSpikes x 4 x 50
                waves = np.transpose(waves, (0, 2, 1))
                # Append clusters to a list to sort later for saving a
                # cluster/ cut file
                new_clusters.append(np.repeat(clust, len(times)))
                # Axona times are sampled at 96KHz
                times = times * 96000
                # There is a time for each spike despite the repetition
                # get the indices for sorting
                times_to_sort.append(times)
                # i_clust_data = np.zeros(len(new_times), dtype=dt)
                new_waves.append(waves)
            # Concatenate, order and reshape some of the lists/ arrays
            if times_to_sort:  # apparently can be empty sometimes
                _times = np.concatenate(times_to_sort)
                _waves = np.concatenate(new_waves)
                _clusters = np.concatenate(new_clusters)
                indices = np.argsort(_times)
                sorted_times = _times[indices]
                sorted_waves = _waves[indices]
                sorted_clusts = _clusters[indices]
                output_times = np.repeat(sorted_times, 4)
                output_waves = np.reshape(
                    sorted_waves,
                    [
                        sorted_waves.shape[0] * sorted_waves.shape[1],
                        sorted_waves.shape[2],
                    ],
                )
                new_tetrode_data = np.zeros(len(output_times), dtype=dt)
                new_tetrode_data["ts"] = output_times
                new_tetrode_data["waveform"] = output_waves
                header = TetrodeHeader()
                header.common["duration"] = str(int(model.duration))
                header.tetrode_entries["num_spikes"] = str(len(_clusters))
                self.writeTetrodeData(str(i + 1), header, new_tetrode_data)
                cut_header = CutHeader()
                self.writeCutData(str(i + 1), cut_header, sorted_clusts)

    def convertSpikeData(self, hdf5_tetrode_data: h5py._hl.group.Group):
        """
        Converts spike data from the Open Ephys Spike Sorter format to Axona format tetrode files.

        Parameters
        ----------
        hdf5_tetrode_data : h5py._hl.group.Group
            The HDF5 group containing the tetrode data.

        Notes
        -----
        Converts the spike data and timestamps, scales them appropriately, and saves
        them in the Axona tetrode format.
        """
        # First lets get the datatype for tetrode files as this will be the
        # same for all tetrodes...
        dt = self.AxonaData.axona_files[".1"]
        # ... and a basic header for the tetrode file that use for each
        # tetrode file, changing only the num_spikes value
        header = TetrodeHeader()
        header.common["duration"] = str(int(self.last_pos_ts - self.first_pos_ts))

        for key in hdf5_tetrode_data.keys():
            spiking_data = np.array(hdf5_tetrode_data[key].get("data"))
            timestamps = np.array(hdf5_tetrode_data[key].get("timestamps"))
            # check if any of the spiking data is captured before/ after the
            #  first/ last bit of position data
            # if there is then discard this as we potentially have no valid
            # position to align the spike to :(
            idx = np.logical_or(
                timestamps < self.first_pos_ts, timestamps > self.last_pos_ts
            )
            spiking_data = spiking_data[~idx, :, :]
            timestamps = timestamps[~idx]
            # subtract the first pos timestamp from the spiking timestamps
            timestamps = timestamps - self.first_pos_ts
            # get the number of spikes here for use below in the header
            num_spikes = len(timestamps)
            # repeat the timestamps in tetrode multiples ready for Axona export
            new_timestamps = np.repeat(timestamps, 4)
            new_spiking_data = spiking_data.astype(np.float64)
            # Convert to microvolts...
            new_spiking_data = new_spiking_data * self.bitvolts
            # And upsample the spikes...
            new_spiking_data = self.resample(new_spiking_data, 4, 5, -1)
            # ... and scale appropriately for Axona and invert as
            # OE seems to be inverted wrt Axona
            new_spiking_data = new_spiking_data / (self.hp_gain / 4 / 128.0) * (-1)
            # ... scale them to the gains specified somewhere
            #  (not sure where / how to do this yet)
            shp = new_spiking_data.shape
            # then reshape them as Axona wants them a bit differently
            new_spiking_data = np.reshape(new_spiking_data, [shp[0] * shp[1], shp[2]])
            # Cap any values outside the range of int8
            new_spiking_data[new_spiking_data < -128] = -128
            new_spiking_data[new_spiking_data > 127] = 127
            # create the new array
            new_tetrode_data = np.zeros(len(new_timestamps), dtype=dt)
            new_tetrode_data["ts"] = new_timestamps * 96000
            new_tetrode_data["waveform"] = new_spiking_data
            # change the header num_spikes field
            header.tetrode_entries["num_spikes"] = str(num_spikes)
            i_tetnum = key.split("electrode")[1]
            print("Exporting tetrode {}".format(i_tetnum))
            self.writeTetrodeData(i_tetnum, header, new_tetrode_data)

    def makeLFPData(self, data: np.ndarray, eeg_type="eeg", gain=5000):
        """
        Downsamples the data and saves the result as either an EGF or EEG file.

        Parameters
        ----------
        data : np.ndarray
            The data to be downsampled. Must have dtype as np.int16.
        eeg_type : str, optional
            The type of LFP data. Legal values are 'egf' or 'eeg'. Default is 'eeg'.
        gain : int, optional
            The scaling factor. Default is 5000.

        Notes
        -----
        Downsamples the data to the specified rate and applies a filter. The data is
        then scaled and saved in the Axona format.
        """
        if eeg_type == "eeg":
            from ephysiopy.axona.file_headers import EEGHeader

            header = EEGHeader()
            dst_rate = 250
        elif eeg_type == "egf":
            from ephysiopy.axona.file_headers import EGFHeader

            header = EGFHeader()
            dst_rate = 4800
        header.common["duration"] = str(int(self.last_pos_ts - self.first_pos_ts))
        print(f"header.common[duration] = {header.common['duration']}")
        _lfp_data = self.resample(data.astype(float), 30000, dst_rate, -1)
        # make sure data is same length as sample_rate * duration
        nsamples = int(dst_rate * int(header.common["duration"]))
        # lfp_data might be shorter than nsamples. If so, fill the
        # remaining values with zeros
        if len(_lfp_data) < nsamples:
            lfp_data = np.zeros(nsamples)
            lfp_data[0 : len(_lfp_data)] = _lfp_data
        else:
            lfp_data = _lfp_data[0:nsamples]
        lfp_data = self.__filterLFP__(lfp_data, dst_rate)
        # convert the data format
        # lfp_data = lfp_data * self.bitvolts # in microvolts

        if eeg_type == "eeg":
            # probably BROKEN
            # lfp_data starts out as int16 (see Parameters above)
            # but gets converted into float64 as part of the
            # resampling/ filtering process
            lfp_data = lfp_data / 32768.0
            lfp_data = lfp_data * gain
            # cap the values at either end...
            lfp_data[lfp_data < -128] = -128
            lfp_data[lfp_data > 127] = 127
            # and convert to int8
            lfp_data = lfp_data.astype(np.int8)

        elif eeg_type == "egf":
            # probably works
            # lfp_data = lfp_data / 256.
            lfp_data = lfp_data.astype(np.int16)

        header.n_samples = str(len(lfp_data))
        self.writeLFP2AxonaFormat(header, lfp_data, eeg_type)

    def makeSetData(self, lfp_channel=4, **kwargs):
        """
        Creates and writes the SET file data.

        Parameters
        ----------
        lfp_channel : int, optional
            The LFP channel number. Default is 4.

        Notes
        -----
        Creates the SET file header and entries based on the provided parameters and
        writes the data to the Axona format.
        """
        if self.OE_data is None:
            # to get the timestamps for duration key
            self.getOEData(self.filename_root)
        from ephysiopy.axona.file_headers import SetHeader

        header = SetHeader()
        # set some reasonable default values
        from ephysiopy.__about__ import __version__

        header.meta_info["sw_version"] = str(__version__)
        # ADC fullscale mv is 1500 in Axona and 0.195 in OE
        # there is a division by 1000 that happens when processing
        # spike data in Axona that looks like that has already
        # happened in OE. So here the OE 0.195 value is multiplied
        # by 1000 as it will get divided by 1000 later on to get
        # the correct scaling of waveforms/ gains -> mv values
        header.meta_info["ADC_fullscale_mv"] = "195"
        header.meta_info["tracker_version"] = "1.1.0"

        for k, v in header.set_entries.items():
            if "gain" in k:
                header.set_entries[k] = str(self.hp_gain)
            if "collectMask" in k:
                header.set_entries[k] = "0"
            if "EEG_ch_1" in k:
                if lfp_channel is not None:
                    header.set_entries[k] = str(int(lfp_channel))
            if "mode_ch_" in k:
                header.set_entries[k] = "0"
        # iterate again to make sure lfp gain set correctly
        for k, v in header.set_entries.items():
            if lfp_channel is not None:
                if k == "gain_ch_" + str(lfp_channel):
                    header.set_entries[k] = str(self.lp_gain)

        # Based on the data in the electrodes dict of the OESettings
        # instance (self.settings - see __init__)
        # determine which tetrodes we can let Tint load
        # make sure we've parsed the electrodes
        tetrode_count = int(self.channel_count / 4)
        for i in range(1, tetrode_count + 1):
            header.set_entries["collectMask_" + str(i)] = "1"
        # if self.lfp_channel is not None:
        #     for chan in self.tetrodes:
        #         key = "collectMask_" + str(chan)
        #         header.set_entries[key] = "1"
        header.set_entries["colactive_1"] = "1"
        header.set_entries["colactive_2"] = "0"
        header.set_entries["colactive_3"] = "0"
        header.set_entries["colactive_4"] = "0"
        header.set_entries["colmap_algorithm"] = "1"
        header.set_entries["duration"] = str(int(self.last_pos_ts - self.first_pos_ts))
        self.writeSetData(header)

    def __filterLFP__(self, data: np.array, sample_rate: int):
        """
        Filters the LFP data.

        Parameters
        ----------
        data : np.array
            The LFP data to be filtered.
        sample_rate : int
            The sampling rate of the data.

        Returns
        -------
        np.array
            The filtered LFP data.

        Notes
        -----
        Applies a bandpass filter to the LFP data using the specified sample rate.
        """
        from scipy.signal import filtfilt, firwin

        if self.fs is None:
            from ephysiopy import fs

            self.fs = fs
        if self.lfp_lowcut is None:
            from ephysiopy import lfp_lowcut

            self.lfp_lowcut = lfp_lowcut
        if self.lfp_highcut is None:
            from ephysiopy import lfp_highcut

            self.lfp_highcut = lfp_highcut
        nyq = sample_rate / 2.0
        lowcut = self.lfp_lowcut / nyq
        highcut = self.lfp_highcut / nyq
        if highcut >= 1.0:
            highcut = 1.0 - np.finfo(float).eps
        if lowcut <= 0.0:
            lowcut = np.finfo(float).eps
        b = firwin(sample_rate + 1, [lowcut, highcut], window="black", pass_zero=False)
        y = filtfilt(b, [1], data.ravel(), padtype="odd")
        return y

    def writeLFP2AxonaFormat(self, header: dataclass, data: np.array, eeg_type="eeg"):
        """
        Writes LFP data to the Axona format.

        Parameters
        ----------
        header : dataclass
            The header information for the LFP file.
        data : np.array
            The LFP data to be written.
        eeg_type : str, optional
            The type of LFP data. Legal values are 'egf' or 'eeg'. Default is 'eeg'.

        Notes
        -----
        Writes the LFP data and header to the Axona format.
        """
        self.AxonaData.setHeader(str(self.axona_root_name) + "." + eeg_type, header)
        self.AxonaData.setData(str(self.axona_root_name) + "." + eeg_type, data)

    def writePos2AxonaFormat(self, header: dataclass, data: np.array):
        self.AxonaData.setHeader(str(self.axona_root_name) + ".pos", header)
        self.AxonaData.setData(str(self.axona_root_name) + ".pos", data)

    def writeTetrodeData(self, itet: str, header: dataclass, data: np.array):
        """
        Writes tetrode data to the Axona format.

        Parameters
        ----------
        itet : str
            The tetrode identifier.
        header : dataclass
            The header information for the tetrode file.
        data : np.array
            The tetrode data to be written.

        Notes
        -----
        Writes the tetrode data and header to the Axona format.
        """
        self.AxonaData.setHeader(str(self.axona_root_name) + "." + itet, header)
        self.AxonaData.setData(str(self.axona_root_name) + "." + itet, data)

    def writeSetData(self, header: dataclass):
        """
        Writes SET data to the Axona format.

        Parameters
        ----------
        header : dataclass
            The header information for the SET file.

        Notes
        -----
        Writes the SET data and header to the Axona format.
        """
        self.AxonaData.setHeader(str(self.axona_root_name) + ".set", header)

    def writeCutData(self, itet: str, header: dataclass, data: np.array):
        """
        Writes cut data to the Axona format.

        Parameters
        ----------
        itet : str
            The tetrode identifier.
        header : dataclass
            The header information for the cut file.
        data : np.array
            The cut data to be written.

        Notes
        -----
        Writes the cut data and header to the Axona format.
        """
        self.AxonaData.setCut(
            str(self.axona_root_name) + "_" + str(itet) + ".cut", header, data
        )
