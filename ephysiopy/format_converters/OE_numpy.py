import numpy as np
import os
from ephysiopy.openephys2py import OESettings
from ephysiopy.io.recording import OpenEphysNWB
from scipy import signal


class OE2Numpy(object):
    """
    Converts openephys data recorded in the nwb format into numpy files.

    Notes
    -----
    Only exports the LFP and TTL files at the moment.
    """

    def __init__(self, filename_root: str):
        self.filename_root = filename_root  # /home/robin/Data/experiment_1.nwb
        self.dirname = os.path.dirname(filename_root)  # '/home/robin/Data'
        # 'experiment_1.nwb'
        self.experiment_name = os.path.basename(self.filename_root)
        self.recording_name = None  # will become 'recording1' etc
        self.OE_data = None  # becomes OpenEphysBase instance
        self._settings = None  # will become an instance of OESettings.Settings
        self.fs = None
        self.lfp_lowcut = None
        self.lfp_highcut = None

    def resample(self, data, src_rate=30, dst_rate=50, axis=0):
        """
        Resamples data using FFT.

        Parameters
        ----------
        data : array_like
            The input data to resample.
        src_rate : int, optional
            The source sampling rate. Default is 30.
        dst_rate : int, optional
            The destination sampling rate. Default is 50.
        axis : int, optional
            The axis along which to resample. Default is 0.

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
            self._settings = OESettings.Settings(self.dirname)
        return self._settings

    @settings.setter
    def settings(self, value):
        self._settings = value

    def getOEData(self, filename_root: str, recording_name="recording1") -> dict:
        """
        Loads the nwb file names in filename_root and returns a dict
        containing some of the nwb data relevant for converting to Axona
        file formats.

        Parameters
        ----------
        filename_root : str
            Fully qualified name of the nwb file.
        recording_name : str, optional
            The name of the recording in the nwb file. Default is 'recording1'.
            Note that the default has changed in different versions of OE from
            'recording0' to 'recording1'.

        Returns
        -------
        dict
            A dictionary containing the nwb data.Loads the nwb file names in
            filename_root and returns a dict
            containing some of the nwb data relevant for converting to Axona
            file formats.

        """
        if os.path.isfile(filename_root):
            OE_data = OpenEphysNWB(self.dirname)
            print("Loading nwb data...")
            OE_data.load(
                self.dirname,
                session_name=self.experiment_name,
                recording_name=recording_name,
                loadspikes=False,
                loadraw=True,
            )
            print("Loaded nwb data from: {}".format(filename_root))
            # It's likely that spikes have been collected after the last
            # position sample
            # due to buffering issues I can't be bothered to resolve.
            # Get the last pos
            # timestamps here and check that spikes don't go beyond this
            # when writing data out later
            # Also the pos and spike data timestamps almost never start at
            #  0 as the user
            # usually acquires data for a while before recording.
            # Grab the first timestamp
            # here with a view to subtracting this from everything
            # (including the spike data)
            # and figuring out what to keep later
            try:  # pos might not be present
                first_pos_ts = OE_data.xyTS[0]
                last_pos_ts = OE_data.xyTS[-1]
                self.first_pos_ts = first_pos_ts
                self.last_pos_ts = last_pos_ts
            except Exception:
                print("No position data in nwb file")
            self.recording_name = recording_name
            self.OE_data = OE_data
            return OE_data

    def exportLFP(self, channels: list, output_freq: int):
        """
        Exports LFP data to numpy files.

        Parameters
        ----------
        channels : list of int
            List of channel indices to export.
        output_freq : int
            The output sampling frequency.

        Notes
        -----
        The LFP data is resampled to the specified output frequency.
        """
        print("Beginning conversion and exporting of LFP data...")
        channels = [int(c) for c in channels]
        if not self.settings.processors:
            self.settings.parse()
        if self.settings.fpga_sample_rate is None:
            self.settings.parseProcessor()
        output_name = os.path.join(self.dirname, "lfp.npy")
        output_ts_name = os.path.join(self.dirname, "lfp_timestamps.npy")
        if len(channels) == 1:
            # resample data
            print(
                "Resampling data from {0} to {1} Hz".format(
                    self.settings.fpga_sample_rate, output_freq
                )
            )
            new_data = self.resample(
                self.OE_data.rawData[:, channels],
                self.settings.fpga_sample_rate,
                output_freq,
            )
            np.save(output_name, new_data, allow_pickle=False)
        if len(channels) > 1:
            print(
                "Resampling data from {0} to {1} Hz".format(
                    self.settings.fpga_sample_rate, output_freq
                )
            )
            new_data = self.resample(
                self.OE_data.rawData[:, channels[0] : channels[-1]],
                self.settings.fpga_sample_rate,
                output_freq,
            )
            np.save(output_name, new_data, allow_pickle=False)
        nsamples = np.shape(new_data)[0]
        new_ts = np.linspace(self.OE_data.ts[0], self.OE_data.ts[-1], nsamples)
        np.save(output_ts_name, new_ts, allow_pickle=False)
        print("Finished exporting LFP data")

    def exportTTL(self):
        """
        Exports TTL data to numpy files.

        Notes
        -----
        The TTL state and timestamps are saved as 'ttl_state.npy' and
        'ttl_timestamps.npy' respectively in the specified directory.
        """
        print("Exporting TTL data...")
        ttl_state = self.OE_data.ttl_data
        ttl_ts = self.OE_data.ttl_timestamps
        np.save(
            os.path.join(self.dirname, "ttl_state.npy"), ttl_state, allow_pickle=False
        )
        np.save(
            os.path.join(self.dirname, "ttl_timestamps.npy"), ttl_ts, allow_pickle=False
        )
        print("Finished exporting TTL data")

    def exportRaw2Binary(self, output_fname=None):
        """
        Exports raw data to a binary file.

        Parameters
        ----------
        output_fname : str, optional
            The name of the output binary file. If not provided, the output
            file name is derived from the root file name with a '.bin' extension.

        Notes
        -----
        The raw data is saved in binary format using numpy's save function.
        """
        if self.OE_data.rawData is None:
            print("Load the data first. See getOEData()")
            return
        if output_fname is None:
            output_fname = os.path.splitext(self.filename_root)[0] + ".bin"
        print(f"Exporting raw data to:\n{output_fname}")
        with open(output_fname, "wb") as f:
            np.save(f, self.OE_data.rawData)
        print("Finished exporting")
