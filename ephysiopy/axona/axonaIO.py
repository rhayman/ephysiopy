import fnmatch
import os
import re
from contextlib import redirect_stdout
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from ephysiopy.axona.file_headers import make_cluster_cut_entries


MAXSPEED = 4.0  # pos data speed filter in m/s
BOXCAR = 20  # this gives a 400ms smoothing window for pos averaging


class IO(object):
    """
    Axona data I/O. Also reads .clu files generated from KlustaKwik

    Parameters
    ----------
    filename_root : str
        The fully-qualified filename
    """

    tetrode_files = dict.fromkeys(
        ["." + str(i) for i in range(1, 17)],
        # ts is a big-endian 32-bit integer
        # waveform is 50 signed 8-bit ints (a signed byte)
        [("ts", ">i"), ("waveform", "50b")]
    )
    other_files = {
        ".pos": [("ts", ">i"), ("pos", ">8h")],
        ".eeg": [("eeg", "=b")],
        ".eeg2": [("eeg", "=b")],
        ".egf": [("eeg", "int16")],
        ".egf2": [("eeg", "int16")],
        ".inp": [("ts", ">i4"), ("type", ">b"), ("value", ">2b")],
        ".log": [("state", "S3"), ("ts", ">i")],
        ".stm": [("ts", ">i")],
    }

    # this only works in >= Python3.5
    axona_files = {**other_files, **tetrode_files}

    def __init__(self, filename_root: Path = ""):
        self.filename_root = filename_root

    def getData(self, filename_root: str) -> np.ndarray:
        """
        Returns the data part of an Axona data file i.e. from "data_start" to
        "data_end"

        Parameters
        ----------
        input :  str
            Fully qualified path name to the data file

        Returns
        -------
        output : ndarray
            The data part of whatever file was fed in
        """
        n_samps = -1
        fType = os.path.splitext(filename_root)[1]
        if fType in self.axona_files:
            header = self.getHeader(filename_root)
            for key in header.keys():
                if len(fType) > 2:
                    if fnmatch.fnmatch(key, "num_*_samples"):
                        n_samps = int(header[key])
                else:
                    if key.startswith("num_spikes"):
                        n_samps = int(header[key]) * 4
            f = open(filename_root, "rb")
            data = f.read()
            st = data.find(b"data_start") + len("data_start")
            f.seek(st)
            dt = np.dtype(self.axona_files[fType])
            a = np.fromfile(f, dtype=dt, count=n_samps)
            f.close()
        else:
            raise IOError("File not in list of recognised Axona files")
        return a

    def getCluCut(self, tet: int) -> np.ndarray:
        """
        Load a clu file and return as an array of integers

        Parameters
        ----------
        tet : int
            The tetrode the clu file relates to

        Returns
        -------
        out : ndarray
            Data read from the clu file
        """
        filename_root = self.filename_root.with_suffix("." + "clu." + str(tet))
        if os.path.exists(filename_root):
            dt = np.dtype([("data", "<i")])
            clu_data = np.loadtxt(filename_root, dtype=dt)
            return clu_data["data"][1::]  # first entry is num of clusters
        else:
            return None

    def getCut(self, tet: int) -> list:
        """
        Returns the cut file as a list of integers

        Parameters
        ----------
        tet : int
            The tetrode the cut file relates to

        Returns
        -------
        out : ndarray
            The data read from the cut file
        """
        a = []
        filename_root = Path(os.path.splitext(
            self.filename_root)[0] + "_" + str(tet) + ".cut")

        if not os.path.exists(filename_root):
            cut = self.getCluCut(tet)
            if cut is not None:
                return cut - 1  # clusters 1 indexed in clu
            return cut
        with open(filename_root, "r") as f:
            cut_data = f.read()
            f.close()
        tmp = cut_data.split("spikes: ")
        tmp1 = tmp[1].split("\n")
        cut = tmp1[1:]
        for line in cut:
            m = line.split()
            for i in m:
                a.append(int(i))
        return a

    def setHeader(self, filename_root: str, header: dataclass):
        """
        Writes out the header to the specified file

        Parameters
        ------------
        filename_root : str
            A fully qualified path to a file with the relevant suffix at
            the end (e.g. ".set", ".pos" or whatever)

        header : dataclass
            See ephysiopy.axona.file_headers
        """
        with open(filename_root, "w") as f:
            with redirect_stdout(f):
                header.print()
            f.write("data_start")
            f.write("\r\n")
            f.write("data_end")
            f.write("\r\n")

    def setCut(self, filename_root: str,
               cut_header: dataclass,
               cut_data: np.array):
        fpath = Path(filename_root)
        n_clusters = len(np.unique(cut_data))
        cluster_entries = make_cluster_cut_entries(n_clusters)
        with open(filename_root, "w") as f:
            with redirect_stdout(f):
                cut_header.print()
            print(cluster_entries, file=f)
            print(f"Exact_cut_for: {fpath.stem}    spikes: {len(cut_data)}",
                  file=f)
            for num in cut_data:
                f.write(str(num))
                f.write(" ")

    def setData(self, filename_root: str, data: np.array):
        """
        Writes Axona format data to the given filename

        Parameters
        ----------
        filename_root : str
            The fully qualified filename including the suffix

        data : ndarray
            The data that will be saved
        """
        fType = os.path.splitext(filename_root)[1]
        if fType in self.axona_files:
            f = open(filename_root, "rb+")
            d = f.read()
            st = d.find(b"data_start") + len("data_start")
            f.seek(st)
            data.tofile(f)
            f.close()
            f = open(filename_root, "a")
            f.write("\r\n")
            f.write("data_end")
            f.write("\r\n")
            f.close()

    def getHeader(self, filename_root: str) -> dict:
        """
        Reads and returns the header of a specified data file as a dictionary

        Parameters
        ----------
        filename_root : str
            Fully qualified filename of Axona type

        Returns
        -------
        headerDict : dict
            key - value pairs of the header part of an Axona type file
        """
        with open(filename_root, "rb") as f:
            data = f.read()
            f.close()
        if os.path.splitext(filename_root)[1] != ".set":
            st = data.find(b"data_start") + len("data_start")
            header = data[0: st - len("data_start") - 2]
        else:
            header = data
        headerDict = {}
        lines = header.splitlines()
        for line in lines:
            line = str(line.decode("ISO-8859-1")).rstrip()
            line = line.split(" ", 1)
            try:
                headerDict[line[0]] = line[1]
            except IndexError:
                headerDict[line[0]] = ""
        return headerDict

    def getHeaderVal(self, header: dict, key: str) -> int:
        """
        Get a value from the header as an int

        Parameters
        ----------
        header : dict
            The header dictionary to read
        key : str
            The key to look up

        Returns
        -------
        value : int
            The value of `key` as an int
        """
        tmp = header[key]
        val = tmp.split(" ")
        val = val[0].split(".")
        val = int(val[0])
        return val


class Pos(IO):
    """
    Processs position data recorded with the Axona recording system

    Parameters
    ----------
    filename_root : str
        The basename of the file i.e mytrial as opposed to mytrial.pos

    Notes
    -----
    Currently the only arg that does anything is 'cm' which will convert
    the xy data to cm, assuming that the pixels per metre value has been
    set correctly
    """

    def __init__(self, filename_root: Path, *args, **kwargs):
        filename_root = Path(filename_root)
        if filename_root.suffix == ".set":
            filename_root = Path(os.path.splitext(filename_root)[0])
        self.filename_root = filename_root
        self.header = self.getHeader(filename_root.with_suffix(".pos"))
        self.setheader = None
        self.setheader = self.getHeader(filename_root.with_suffix(".set"))
        self.posProcessed = False
        posData = self.getData(filename_root.with_suffix(".pos"))
        self.nLEDs = 1
        if self.setheader is not None:
            self.nLEDs = sum(
                [
                    self.getHeaderVal(self.setheader, "colactive_1"),
                    self.getHeaderVal(self.setheader, "colactive_2"),
                ]
            )
        if self.nLEDs == 1:
            self.led_pos = np.ma.MaskedArray(posData["pos"][:, 0:2])
            self.led_pix = np.ma.MaskedArray([posData["pos"][:, 4]])
        if self.nLEDs == 2:
            self.led_pos = np.ma.MaskedArray(posData["pos"][:, 0:4])
            self.led_pix = np.ma.MaskedArray(posData["pos"][:, 4:6])
        self.led_pos = np.ma.masked_equal(self.led_pos, 1023)
        self.led_pix = np.ma.masked_equal(self.led_pix, 1023)
        self.ts = np.array(posData["ts"])
        self.npos = len(self.led_pos[0])
        self.xy = np.ones([2, self.npos]) * np.nan
        self.dir = np.ones([self.npos]) * np.nan
        self.dir_disp = np.ones([self.npos]) * np.nan
        self.speed = np.ones([self.npos]) * np.nan
        self.pos_sample_rate = self.getHeaderVal(self.header, "sample_rate")
        self._ppm = None
        if "cm" in kwargs:
            self.cm = kwargs["cm"]
        else:
            self.cm = False

    @property
    def ppm(self):
        if self._ppm is None:
            self._ppm = self.getHeaderVal(self.header, "pixels_per_metre")
        return self._ppm

    @ppm.setter
    def ppm(self, value):
        self._ppm = value


class Tetrode(IO):
    """
    Processes tetrode files recorded with the Axona recording system

    Mostly this class deals with interpolating tetrode and position timestamps
    and getting indices for particular clusters.

    Parameters
    ---------
    filename_root : str
        The fully qualified name of the file without it's suffix
    tetrode : int
        The number of the tetrode
    volts : bool, optional
        Whether to convert the data values volts. Default True
    """

    def __init__(self, filename_root: Path, tetrode, volts=True):
        filename_root = Path(filename_root)
        if filename_root.suffix == ".set":
            filename_root = Path(os.path.splitext(filename_root)[0])
        self.filename_root = filename_root
        self.tetrode = tetrode
        self.volts = volts
        self.header = self.getHeader(
            self.filename_root.with_suffix("." + str(tetrode)))
        data = self.getData(filename_root.with_suffix("." + str(tetrode)))
        self.spk_ts = data["ts"][::4]
        self.nChans = self.getHeaderVal(self.header, "num_chans")
        self.samples = self.getHeaderVal(self.header, "samples_per_spike")
        self.nSpikes = self.getHeaderVal(self.header, "num_spikes")
        self.posSampleRate = self.getHeaderVal(
            self.getHeader(
                self.filename_root.with_suffix(".pos")), "sample_rate"
        )
        self.waveforms = data["waveform"].reshape(
            self.nSpikes, self.nChans, self.samples
        )
        del data
        if volts:
            set_header = self.getHeader(self.filename_root.with_suffix(".set"))
            gains = np.zeros(4)
            st = (tetrode - 1) * 4
            for i, g in enumerate(np.arange(st, st + 4)):
                gains[i] = int(set_header["gain_ch_" + str(g)])
            ADC_mv = int(set_header["ADC_fullscale_mv"])
            scaling = (ADC_mv / 1000.0) / gains
            self.scaling = scaling
            self.gains = gains
            self.waveforms = (self.waveforms / 128.0) * scaling[:, np.newaxis]
        self.timebase = self.getHeaderVal(self.header, "timebase")
        cut = np.array(self.getCut(self.tetrode), dtype=int)
        self.cut = cut
        self.clusters = np.unique(self.cut)
        self.pos_samples = None

    def getSpkTS(self):
        """
        Return all the timestamps for all the spikes on the tetrode
        """
        return np.ma.compressed(self.spk_ts)

    def getClustTS(self, cluster: int = None):
        """
        Returns the timestamps for a cluster on the tetrode

        Parameters
        ----------
        cluster : int
            The cluster whose timestamps we want

        Returns
        -------
        clustTS : ndarray
            The timestamps

        Notes
        -----
        If None is supplied as input then all timestamps for all clusters
        is returned i.e. getSpkTS() is called
        """
        clustTS = None
        if cluster is None:
            clustTS = self.getSpkTS()
        else:
            if self.cut is None:
                cut = np.array(self.getCut(self.tetrode), dtype=int)
                self.cut = cut
            if self.cut is not None:
                clustTS = np.ma.compressed(self.spk_ts[self.cut == cluster])
        return clustTS

    def getPosSamples(self):
        """
        Returns the pos samples at which the spikes were captured
        """
        self.pos_samples = np.floor(
            self.getSpkTS() / float(self.timebase) * self.posSampleRate
        ).astype(int)
        return np.ma.compressed(self.pos_samples)

    def getClustSpks(self, cluster: int):
        """
        Returns the waveforms of `cluster`

        Parameters
        ----------
        cluster : int
            The cluster whose waveforms we want

        Returns
        -------
        waveforms : ndarray
            The waveforms on all 4 electrodes of the tgtrode so the shape of
            the returned array is [nClusterSpikes, 4, 50]
        """
        if self.cut is None:
            self.getClustTS(cluster)
        return self.waveforms[self.cut == cluster, :, :]

    def getClustIdx(self, cluster: int):
        """
        Get the indices of the position samples corresponding to the cluster

        Parameters
        ----------
        cluster : int
            The cluster whose position indices we want

        Returns
        -------
        pos_samples : ndarray
            The indices of the position samples, dtype is int
        """
        if self.cut is None:
            cut = np.array(self.getCut(self.tetrode), dtype=int)
            self.cut = cut
            if self.cut is None:
                return None
        if self.pos_samples is None:
            self.getPosSamples()  # sets self.pos_samples
        return self.pos_samples[self.cut == cluster].astype(int)

    def getUniqueClusters(self):
        """
        Returns the unique clusters
        """
        if self.cut is None:
            cut = np.array(self.getCut(self.tetrode), dtype=int)
            self.cut = cut
        return np.unique(self.cut)


class EEG(IO):
    """
    Processes eeg data collected with the Axona recording system

    Parameters
    ---------
    filename_root : str
        The fully qualified filename without the suffix
    egf: int
        Whether to read the 'eeg' file or the 'egf' file. 0 is False, 1 is True
    eeg_file: int
        If more than one eeg channel was recorded from then they are numbered
        from 1 onwards i.e. trial.eeg, trial.eeg1, trial.eeg2 etc
        This number specifies that

    """

    def __init__(self, filename_root: Path, eeg_file=1, egf=0):
        self.showfigs = 0
        filename_root = Path(os.path.splitext(filename_root)[0])
        self.filename_root = filename_root
        if egf == 0:
            if eeg_file == 1:
                eeg_suffix = ".eeg"
            else:
                eeg_suffix = ".eeg" + str(eeg_file)
        elif egf == 1:
            if eeg_file == 1:
                eeg_suffix = ".egf"
            else:
                eeg_suffix = ".egf" + str(eeg_file)
        self.header = self.getHeader(
            self.filename_root.with_suffix(eeg_suffix))
        self.eeg = self.getData(filename_root.with_suffix(eeg_suffix))["eeg"]
        # sometimes the eeg record is longer than reported in
        # the 'num_EEG_samples'
        # value of the header so eeg record should be truncated
        # to match 'num_EEG_samples'
        # TODO: this could be taken care of in the IO base class
        if egf:
            self.eeg = self.eeg[0: int(self.header["num_EGF_samples"])]
        else:
            self.eeg = self.eeg[0: int(self.header["num_EEG_samples"])]
        self.sample_rate = int(self.getHeaderVal(self.header, "sample_rate"))
        set_header = self.getHeader(self.filename_root.with_suffix(".set"))
        eeg_ch = int(set_header["EEG_ch_1"]) - 1
        eeg_gain = int(set_header["gain_ch_" + str(eeg_ch)])
        # EEG polarity is determined by the "mode_ch_n" key in the setfile
        # where n is the channel # for the eeg. The possibles values to these
        # keys are as follows:
        # 0 = Signal
        # 1 = Ref
        # 2 = -Signal
        # 3 = -Ref
        # 4 = Sig-Ref
        # 5 = Ref-Sig
        # 6 = grounded
        # So if the EEG has been recorded with -Signal (2) then the recorded
        # polarity is inverted with respect to that in the brain
        eeg_mode = int(set_header["mode_ch_" + set_header["EEG_ch_1"]])
        polarity = 1  # ensure it always has a value
        if eeg_mode == 2:
            polarity = -1
        ADC_mv = float(set_header["ADC_fullscale_mv"])
        scaling = (ADC_mv / 1000.0) * eeg_gain
        self.scaling = scaling
        self.gain = eeg_gain
        self.polarity = polarity
        denom = 128.0
        self.sig = (self.eeg / denom) * scaling * polarity  # eeg in microvolts
        self.EEGphase = None
        # x1 / x2 are the lower and upper limits of the eeg filter
        self.x1 = 6
        self.x2 = 12


class Stim(dict, IO):
    """
    Processes the stimulation data recorded using Axona

    Parameters
    ----------
    filename_root : str
        The fully qualified filename without the suffix
    """

    def __init__(self, filename_root: Path, *args, **kwargs):
        self.update(*args, **kwargs)
        filename_root = Path(os.path.splitext(filename_root)[0])
        self.filename_root = filename_root
        stmData = self.getData(filename_root.with_suffix(".stm"))
        self.__setitem__("ttl_timestamps", stmData["ts"])  # already in ms
        stmHdr = self.getHeader(filename_root.with_suffix(".stm"))
        for k, v in stmHdr.items():
            self.__setitem__(k, v)
        tb = int(self["timebase"].split(" ")[0])
        self.timebase = tb
        # the 'duration' value in the header of the .stm file
        # is not correct so we need to read this from the .set
        # file and update
        setHdr = self.getHeader(filename_root.with_suffix(".set"))
        stim_duration = [setHdr[k] for k in setHdr.keys() if 'stim_pwidth' in k][0]
        stim_duration = int(stim_duration)
        stim_duration = stim_duration / self.timebase  # in ms now
        self.__setitem__('stim_duration', stim_duration)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def __getitem__(self, key):
        val = dict.__getitem__(self, key)
        return val

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)


class ClusterSession(object):
    '''
    Loads all the cut file data and timestamps from the data 
    associated with the *.set filename given to __init__

    Meant to be a method-replica of the KiloSortSession class
    but really both should inherit from the same meta-class
    '''
    def __init__(self, fname_root) -> None:
        fname_root = Path(fname_root)
        assert fname_root.suffix == ".set"
        assert fname_root.exists()
        self.fname_root = fname_root

        self.cluster_id = None
        self.spk_clusters = None
        self.spk_times = None
        self.good_clusters = {}

    def load(self):
        pname = self.fname_root.parent
        pattern = re.compile(r'^'+str(
            self.fname_root.with_suffix(""))+r'_[1-9][0-9].cut')
        pattern1 = re.compile(r'^'+str(
            self.fname_root.with_suffix(""))+r'_[1-9]$.cut')
        cut_files = sorted(list(f for f in pname.iterdir()
                           if pattern.search(str(f)) or
                           pattern1.search(str(f))))
        # extract the clusters from each cut file
        # get the corresponding tetrode files
        tet_files = [str(c.with_suffix("")) for c in cut_files]
        tet_files = [t[::-1].replace("_", ".", 1)[::-1] for t in tet_files]
        tetrode_clusters = {}
        for fname in tet_files:
            T = IO(fname)
            idx = fname.rfind(".")
            tetnum = int(fname[idx+1:])
            cut = T.getCut(tetnum)
            tetrode_clusters[tetnum] = cut

        self.good_clusters = tetrode_clusters