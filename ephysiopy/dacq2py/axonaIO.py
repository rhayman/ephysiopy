import scipy
import scipy.interpolate
import scipy.stats
import scipy.ndimage
import scipy.signal
import numpy as np
import math
import os
import pickle
import fnmatch
from ephysiopy.common.utils import smooth
from ephysiopy.dacq2py.spikecalcs import SpikeCalcs

MAXSPEED = 4.0  # pos data speed filter in m/s
BOXCAR = 20  # this gives a 400ms smoothing window for pos averaging

empty_headers = {
    "tetrode": os.path.join(os.path.dirname(__file__), "tetrode_header.pkl"),
    "pos": os.path.join(os.path.dirname(__file__), "pos_header.pkl"),
    "set": os.path.join(os.path.dirname(__file__), "set_header.pkl"),
    "eeg": os.path.join(os.path.dirname(__file__), "eeg_header.pkl"),
    "egf": os.path.join(os.path.dirname(__file__), "egf_header.pkl")
}


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
        [('ts', '>i'), ('waveform', '50b')])
    other_files = {'.pos': [('ts', '>i'), ('pos', '>8h')],
                   '.eeg': [('eeg', '=b')],
                   '.eeg2': [('eeg', '=b')],
                   '.egf': [('eeg', 'int16')],
                   '.egf2': [('eeg', 'int16')],
                   '.inp': [('ts', '>i4'), ('type', '>b'), ('value', '>2b')],
                   '.log': [('state', 'S3'), ('ts', '>i')],
                   '.stm': [('ts', '>i')]}

    # this only works in >= Python3.5
    axona_files = {**other_files, **tetrode_files}

    def __init__(self, filename_root=''):
        self.filename_root = filename_root

    @staticmethod
    def getEmptyHeader(ftype: str) -> dict:
        pname = empty_headers.get(ftype, '')
        if os.path.isfile(pname):
            with open(pname, 'rb') as f:
                return pickle.load(f)

    def getData(self, filename_root):
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
            try:
                header = self.getHeader(filename_root)
                for key in header.keys():
                    if len(fType) > 2:
                        if fnmatch.fnmatch(key, 'num_*_samples'):
                            n_samps = int(header[key])
                    else:
                        if key.startswith('num_spikes'):
                            n_samps = int(header[key]) * 4
            except IOError:
                print('File type not recognised')
            f = open(filename_root, 'rb')
            data = f.read()
            st = data.find(b'data_start') + len('data_start')
            f.seek(st)
            if fType == '.log':
                f.seek(0)
            dt = np.dtype(self.axona_files[fType])
            a = np.fromfile(f, dtype=dt, count=n_samps)
            f.close()
        return a

    def getCluCut(self, tet):
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
        filename_root = self.filename_root + '.' + 'clu.' + str(tet)
        dt = np.dtype([('data', '<i')])
        clu_data = np.loadtxt(filename_root, dtype=dt)
        return clu_data['data'][1::]  # first entry is number of clusters found

    def getCut(self, tet):
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
        filename_root = self.filename_root + '_' + str(tet) + '.cut'
        if not os.path.exists(filename_root):
            cut = self.getCluCut(tet)
            return cut - 1
        with open(filename_root, 'r') as f:
            cut_data = f.read()
            f.close()
        tmp = cut_data.split('spikes: ')
        tmp1 = tmp[1].split('\n')
        cut = tmp1[1:]
        for line in cut:
            m = line.split()
            for i in m:
                a.append(int(i))
        return a

    def setHeader(self, filename_root: str, header: dict):
        """
        Writes out the header to the specified file

        Parameters
        ------------
        filename_root : str
            A fully qualified path to a file with the relevant suffix at
            the end (e.g. ".set", ".pos" or whatever)

        header : dict
            An empty version of which can be loaded using getEmptyHeader()
            above
        """
        with open(filename_root, 'w') as f:
            for key, val in header.items():
                f.write(key)
                f.write(" ")
                if val is None:
                    val = ""
                f.write(val)
                f.write('\r\n')
            f.write('data_start')
            f.write('\r\n')
            f.write('data_end')
            f.write('\r\n')

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
            f = open(filename_root, 'rb+')
            d = f.read()
            st = d.find(b'data_start') + len('data_start')
            f.seek(st)
            data.tofile(f)
            f.close()
            f = open(filename_root, 'a')
            f.write('\r\n')
            f.write('data_end')
            f.write('\r\n')
            f.close()

    def getHeader(self, filename_root):
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
        with open(filename_root, 'rb') as f:
            data = f.read()
            f.close()
        if os.path.splitext(filename_root)[1] != '.set':
            st = data.find(b'data_start') + len('data_start')
            header = data[0:st-len('data_start')-2]
        else:
            header = data
        headerDict = {}
        lines = header.splitlines()
        for line in lines:
            line = str(line.decode("ISO-8859-1")).rstrip()
            line = line.split(' ', 1)
            try:
                headerDict[line[0]] = line[1]
            except IndexError:
                headerDict[line[0]] = ''
        return headerDict

    def getHeaderVal(self, header, key):
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
        val = tmp.split(' ')
        val = val[0].split('.')
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

    def __init__(self, filename_root, *args, **kwargs):
        self.filename_root = filename_root
        self.header = self.getHeader(filename_root + '.pos')
        self.setheader = None
        try:
            self.setheader = self.getHeader(filename_root + '.set')
        except Exception:
            pass
        self.posProcessed = False
        posData = self.getData(filename_root + '.pos')
        self.nLEDs = 1
        if self.setheader is not None:
            self.nLEDs = sum([self.getHeaderVal(
                self.setheader, 'colactive_1'),
                self.getHeaderVal(self.setheader, 'colactive_2')])
        if self.nLEDs == 1:
            self.led_pos = np.ma.masked_values(
                [posData['pos'][:, 0], posData['pos'][:, 1]], 1023)
            self.led_pix = np.ma.masked_values(
                posData['pos'][:, 4], 1023)
        elif self.nLEDs == 2:
            self.led_pos = np.ma.masked_values(
                [posData['pos'][:, 0], posData['pos'][:, 1],
                    posData['pos'][:, 2], posData['pos'][:, 3]], 1023)
            self.led_pix = np.ma.masked_values(
                [posData['pos'][:, 4], posData['pos'][:, 5]], 1023)
        self.npos = len(self.led_pos[0])
        self.xy = np.ones([2, self.npos]) * np.nan
        self.dir = np.ones([self.npos]) * np.nan
        self.dir_disp = np.ones([self.npos]) * np.nan
        self.speed = np.ones([self.npos]) * np.nan
        self.pos_sample_rate = self.getHeaderVal(self.header, 'sample_rate')
        self._ppm = None
        if 'cm' in kwargs:
            self.cm = kwargs['cm']
        else:
            self.cm = False

    @property
    def ppm(self):
        if self._ppm is None:
            try:
                self._ppm = self.getHeaderVal(self.header, 'pixels_per_metre')
            except IOError:
                self._ppm = None
        return self._ppm

    @ppm.setter
    def ppm(self, value):
        self._ppm = value
        self.posProcessed = False
        self.postprocesspos()

    def __getitem__(self, key):
        try:
            val = self.__dict__[key]
            return val
        except Exception:
            pass

    def postprocesspos(self):
        """
        Post processes position data

        Calls a few of the other methods in this class that deal with
        led swaps, impossibly fast points etc
        """

        if self.posProcessed is True:
            return
        elif self.posProcessed is False:
            led_pos = self.led_pos
            led_pix = self.led_pix
            # as with AJ's implementation in mtint, calculate weights for a
            # weighted mean
            # of the front and back leds for when trials have poorly tracked
            #  trials
            # NB could probably do this straight from the number of leds
            #  tracked which
            # is available in the raw data
            # need to mask all values of the array using logical
            #  or to replicate
            # mtints way of filling in missing values
            nLED_idx = self.nLEDs * 2
            led_pos[0:nLED_idx].__setmask__(
                led_pos[0:nLED_idx].mask.any(axis=0))
            weights = np.zeros(2)
            weights[0] = float(np.sum(np.nonzero(
                led_pos[0:2]), axis=1)[0]) / self.npos
            try:
                weights[1] = float(np.sum(np.nonzero(
                    led_pos[2:4]), axis=1)[0]) / self.npos
            except IndexError:
                pass
            # need to deal with improperly tracked positions where the values
            # are plainly ridiculous
            # values less than 0 are masked
            led_pos[led_pos < 0] = np.ma.masked
            # deal with values outside the range of the tracked window
            led_pos[0, (led_pos[0] > int(self.header['max_x']))] = np.ma.masked
            led_pos[1, (led_pos[1] > int(self.header['max_y']))] = np.ma.masked
            # try and deal with other led if present
            try:
                led_pos[2, (led_pos[2] > int(
                    self.header['max_x']))] = np.ma.masked
                led_pos[3, (led_pos[3] > int(
                    self.header['max_y']))] = np.ma.masked
            except IndexError:
                pass
            if np.logical_and(np.any(np.nonzero(led_pix)), self.nLEDs == 2):
                swap_list = self.ledswapFilter(led_pos, led_pix)
                tmp = led_pos[0:2, swap_list]
                led_pos[0:2, swap_list] = led_pos[2:4, swap_list]
                led_pos[2:4, swap_list] = tmp
                tmp = led_pix[0, swap_list]
                led_pix[0, swap_list] = led_pix[1, swap_list]
                led_pix[1, swap_list] = tmp
            ppm = self.ppm
            max_ppm_per_sample = MAXSPEED * ppm / self.pos_sample_rate
            led_pos = self.ledspeedFilter(led_pos, max_ppm_per_sample)
            led_pos = self.interpNans(led_pos)
            # get distances and angles of LEDs from rat
            pos1 = np.arange(0, self.npos)
            pos2 = np.arange(0, self.npos-1)
            if self.nLEDs == 1:
                self.xy[0:2, pos1] = led_pos[0:2, pos1]
                self.xy[0, :] = smooth(self.xy[0, :], BOXCAR, 'flat')
                self.xy[1, :] = smooth(self.xy[1, :], BOXCAR, 'flat')
                self.dir[pos2] = np.mod(((180/math.pi) * (np.arctan2(
                    -self.xy[1, pos2+1] + self.xy[1, pos2], +self.xy[0, pos2+1]
                    - self.xy[0, pos2]))), 360)
                self.dir[-1] = self.dir[-2]
                self.dir_disp = self.dir
            elif self.nLEDs == 2:
                lightBearings = np.zeros([2, 1])
                lightBearings[0] = self.getHeaderVal(
                    self.setheader, 'lightBearing_1')
                lightBearings[1] = self.getHeaderVal(
                    self.setheader, 'lightBearing_2')
                front_back_xy_sm = np.zeros([4, self.npos])
                for i in range(len(front_back_xy_sm)):
                    front_back_xy_sm[i, pos1] = smooth(
                        led_pos[i, pos1], BOXCAR, 'flat')
                correction = lightBearings[0]
                self.dir[pos1] = np.mod((180/math.pi) * (np.arctan2(
                    -front_back_xy_sm[1, pos1]+front_back_xy_sm[3, pos1],
                    +front_back_xy_sm[0, pos1]-front_back_xy_sm[2, pos1]) -
                    correction), 360)
                # get xy from smoothed individual lights weighting
                # for reliability
                self.xy[0, pos1] = (weights[0]*front_back_xy_sm[0, pos1]
                                    + weights[1]*front_back_xy_sm[2, pos1]) / \
                    np.sum(weights)
                self.xy[1, pos1] = (weights[0]*front_back_xy_sm[1, pos1]
                                    + weights[1]*front_back_xy_sm[3, pos1]) / \
                    np.sum(weights)
                self.dir_disp[pos2] = np.mod(((180/math.pi) * (np.arctan2(
                    -self.xy[1, pos2+1] + self.xy[1, pos2], +self.xy[0, pos2+1]
                    - self.xy[0, pos2]))), 360)
                self.dir_disp[-1] = self.dir_disp[-2]

            if self.cm:
                self.xy = self.xy / ppm * 100  # xy now in cm
            # calculate speed based on distance
            self.speed[pos2] = np.sqrt(np.sum(
                np.power(np.diff(self.xy), 2), 0))
            self.speed[self.npos-1] = self.speed[-1]
            # Get speed into cm/s
            self.speed = self.speed * (100 * self.pos_sample_rate / ppm)
            if np.isnan(self.speed[-1]):
                self.speed[-1] = 0

            self.posProcessed = True

    def ledspeedFilter(self, led_pos, max_ppm_per_sample):
        """
        Filters for impossibly fast tracked points

        Parameters
        ----------
        led_pos : np.ma.MaskedArray
            Masked led_pos array [x1,y1,x2,y2]
        max_ppm_per_sample : int
            The maximum distance (in pixels) that points are allowed to jump

        Returns
        -------
        led_pos : np.ma.MaskedArray
            The filtered data
        """
        max_ppms_sqd = max_ppm_per_sample ** 2
        for i in range(0, len(led_pos), 2):
            ok_pos = led_pos[i, :]
            prev_pos = ok_pos[0:-1]
            cur_pos = ok_pos[1:]
            pix_per_sample_sqd = (np.power((np.subtract(
                led_pos[i, cur_pos], led_pos[i, prev_pos])), 2) + np.power(
                    (np.subtract(led_pos[i+1, cur_pos], led_pos[
                        i+1, prev_pos])), 2)) / np.power(np.subtract(
                            cur_pos, prev_pos), 2)
            pix_per_sample_sqd = np.insert(pix_per_sample_sqd, -1, 0)
            led_pos[i:i+2, pix_per_sample_sqd > max_ppms_sqd] = np.ma.masked
        return led_pos

    def ledswapFilter(self, led_pos, led_pix):
        """
        Checks for led swapping in 2-spot mode

        Parameters
        ----------
        led_pos : np.ma.MaskedArray
            Masked array of dims [4 x nPosSamples]
        led_pix : np.ma.MaskedArray
            The number of pixels tracked for each LED.
            Has dimensions [2 x nPosSamples]

        Returns
        -------
        swap_list : ndarray
            Array of swapped positions
        """
        thresh = 5
        mean_npix = led_pix.mean(axis=1).data
        std_npix = led_pix.std(axis=1).data
        pos = np.arange(1, led_pix.shape[1])
        # Calculate distances
        dist12 = np.sqrt(np.nansum(((np.squeeze(
            led_pos[0:2, pos])-np.squeeze(
            led_pos[2:4, pos-1]))**2), axis=0))
        dist11 = np.sqrt(np.nansum(((np.squeeze(
            led_pos[0:2, pos])-np.squeeze(
            led_pos[0:2, pos-1]))**2), axis=0))
        dist21 = np.sqrt(np.nansum(((np.squeeze(
            led_pos[2:4, pos])-np.squeeze(
            led_pos[0:2, pos-1]))**2), axis=0))
        dist22 = np.sqrt(np.nansum(((np.squeeze(
            led_pos[2:4, pos])-np.squeeze(
            led_pos[2:4, pos-1]))**2), axis=0))
        switched = np.logical_or(np.logical_and(
            (dist12 < dist11 - thresh).data, led_pos[
                2, pos].mask), (dist21 < dist22-thresh).data)
        z11 = (mean_npix[0] - led_pix[0, pos]) / std_npix[0]
        z12 = (led_pix[0, pos] - mean_npix[1]) / std_npix[1]
        shrunk = z11 > z12
        swap_list = np.nonzero(np.logical_and(switched, shrunk.data))[0] + 1
        return swap_list

    def interpNans(self, led_pos):
        """
        Interpolates over missing values

        Parameters
        ----------
        led_pos : np.ma.MaskedArray
            The LED positions

        Returns
        -------
        output : np.ma.MaskedArray
            The smoothed, unmasked array
        """
        for i in range(0, len(led_pos), 2):
            missing = led_pos[i:i+2].mask.any(axis=0)
            ok = np.logical_not(missing)
            ok_idx = ok.ravel().nonzero()[0]
            missing_idx = missing.ravel().nonzero()[0]
            good_data = led_pos.data[i, ok_idx]
            good_data1 = led_pos.data[i+1, ok_idx]
            led_pos.data[i, missing_idx] = np.interp(
                missing_idx, ok_idx, good_data)
            led_pos.data[i+1, missing_idx] = np.interp(
                missing_idx, ok_idx, good_data1)
        # unmask the array
        led_pos.mask = 0
        return led_pos

    def filterPos(self, filterDict):
        """
        Filters position data depending on the specified filter

        Parameters
        ----------
        filterDict : dict
            Contains the type(s) of filter to be used and the
            range of values to filter for. Values are pairs specifying the
            range of values to filter for NB can take multiple filters and
             iteratively apply them
            legal values are:
            * 'dir' - the directional range to filter for NB this can
                contain 'w','e','s' or 'n'
            * 'speed' - min and max speed to filter for
            * 'xrange' - min and max values to filter x pos values
            * 'yrange' - same as xrange but for y pos
            * 'time' - the times to keep / remove specified in ms

        Returns
        --------
        pos_index_to_keep : ndarray
            The position indices that should be kept
        """
        if filterDict is None:
            return
        nSamples = int(self.header['num_pos_samples'])
        bool_arr = np.ones(shape=(len(filterDict), nSamples), dtype=np.bool)
        for idx, key in enumerate(filterDict):
            if isinstance(filterDict[key], str):
                if len(filterDict[key]) == 1 and 'dir' in key:
                    if 'w' in filterDict[key]:
                        filterDict[key] = (135, 225)
                    elif 'e' in filterDict[key]:
                        filterDict[key] = (315, 45)
                    elif 's' in filterDict[key]:
                        filterDict[key] = (225, 315)
                    elif 'n' in filterDict[key]:
                        filterDict[key] = (45, 135)
                else:
                    raise ValueError("filter must contain a key / value pair")
            if 'speed' in key:
                if filterDict[key][0] > filterDict[key][1]:
                    raise ValueError("First value must be less \
                        than the second one")
                else:
                    bool_arr[idx, :] = np.logical_and(
                        self.speed > filterDict[key][0],
                        self.speed < filterDict[key][1])
            elif 'dir' in key:
                if filterDict[key][0] < filterDict[key][1]:
                    bool_arr[idx, :] = np.logical_and(
                        self.dir > filterDict[key][0],
                        self.dir < filterDict[key][1])
                else:
                    bool_arr[idx, :] = np.logical_or(
                        self.dir > filterDict[key][0],
                        self.dir < filterDict[key][1])
            elif 'xrange' in key:
                bool_arr[idx, :] = np.logical_and(
                    self.xy[0, :] > filterDict[key][0],
                    self.xy[0, :] < filterDict[key][1])
            elif 'yrange' in key:
                bool_arr[idx, :] = np.logical_and(
                    self.xy[1, :] > filterDict[key][0],
                    self.xy[1, :] < filterDict[key][1])
            elif 'time' in key:
                # takes the form of 'from' - 'to' times in SECONDS
                # such that only pos's between these ranges are KEPT
                filterDict[key] = filterDict[key] * self.pos_sample_rate
                if filterDict[key].ndim == 1:
                    bool_arr[idx, filterDict[
                        key][0]:filterDict[key][1]] = False
                else:
                    for i in filterDict[key]:
                        bool_arr[idx, i[0]:i[1]] = False
                bool_arr = ~bool_arr
            else:
                print("Unrecognised key in dict")
                pass
        return np.expand_dims(np.any(~bool_arr, axis=0), 0)


class Tetrode(IO, SpikeCalcs):
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

    def __init__(self, filename_root, tetrode, volts=True):
        self.filename_root = filename_root
        self.tetrode = tetrode
        self.volts = volts
        self.header = self.getHeader(self.filename_root + '.' + str(tetrode))
        data = self.getData(filename_root + '.' + str(tetrode))
        self.spk_ts = data['ts'][::4]
        self.nChans = self.getHeaderVal(self.header, 'num_chans')
        self.samples = self.getHeaderVal(self.header, 'samples_per_spike')
        self.nSpikes = self.getHeaderVal(self.header, 'num_spikes')
        self.posSampleRate = self.getHeaderVal(
            self.getHeader(self.filename_root + '.' + 'pos'), 'sample_rate')
        self.waveforms = data['waveform'].reshape(
            self.nSpikes, self.nChans, self.samples)
        del data
        if volts:
            set_header = self.getHeader(self.filename_root + '.set')
            gains = np.zeros(4)
            st = (tetrode - 1) * 4
            for i, g in enumerate(np.arange(st, st+4)):
                gains[i] = int(set_header['gain_ch_' + str(g)])
            ADC_mv = int(set_header['ADC_fullscale_mv'])
            scaling = (ADC_mv/1000.) / gains
            self.scaling = scaling
            self.gains = gains
            self.waveforms = (self.waveforms / 128.) * scaling[:, np.newaxis]
        self.timebase = self.getHeaderVal(self.header, 'timebase')
        try:
            cut = np.array(self.getCut(self.tetrode), dtype=int)
            self.cut = cut
            self.clusters = np.unique(self.cut)
        except IOError:
            try:
                cut = self.getCluCut(self.tetrode)
                cut = np.array(cut) - 1
                self.cut = cut
                self.clusters = np.unique(self.cut)
            except IOError:
                self.cut = None
        self.pos_samples = None

    def getSpkTS(self):
        """
        Return all the timestamps for all the spikes on the tetrode
        """
        return np.ma.compressed(self.spk_ts)

    def getClustTS(self, cluster=None):
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
        if cluster is None:
            clustTS = self.getSpkTS()
        else:
            if self.cut is None:
                try:
                    cut = np.array(self.getCut(self.tetrode), dtype=int)
                except IOError:
                    cut = self.getCluCut(self.tetrode)
                    cut = np.array(cut) - 1
                self.cut = cut
            self.getSpkTS()
            clustTS = np.ma.compressed(self.spk_ts[self.cut == cluster])
        return clustTS

    def getPosSamples(self):
        """
        Returns the pos samples at which the spikes were captured
        """
        self.pos_samples = np.floor(
            self.getSpkTS() / float(
                self.timebase) * self.posSampleRate).astype(int)
        return np.ma.compressed(self.pos_samples)

    def getClustSpks(self, cluster):
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

    def getClustIdx(self, cluster):
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
            try:
                cut = np.array(self.getCut(self.tetrode), dtype=int)
            except IOError:
                cut = self.getCluCut(self.tetrode)
                cut = np.array(cut) - 1
            self.cut = cut
        if self.pos_samples is None:
            self.getPosSamples()
        return self.pos_samples[self.cut == cluster].astype(int)

    def getUniqueClusters(self):
        """
        Returns the unique clusters
        """
        if self.cut is None:
            try:
                cut = np.array(self.getCut(self.tetrode), dtype=int)
            except IOError:
                cut = self.getCluCut(self.tetrode)
                cut = np.array(cut) - 1
            self.cut = cut
        else:
            cut = self.cut
        return np.unique(cut)


class EEG(IO):
    """
    Processes eeg data collected with the Axona recording system

    Parameters
    ---------
    filename_root : str
        The fully qualified filename without the suffix
    eeg_file, egf : int
        Whether to read the 'eeg' file or the 'egf' file. 0 is False, 1 is True
    """

    def __init__(self, filename_root, eeg_file=1, egf=0):
        self.showfigs = 0
        self.filename_root = filename_root
        if egf == 0:
            denom = 128.0  # used below to normalise data
            if eeg_file == 1:
                eeg_suffix = '.eeg'
            else:
                eeg_suffix = '.eeg' + str(eeg_file)
        elif egf == 1:
            denom = 128.0  # used below to normalise data
            if eeg_file == 1:
                eeg_suffix = '.egf'
            else:
                eeg_suffix = '.egf' + str(eeg_file)
        self.header = self.getHeader(self.filename_root + eeg_suffix)
        self.eeg = self.getData(filename_root + eeg_suffix)['eeg']
        # sometimes the eeg record is longer than reported in
        # the 'num_EEG_samples'
        # value of the header so eeg record should be truncated
        # to match 'num_EEG_samples'
        # TODO: this could be taken care of in the IO base class
        if egf:
            self.eeg = self.eeg[0:int(self.header['num_EGF_samples'])]
        else:
            self.eeg = self.eeg[0:int(self.header['num_EEG_samples'])]
        if egf == 1:
            self.eeg = self.eeg[:, 1]
        self.sample_rate = int(self.getHeaderVal(self.header, 'sample_rate'))
        set_header = self.getHeader(self.filename_root + '.set')
        eeg_ch = int(set_header['EEG_ch_1']) - 1
        if eeg_ch < 0:
            eeg_ch = 0
        eeg_gain = int(set_header['gain_ch_' + str(eeg_ch)])
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
        eeg_mode = int(set_header['mode_ch_' + set_header['EEG_ch_1']])
        polarity = 1  # ensure it always has a value
        if eeg_mode == 2:
            polarity = -1
        ADC_mv = float(set_header['ADC_fullscale_mv'])
        scaling = (ADC_mv/1000.) * eeg_gain
        self.scaling = scaling
        self.gain = eeg_gain
        self.polarity = polarity
        self.eeg = (self.eeg / denom) * scaling * polarity  # eeg in microvolts
        self.EEGphase = None
        # x1 / x2 are the lower and upper limits of the eeg filter
        self.x1 = 6
        self.x2 = 12

    def eegfilter(self, E=None):
        """
        Filter the LFP signal

        Filters the eeg using a 251-tap bandpass (6-12Hz) blackman filter
        between the values given in x1 and x2: defaults to filtering between
        theta frequency (6-12Hz)

        Parameters
        ----------
        E : ndarray
            The LFP signal

        Returns
        -------
        filtEEG : ndarray
            The filtered LFP signal
        """
        if E is None:
            E = self.eeg
        nyquist = self.sample_rate / 2.
        eegfilter = scipy.signal.firwin(
            int(self.sample_rate) + 1, [self.x1/nyquist, self.x2/nyquist],
            window='black', pass_zero=False)
        filtEEG = scipy.signal.filtfilt(
            eegfilter, [1], E.ravel(), padtype='odd')
        if np.ma.is_masked(self.eeg):
            mask = np.ma.getmask(self.eeg)
            return np.ma.masked_where(mask, filtEEG)
        else:
            return filtEEG

    def thetaAmpPhase(self, fx=None):
        """
        Extracts the amplitude and instantaneous frequency of the LFP

        Parameters
        ----------
        fs : ndarray
            The filtered LFP signal, typically filtered using eegfilter()
        """
        if fx is None:
            fx = self.eegfilter(E=self.eeg)
        analytic = scipy.signal.hilbert(fx)
        self.EEGphase = np.angle(analytic)
        self.UWphase = np.unwrap(self.EEGphase)
        tmp = np.append(self.UWphase, np.nan)
        self.EEGinstfreq = np.diff(tmp) * (self.sample_rate / (2*np.pi))
        self.thAmp = np.abs(analytic)

    def nextpow2(self, val):
        """
        Calculates the next power of 2 that will hold val

        Parameters
        ----------
        val : int
            The value to find the next power of 2 of

        Returns
        -------
        out : ndarray
            The next power of 2 that contains val
        """
        val = val - 1
        val = (val >> 1) | val
        val = (val >> 2) | val
        val = (val >> 4) | val
        val = (val >> 8) | val
        val = (val >> 16) | val
        val = (val >> 32) | val
        return np.log2(val + 1)


class Stim(dict, IO):
    """
    Processes the stimulation data recorded using Axona

    Parameters
    ----------
    filename_root : str
        The fully qualified filename without the suffix
    """
    def __init__(self, filename_root, *args, **kwargs):
        self.update(*args, **kwargs)
        self.filename_root = filename_root
        stmData = self.getData(filename_root + '.stm')
        self.__setitem__('on', stmData['ts'])
        stmHdr = self.getHeader(filename_root + '.stm')
        for k, v in stmHdr.items():
            self.__setitem__(k, v)
        tb = int(self['timebase'].split(' ')[0])
        self.timebase = tb

    def update(self, *args, **kwargs):
        d = dict(*args, **kwargs)
        for k in d.keys():
            self[k] = d[k]

    def __getitem__(self, key):
        try:
            val = dict.__getitem__(self, key)
            return val
        except KeyError:
            print('KeyError')

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)

    def getTS(self):
        """
        Gets the timestamps of the on events
        """
        return self['on'] / int(self.timebase / 1000)  # in ms

    def getPosIdx(self):
        """
        Gets the position indices of the on events
        """
        scale = self.timebase / float(self['posSampRate'])
        return self['on'] / scale

    def getEEGIdx(self):
        """
        Gets the EEG indices of the on events
        """
        scale = self.timebase / float(self['eegSampRate'])
        return (self['on'] / scale).astype(int)

    def getEGFIdx(self):
        """
        Gets the EGF indices of the on events
        """
        scale = self.timebase / float(self['egfSampRate'])
        return (self['on'] / scale).astype(int)
