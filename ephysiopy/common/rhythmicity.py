import numpy as np
import matplotlib
import matplotlib.pylab as plt
from ephysiopy.common.ephys_generic import PosCalcsGeneric, SpikeCalcsGeneric

class CosineDirectionalTuning(object):
    """
    Produces output to do with Welday et al (2011) like analysis
    of rhythmic firing a la oscialltory interference model
    """

    @staticmethod
    def bisection(array,value):
        '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
        and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
        to indicate that ``value`` is out of range below and above respectively.
        
        NB From SO:
        https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
        '''
        n = len(array)
        if (value < array[0]):
            return -1
        elif (value > array[n-1]):
            return n
        jl = 0# Initialize lower
        ju = n-1# and upper limits.
        while (ju-jl > 1):# If we are not yet done,
            jm=(ju+jl) >> 1# compute a midpoint with a bitshift
            if (value >= array[jm]):
                jl=jm# and replace either the lower limit
            else:
                ju=jm# or the upper limit, as appropriate.
            # Repeat until the test condition is satisfied.
        if (value == array[0]):# edge cases at bottom
            return 0
        elif (value == array[n-1]):# and top
            return n-1
        else:
            return jl

    def __init__(self, spike_times: np.array, pos_times: np.array, spk_clusters: np.array, x: np.array, y: np.array, tracker_params: dict):
        """
        Parameters
        ----------
        spike_times - 1d np.array
        pos_times - 1d np.array
        spk_clusters - 1d np.array
        pos_xy - 1d np.array
        tracker_params - dict - from the PosTracker as created in OEKiloPhy.Settings.parsePos

        NB All timestamps should be given in sub-millisecond accurate seconds and pos_xy in cms
        """
        self.spike_times = spike_times
        self.pos_times = pos_times
        self.spk_clusters = spk_clusters
        '''
        There can be more spikes than pos samples in terms of sampling as the
        open-ephys buffer probably needs to finish writing and the camera has
        already stopped, so cut of any cluster indices and spike times
        that exceed the length of the pos indices
        '''
        idx_to_keep = self.spike_times < self.pos_times[-1]
        self.spike_times = self.spike_times[idx_to_keep]
        self.spk_clusters = self.spk_clusters[idx_to_keep]
        self._pos_sample_rate = 30
        self._spk_sample_rate = 3e4
        self._pos_samples_for_spike = None
        self._min_runlength = 0.4 # in seconds
        self.posCalcs = PosCalcsGeneric(x, y, 230, cm=True, jumpmax=100)
        self.spikeCalcs = SpikeCalcsGeneric(spike_times)
        xy, hdir = self.posCalcs.postprocesspos(tracker_params)
        self.posCalcs.calcSpeed(xy)
        self._xy = xy
        self._hdir = hdir
        self._speed = self.posCalcs.speed
        
    @property
    def spk_sample_rate(self):
        return self._spk_sample_rate

    @spk_sample_rate.setter
    def spk_sample_rate(self, value):
        self._spk_sample_rate = value
    
    @property
    def pos_sample_rate(self):
        return self._pos_sample_rate
    
    @pos_sample_rate.setter
    def pos_sample_rate(self, value):
        self._pos_sample_rate = value

    @property
    def min_runlength(self):
        return self._min_runlength
    
    @min_runlength.setter
    def min_runlength(self, value):
        self._min_runlength = value

    @property
    def xy(self):
        return self._xy

    @xy.setter
    def xy(self, value):
        self._xy = value

    @property
    def hdir(self):
        return self._hdir

    @hdir.setter
    def hdir(self, value):
        self._hdir = value

    @property
    def speed(self):
        return self._speed
    
    @speed.setter
    def speed(self, value):
        self._speed = value

    @property
    def pos_samples_for_spike(self):
        return self._pos_samples_for_spike

    @pos_samples_for_spike.setter
    def pos_samples_for_spike(self, value):
        self._pos_samples_for_spike = value

    def _rolling_window(self, a: np.array, window: int):
            """
            Totally nabbed from SO:
            https://stackoverflow.com/questions/6811183/rolling-window-for-1d-arrays-in-numpy
            """
            shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
            strides = a.strides + (a.strides[-1],)
            return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    
    def getPosIndices(self):
        self.pos_samples_for_spike = np.floor(self.spike_times * self.pos_sample_rate).astype(int)
    
    def getClusterPosIndices(self, cluster: int)->np.array:
        if self.pos_samples_for_spike is None:
            self.getPosIndices()
        cluster_pos_indices = self.pos_samples_for_spike[self.spk_clusters==cluster]
        return cluster_pos_indices
    
    def getDirectionalBinPerPosition(self, binwidth: int):
        """
        Direction is in degrees as that what is created by me in some of the
        other bits of this package.

        Parameters
        ----------
        binwidth : int - binsizethe bin width in degrees

        Outputs
        -------
        A digitization of which directional bin each position sample belongs to
        """
        
        bins = np.arange(0, 360, binwidth)
        return np.digitize(self.hdir, bins)

    def getDirectionalBinForCluster(self, cluster: int):
        b = self.getDirectionalBinPerPosition(45)
        cluster_pos = self.getClusterPosIndices(cluster)
        return b[cluster_pos]

    def getRunsOfMinLength(self):
        """
        Identifies runs of at least self.min_runlength seconds long, which at 30Hz pos
        sampling rate equals 12 samples, and returns the start and end indices at which
        the run was occurred and the directional bin that run belongs to

        Returns
        -------
        np.array - the start and end indices into position samples of the run and the
                    directional bin to which it belongs
        """

        b = self.getDirectionalBinPerPosition(45)
        # nabbed from SO
        from itertools import groupby
        grouped_runs = [(k,sum(1 for i in g)) for k,g in groupby(b)]
        grouped_runs_array = np.array(grouped_runs)
        run_start_indices = np.cumsum(grouped_runs_array[:,1]) - grouped_runs_array[:,1]
        minlength_in_samples = int(self.pos_sample_rate * self.min_runlength)
        runs_at_least_minlength_to_keep_mask = grouped_runs_array[:, 1] >= minlength_in_samples
        ret = np.array([run_start_indices[runs_at_least_minlength_to_keep_mask], grouped_runs_array[runs_at_least_minlength_to_keep_mask,1]]).T
        ret = np.insert(ret, 1, np.sum(ret, 1), 1) # contains run length as last column
        ret = np.insert(ret, 2, grouped_runs_array[runs_at_least_minlength_to_keep_mask, 0], 1)
        return ret[:,0:3]

    def speedFilterRuns(self, runs: np.array, minspeed=5.0):
        """
        Given the runs identified in getRunsOfMinLength, filter for speed and return runs
        that meet the min speed criteria

        The function goes over the runs with a moving window of length equal to self.min_runlength in samples
        and sees if any of those segments meets the speed criteria and splits them out into separate runs if
        true
        
        NB For now this means the same spikes might get included in the autocorrelation procedure later as the 
        moving window will use overlapping periods - can be modified later


        Parameters
        ----------
        runs - 3 x nRuns np.array generated from getRunsOfMinLength
        minspeed - float - min running speed in cm/s for an epoch (minimum epoch length defined previously
                            in getRunsOfMinLength as minlength, usually 0.4s)

        Returns
        -------
        3 x nRuns np.array - A modified version of the "runs" input variable
        """
        minlength_in_samples = int(self.pos_sample_rate * self.min_runlength)
        run_list = runs.tolist()
        all_speed = np.array(self.speed)
        for start_idx, end_idx, dir_bin in run_list:
            this_runs_speed = all_speed[start_idx:end_idx]
            this_runs_runs = self._rolling_window(this_runs_speed, minlength_in_samples)
            run_mask = np.all(this_runs_runs > minspeed, 1)
            if np.any(run_mask):
                print("got one")

    def testing(self, cluster: int):
        pos_idx = np.zeros_like(self.spike_times)
        for i, s in enumerate(self.spike_times):
            pos_idx[i] = self.bisection(self.pos_times, s)
        dir_bins = self.getDirectionalBinPerPosition(45)
        spike_dir_bins = dir_bins[pos_idx.astype(int)]

        cluster_bin_indices = spike_dir_bins[self.spk_clusters==cluster]
        cluster_spk_times = self.spike_times[self.spk_clusters==cluster]
        acorr_range = np.array([-500,500])
        acorrs = []
        for i in range(1,9):
            this_bin_indices = cluster_bin_indices == i
            ts = cluster_spk_times[this_bin_indices] # in seconds still so * 1000 for ms
            y = self.spikeCalcs.xcorr(ts*1000, Trange=acorr_range)
            acorrs.append(y)
        return acorrs

        