import numpy as np
import matplotlib
import matplotlib.pylab as plt
from ephysiopy.common import ephys_generic

class CosineDirectionalTuning(object):
    """
    Produces output to do with Welday et al (2011) like analysis
    of rhythmic firing a la oscialltory interference model
    """

    def __init__(self, spike_times: np.array, pos_times: np.array, spk_clusters: np.array, pos_xy: np.array):
        """
        All timestamps should be given in sub-millisecond accurate seconds
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
        self.xy = pos_xy
        self._hdir = None
        self._pos_sample_rate = 30
        self._spk_sample_rate = 3e4
        self._pos_samples_for_spike = None
        self.spikeCalcs = ephys_generic.SpikeCalcsGeneric(spike_times)
        
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
    def pos_samples_for_spike(self):
        return self._pos_samples_for_spike

    @pos_samples_for_spike.setter
    def pos_samples_for_spike(self, value):
        self._pos_samples_for_spike = value
    
    def getPosIndices(self):
        self.pos_samples_for_spike = np.floor(self.spike_times * self.pos_sample_rate).astype(int)
    
    def getClusterPosIndices(self, cluster: int)->np.array:
        if self.pos_samples_for_spike is None:
            self.getPosIndices()
        cluster_pos_indices = self.pos_samples_for_spike[self.spk_clusters==cluster]        
        return cluster_pos_indices[idx_to_keep]
    
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
        if (self.hdir is None):
            import math
            pos2 = np.arange(0, len(self.pos_times)-1)
            xy_f = self.xy.astype(np.float)
            self.hdir = np.zeros_like(self.pos_times)
            self.hdir[pos2] = np.mod(((180/math.pi) * (np.arctan2(-xy_f[pos2+1,1] + xy_f[pos2,1],+xy_f[pos2+1,0]-xy_f[pos2,0]))), 360)
            self.hdir[-1] = self.hdir[-2]
        bins = np.arange(0, 360, binwidth)
        return np.digitize(self.hdir, bins)

    def getDirectionalBinForCluster(self, cluster: int):
        b = self.getDirectionalBinPerPosition(45)
        cluster_pos = self.getClusterPosIndices(cluster)
        return b[cluster_pos]

    