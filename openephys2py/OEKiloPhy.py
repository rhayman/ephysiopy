#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:53:10 2017

@author: robin
"""
import numpy as np
import matplotlib.pylab as plt

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from collections import OrderedDict

try:
    from .ephysiopy.openephys2py.OESettings import Settings
except ImportError:
    from ephysiopy.openephys2py.OESettings import Settings
'''
The results of a kilosort session are a load of .npy files, a .csv file
and some other stuff
The .npy files contain things like spike times, cluster ids etc. Importantly
the .csv file ('cluster_groups.csv') contains the results (more or less) of
the SAVED part of the phy template-gui (ie when you click "Save" from the
Clustering menu): this file consists of a header ('cluster_id' and 'group')
where 'cluster_id' is obvious (and relates to the identity in spk_clusters.npy),
the 'group' is a string that contains things like 'noise' or 'unsorted' or
presumably a number or quality measure as determined by the user
Load all these things to get a restricted list of things to look at...
'''
class KiloSortSession(object):
    '''
    Parameters
    ----------
    fname_root : str
        Should contain all the files from a kilosort session and
        the .dat file (extracted from the nwb OE session)

    '''
    def __init__(self, fname_root):
        self.fname_root = fname_root
        self.cluster_id = None
        self.spk_clusters = None
        self.spk_times = None

    def load(self):
        '''
        Load all the relevant files
        '''
        import os
        dtype = {'names': ('cluster_id', 'group'), 'formats': ('i4', 'S10')}
        if os.path.exists(os.path.join(self.fname_root, 'cluster_groups.csv')):
            self.cluster_id, self.group = np.loadtxt(os.path.join(self.fname_root, 'cluster_groups.csv'), unpack=True, skiprows=1, dtype=dtype)
        if os.path.exists(os.path.join(self.fname_root, 'cluster_group.tsv')):
            self.cluster_id, self.group = np.loadtxt(os.path.join(self.fname_root, 'cluster_group.tsv'), unpack=True, skiprows=1, dtype=dtype)
        self.spk_clusters = np.load(os.path.join(self.fname_root, 'spike_clusters.npy'))
        self.spk_times    = np.load(os.path.join(self.fname_root, 'spike_times.npy'))

    def removeNoiseClusters(self):
        '''
        Restricts analysis to anything that isn't listed as 'noise' in self.group
        '''
        if self.cluster_id is not None:
            self.good_clusters = []
            for id_group in zip(self.cluster_id, self.group):
                if 'noise' not in id_group[1].decode():
                    self.good_clusters.append(id_group[0])

class OpenEphysNWB(object):
    '''
    Parameters
    ------------
    fname_root:- str
        Should contain the settings.xml file and the .nwb file
    '''

    def __init__(self, fname_root, **kwargs):
        self.fname_root = fname_root # str
        self.kilodata = None # a KiloSortSession object - see class above
        self.nwbData = None # handle to the open nwb file (HDF5 file object)
        self.rawData = None # np.array holding the raw, continuous recording
        self.spikeData = None # a list of np.arrays, nominally containing tetrode data in format nspikes x 4 x 40
        self.accelerometerData = None # np.array
        self.timeAligned = False # deprecated
        self.mapiter = None # iterator plotting rate maps etc
        self.settings = None # OESettings.Settings instance
        self.recording_name = None # the recording name inside the nwb file ('recording0', 'recording1', etc)
        if ('jumpmax' in kwargs.keys()):
            self.jumpmax = kwargs['jumpmax']
        else:
            self.jumpmax = 100

    def load(self, session_name=None, recording_name=None, loadraw=False, loadspikes=False, savedat=False):
        '''
        Loads xy pos from binary part of the hdf5 file and data resulting from
        a Kilosort session (see KiloSortSession class above)

        Parameters
        ----------
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
        '''

        import h5py
        import os
        if session_name is None:
            session_name = 'experiment_1.nwb'
        self.nwbData = h5py.File(os.path.join(self.fname_root, session_name))
        # Position data...
        if self.recording_name is None:
            if recording_name is None:
                recording_name = 'recording0'
            self.recording_name = recording_name
        self.xy = np.array(self.nwbData['acquisition']['timeseries'][self.recording_name]['events']['binary1']['data'])

        self.xyTS = np.array(self.nwbData['acquisition']['timeseries'][self.recording_name]['events']['binary1']['timestamps'])
        self.xyTS = self.xyTS - (self.xy[:,2] / 1e6)
        self.xy = self.xy[:,0:2]
        # TTL data...
        self.ttl_data = np.array(self.nwbData['acquisition']['timeseries'][self.recording_name]['events']['ttl1']['data'])
        self.ttl_timestamps = np.array(self.nwbData['acquisition']['timeseries'][self.recording_name]['events']['ttl1']['timestamps'])

        # ...everything else
        try:
            settings = Settings(os.path.join(self.fname_root, 'settings.xml'))
            settings.load()
            settings.parse()
            self.settings = settings
            fpgaId = settings.fpga_nodeId
            fpgaNode = 'processor' + str(fpgaId) + '_' + str(fpgaId)
            self.ts = np.array(self.nwbData['acquisition']['timeseries'][self.recording_name]['continuous'][fpgaNode]['timestamps'])
            if (loadraw == True):
                self.rawData = np.array(self.nwbData['acquisition']['timeseries'][self.recording_name]['continuous'][fpgaNode]['data'])
                settings.parseChannels() # to get the neural data channels
                self.accelerometerData = self.rawData[:,64:]
                self.rawData = self.rawData[:,0:64]
                if (savedat == True):
                    data2save = self.rawData[:,0:64]
                    data2save.tofile(os.path.join(self.fname_root, 'experiment_1.dat'))
            if loadspikes == True:
                if self.nwbData['acquisition']['timeseries'][self.recording_name]['spikes']:
                    # Create a dictionary containing keys 'electrode1', 'electrode2' etc and None for values
                    electrode_dict = dict.fromkeys(self.nwbData['acquisition']['timeseries'][self.recording_name]['spikes'].keys())
                    # Each entry in the electrode dict is itself a dict containing keys 'timestamps' and 'data'...
                    for i_electrode in electrode_dict.keys():
                        data_and_ts_dict = {'timestamps': None, 'data': None}
                        data_and_ts_dict['timestamps'] = np.array(self.nwbData['acquisition']['timeseries'][self.recording_name]['spikes'][i_electrode]['timestamps'])
                        data_and_ts_dict['data'] = np.array(self.nwbData['acquisition']['timeseries'][self.recording_name]['spikes'][i_electrode]['data'])
                        electrode_dict[i_electrode] = data_and_ts_dict
            self.spikeData = electrode_dict
        except:
            self.ts = self.xy

    def save_ttl(self, out_fname):
        '''
        Saves the ttl data to text file out_fname
        '''
        import numpy as np
        if ( len(self.ttl_data) > 0 ) and ( len(self.ttl_timestamps) > 0 ):
            data = np.array([self.ttl_data, self.ttl_timestamps])
            if data.shape[0] == 2:
                data = data.T
            np.savetxt(out_fname, data, delimiter='\t')

    def exportPos(self):
        xy = self.plotPos(show=False)
        out = np.hstack([xy.T, self.xyTS[:,np.newaxis]])
        np.savetxt('position.txt', out, delimiter=',', fmt=['%3.3i','%3.3i','%3.3f'])

    def loadKilo(self):
        '''
        Loads a kilosort session
        '''
        kilodata = KiloSortSession(self.fname_root)
        kilodata.load()
        kilodata.removeNoiseClusters()
        self.kilodata = kilodata

    def __alignTimeStamps__(self):
        '''
        For some reason the timestamps of the data in self.xyTS and self.ts start
        at some positive, non-zero value (timestamps might 'start' when the Play
        button is pressed as opposed to the Record button in openephys). Also
        the position capture might not start until half a second or so after 
        continuous acquisition has. Lastly, the spike times that come out of
        KiloSort are zero-based as they work from the number of samples saved in
        the .dat file.

        Zero to the first timestamp of the continuous recording
        TODO: Could remove data from the continuous recording that happens before the
        first pos timestamp but leave for now

        '''
        # self.xyTS = self.xyTS - self.ts[0]
        # self.ts = self.ts - self.ts[0]
        self.timeAligned = True

    def plotXCorrs(self):
        if self.kilodata is None:
            self.loadKilo()
        from ephysiopy.ephys_generic.ephys_generic import SpikeCalcsGeneric
        corriter = SpikeCalcsGeneric(self.kilodata.spk_times)
        corriter.spk_clusters = self.kilodata.spk_clusters
        corriter.plotAllXCorrs(self.kilodata.good_clusters)
        # for cluster in corriter:
            # print("Cluster {0}".format(cluster))

    def plotWaves(self):
        if self.kilodata is None:
            self.loadKilo()
        if self.rawData is None:
            print("Loading raw data...")
            self.load(loadraw=True)
        import os
        amplitudes = np.load(os.path.join(self.fname_root, 'amplitudes.npy'))
        waveiter = SpkWaveform(self.kilodata.good_clusters, self.kilodata.spk_times, self.kilodata.spk_clusters, amplitudes, self.rawData)
        for cluster in waveiter:
            print("Cluster {}".format(cluster))

    def plotPos(self, jumpmax=None, show=True):
        '''
        Plots x vs y position for this trial

        Parameters
        ------------
        jumpmax - the max amount the LED is allowed to instantaneously move. int or greater
        show - boolean - whether to plot the pos or not (default is True)

        Returns
        ----------
        xy - a nx2 np.array of xy
        '''
        if jumpmax is None:
            jumpmax = self.jumpmax
        import matplotlib.pylab as plt
        from ephysiopy.ephys_generic.ephys_generic import PosCalcsGeneric
        import os
        settings = Settings(os.path.join(self.fname_root, 'settings.xml'))
        settings.parsePos()
        posProcessor = PosCalcsGeneric(self.xy[:,0], self.xy[:,1], 300, True, jumpmax)
        settings = Settings(os.path.join(self.fname_root, 'settings.xml'))
        settings.parsePos()
        xy, hdir = posProcessor.postprocesspos(settings.tracker_params)
        self.hdir = hdir
        if show:
            plt.plot(xy[0], xy[1])
            plt.gca().invert_yaxis()
            plt.show()
        return xy

    def plotMaps(self, plot_type='map', **kwargs):
        '''
        Parameters
        ------------
        plot_type - str - valid strings include:
                        'map' - just ratemap plotted
                        'path' - just spikes on path
                        'both' - both of the above
                        'all' - both spikes on path, ratemap & SAC plotted
                        can also be a list
        Valid kwargs only 'ppm' at the moment - this is an integer denoting pixels per metre:
                                                lower values = more bins in ratemap / SAC
        '''
        if self.kilodata is None:
            self.loadKilo()
        if self.timeAligned == False:
            self.__alignTimeStamps__()
        if ( 'ppm' in kwargs.keys() ):
            ppm = kwargs['ppm']
        else:
            ppm = 400
        from ephysiopy.ephys_generic.ephys_generic import PosCalcsGeneric, MapCalcsGeneric
        posProcessor = PosCalcsGeneric(self.xy[:,0], self.xy[:,1], ppm, jumpmax=self.jumpmax)
        import os
        settings = Settings(os.path.join(self.fname_root, 'settings.xml'))
        settings.parsePos()
        xy, hdir = posProcessor.postprocesspos(settings.tracker_params)
        self.hdir = hdir
        spk_times = (self.kilodata.spk_times.T[0] / 3e4) + self.ts[0]
        mapiter = MapCalcsGeneric(xy, np.squeeze(hdir), posProcessor.speed, self.xyTS, spk_times, plot_type, **kwargs)
        mapiter.good_clusters = self.kilodata.good_clusters
        mapiter.spk_clusters = self.kilodata.spk_clusters
        self.mapiter = mapiter
        mapiter.plotAll()
        # [ print("") for cluster in mapiter ]

    def plotMapsOneAtATime(self, plot_type='map', **kwargs):
        '''
        Parameters
        ------------
        plot_type - str - valid strings include:
                        'map' - just ratemap plotted
                        'path' - just spikes on path
                        'both' - both of the above
                        'all' - both spikes on path, ratemap & SAC plotted
                        can also be a list
        Valid kwargs only 'ppm' at the moment - this is an integer denoting pixels per metre:
                                                lower values = more bins in ratemap / SAC
        '''
        if self.kilodata is None:
            self.loadKilo()
        if self.timeAligned == False:
            self.__alignTimeStamps__()
        if ( 'ppm' in kwargs.keys() ):
            ppm = kwargs['ppm']
        else:
            ppm = 400
        from ephysiopy.ephys_generic.ephys_generic import PosCalcsGeneric, MapCalcsGeneric
        posProcessor = PosCalcsGeneric(self.xy[:,0], self.xy[:,1], ppm, jumpmax=self.jumpmax)
        import os
        settings = Settings(os.path.join(self.fname_root, 'settings.xml'))
        settings.parsePos()
        xy, hdir = posProcessor.postprocesspos(settings.tracker_params)
        self.hdir = hdir
        spk_times = (self.kilodata.spk_times.T[0] / 3e4) + self.ts[0]
        mapiter = MapCalcsGeneric(xy, np.squeeze(hdir), posProcessor.speed, self.xyTS, spk_times, plot_type, **kwargs)
        mapiter.good_clusters = self.kilodata.good_clusters
        mapiter.spk_clusters = self.kilodata.spk_clusters
        self.mapiter = mapiter
        # mapiter.plotAll()
        [ print("") for cluster in mapiter ]

    def plotEEGPower(self, channel=0):
        from ephysiopy.ephys_generic.ephys_generic import EEGCalcsGeneric
        if self.rawData is None:
            print("Loading raw data...")
            self.load(loadraw=True)
        from scipy import signal
        n_samples = np.shape(self.rawData[:,channel])[0]
        s = signal.resample(self.rawData[:,channel], int(n_samples/3e4) * 500)
        E = EEGCalcsGeneric(s, 500)

        # E = EEGCalcsGeneric(self.rawData[:,channel], 3e4)
        E.plotPowerSpectrum()

    def plotSpectrogram(self, nSeconds=30, secsPerBin=2, ax=None, ymin=0, ymax=250):
        from ephysiopy.ephys_generic.ephys_generic import EEGCalcsGeneric
        if self.rawData is None:
            print("Loading raw data...")
            self.load(loadraw=True)
        # load first 30 seconds by default
        fs = 3e4
        E = EEGCalcsGeneric(self.rawData[0:int(3e4*nSeconds),0], fs)
        nperseg = int(fs * secsPerBin)
        from scipy import signal
        freqs, times, Sxx = signal.spectrogram(E.sig, fs, nperseg=nperseg)
        Sxx_sm = Sxx
        from ephysiopy.ephys_generic import binning
        R = binning.RateMap()
        Sxx_sm = R.blurImage(Sxx, (secsPerBin*2)+1)
        x, y = np.meshgrid(times, freqs)
        from matplotlib import colors
        if ax is None:
            plt.figure()
            ax = plt.gca()
            ax.pcolormesh(x, y, Sxx_sm, edgecolors='face', norm=colors.LogNorm())
        ax.pcolormesh(x, y, Sxx_sm, edgecolors='face', norm=colors.LogNorm())
        ax.set_xlim(times[0], times[-1])
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('Time(s)')
        ax.set_ylabel('Frequency(Hz)')

    def plotPSTH(self):
        import os
        settings = Settings(os.path.join(self.fname_root, 'settings.xml'))
        settings.parseStimControl()
        if self.kilodata is None:
            self.loadKilo()
        if self.timeAligned == False:
            self.__alignTimeStamps__()
        from ephysiopy.ephys_generic.ephys_generic import SpikeCalcsGeneric
        spk_times = (self.kilodata.spk_times.T[0] / 3e4) + self.ts[0] # in seconds
        S = SpikeCalcsGeneric(spk_times)
        S.event_ts = self.ttl_timestamps[2::2] # this is because some of the trials have two weird events logged at about 2-3 minutes in...
        S.spk_clusters = self.kilodata.spk_clusters
        S.stim_width = 0.01 # in seconds
        for x in self.kilodata.good_clusters:
            print(next(S.plotPSTH(x)))

    def plotEventEEG(self):
        from ephysiopy.ephys_generic.ephys_generic import EEGCalcsGeneric
        if self.rawData is None:
            print("Loading raw data...")
            self.load(loadraw=True)
        E = EEGCalcsGeneric(self.rawData[:, 0], 3e4)
        event_ts = self.ttl_timestamps[2::2] # this is because some of the trials have two weird events logged at about 2-3 minutes in...
        E.plotEventEEG(event_ts)


class SpkTimeCorrelogram(object):
    def __init__(self, clusters, spk_times, spk_clusters):
        from dacq2py import spikecalcs
        self.SpkCalcs = spikecalcs.SpikeCalcs()
        self.clusters = clusters
        self.spk_times = spk_times
        self.spk_clusters = spk_clusters

    def plotAll(self):
        fig = plt.figure(figsize=(10,20))
        nrows = np.ceil(np.sqrt(len(self.clusters))).astype(int)
        for i, cluster in enumerate(self.clusters):
            cluster_idx = np.nonzero(self.spk_clusters == cluster)[0]
            cluster_ts = np.ravel(self.spk_times[cluster_idx])
            # ts into milliseconds ie OE sample rate / 1000
            y = self.SpkCalcs.xcorr(cluster_ts.T / 30.)
            ax = fig.add_subplot(nrows,nrows,i+1)
            ax.hist(y[y != 0], bins=201, range=[-500, 500], color='k', histtype='stepfilled')
            ax.set_xlabel('Time(ms)')
            ax.set_xlim(-500,500)
            ax.set_xticks((-500, 0, 500))
            ax.set_xticklabels((str(-500), '0', str(500)))
            ax.tick_params(axis='both', which='both', left=False, right=False,
                            bottom=False, top=False)
            ax.set_yticklabels('')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.set_title(cluster, fontweight='bold', size=8, pad=1)
        plt.show()

    def __iter__(self):
        # NOTE:
        # Will plot clusters in self.clusters in separate figure windows
        for cluster in self.clusters:
            cluster_idx = np.nonzero(self.spk_clusters == cluster)[0]
            cluster_ts = np.ravel(self.spk_times[cluster_idx])
            # ts into milliseconds ie OE sample rate / 1000
            y = self.SpkCalcs.xcorr(cluster_ts.T / 30.)
            plt.figure()
            ax = plt.gca()
            ax.hist(y[y != 0], bins=201, range=[-500, 500], color='k', histtype='stepfilled')
            ax.set_xlabel('Time(ms)')
            ax.set_xlim(-500,500)
            ax.set_xticks((-500, 0, 500))
            ax.set_xticklabels((str(-500), '0', str(500)))
            ax.tick_params(axis='both', which='both', left='off', right='off',
							bottom='off', top='off')
            ax.set_yticklabels('')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.set_title('Cluster ' + str(cluster))
            plt.show()
            yield cluster

class SpkWaveform(object):
    '''

    '''
    def __init__(self, clusters, spk_times, spk_clusters, amplitudes, raw_data):
        '''
        spk_times in samples
        '''
        self.clusters = clusters
        self.spk_times = spk_times
        self.spk_clusters = spk_clusters
        self.amplitudes = amplitudes
        self.raw_data = raw_data

    def __iter__(self):
        # NOTE:
        # Will plot in a separate figure window for each cluster in self.clusters
        #
        # get 500us pre-spike and 1000us post-spike interval
        # calculate outside for loop
        pre = int(0.5 * 3e4 / 1000)
        post = int(1.0 * 3e4 / 1000)
        nsamples = np.shape(self.raw_data)[0]
        nchannels = np.shape(self.raw_data)[1]
        times = np.linspace(-pre, post, pre+post, endpoint=False) / (3e4 / 1000)
        times = np.tile(np.expand_dims(times,1),nchannels)
        for cluster in self.clusters:
            cluster_idx = np.nonzero(self.spk_clusters == cluster)[0]
            nspikes = len(cluster_idx)
            data_idx = self.spk_times[cluster_idx]
            data_from_idx = (data_idx-pre).astype(int)
            data_to_idx = (data_idx+post).astype(int)
            raw_waves = np.zeros([nspikes, pre+post, nchannels], dtype=np.int16)

            for i, idx in enumerate(zip(data_from_idx, data_to_idx)):
                if (idx[0][0] < 0):
                    raw_waves[i,0:idx[1][0],:] = self.raw_data[0:idx[1][0],:]
                elif (idx[1][0] > nsamples):
                    raw_waves[i,(pre+post)-((pre+post)-(idx[1][0]-nsamples)):(pre+post),:] = self.raw_data[idx[0][0]:nsamples,:]
                else:
                    raw_waves[i,:,:] = self.raw_data[idx[0][0]:idx[1][0]]

#            filt_waves = self.butterFilter(raw_waves,300,6000)
            mean_filt_waves = np.mean(raw_waves,0)
            plt.figure()
            ax = plt.gca()
            ax.plot(times, mean_filt_waves[:,:])
            ax.set_title('Cluster ' + str(cluster))
            plt.show()
            yield cluster
    def plotAll(self):
        # NOTE:
        # Will plot all clusters in self.clusters in a single figure window
        fig = plt.figure(figsize=(10,20))
        nrows = np.ceil(np.sqrt(len(self.clusters))).astype(int)
        for i, cluster in enumerate(self.clusters):
            cluster_idx = np.nonzero(self.spk_clusters == cluster)[0]
            nspikes = len(cluster_idx)
            data_idx = self.spk_times[cluster_idx]
            data_from_idx = (data_idx-pre).astype(int)
            data_to_idx = (data_idx+post).astype(int)
            raw_waves = np.zeros([nspikes, pre+post, nchannels], dtype=np.int16)

            for i, idx in enumerate(zip(data_from_idx, data_to_idx)):
                if (idx[0][0] < 0):
                    raw_waves[i,0:idx[1][0],:] = self.raw_data[0:idx[1][0],:]
                elif (idx[1][0] > nsamples):
                    raw_waves[i,(pre+post)-((pre+post)-(idx[1][0]-nsamples)):(pre+post),:] = self.raw_data[idx[0][0]:nsamples,:]
                else:
                    raw_waves[i,:,:] = self.raw_data[idx[0][0]:idx[1][0]]

            mean_filt_waves = np.mean(raw_waves,0)
            ax = fig.add_subplot(nrows,nrows,i+1)
            ax.plot(times, mean_filt_waves[:,:])
            ax.set_title(cluster, fontweight='bold', size=8)
        plt.show()

    def butterFilter(self, sig, low, high, order=5):
        nyqlim = 3e4 / 2
        lowcut = low / nyqlim
        highcut = high / nyqlim
        from scipy import signal as signal
        b, a = signal.butter(order, [lowcut, highcut], btype='band')
        return signal.filtfilt(b, a, sig)