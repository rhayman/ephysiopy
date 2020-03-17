#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:53:10 2017

@author: robin

The main interface for dealing with openephys data recorded in
either the .nwb or binary format (ie when using neuropixels)

4 classes defined:

KiloSortSession - deals with the output from kilosort or kilosort2
OpenEphysBase - a not really abstract base class which is the parent (super)
				class for these two
	OpenEphysNPX - deals with neuropixels data
	OpenEphysNWB - deals with data recorded in nwb format

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
		'''
		Walk through the path to find the location of the files in case this has been
		called in another way i.e. binary format a la Neuropixels
		'''
		self.fname_root = fname_root
		import os
		for d, c, f in os.walk(fname_root):
			for ff in f:
				if 'spike_times.npy' in ff:
					self.fname_root = d
		self.cluster_id = None
		self.spk_clusters = None
		self.spk_times = None

	def load(self):
		'''
		Load all the relevant files
		KSLabel - output from KiloSort - algorithm defined
		cluster_group - group labels from phy - user defined ('good', 'MUA', 'noise')
		'''
		import os
		dtype = {'names': ('cluster_id', 'group'), 'formats': ('i4', 'S10')}
		# One of these is from kilosort and the other from kilosort2
		# and is updated by the user when doing cluster assignment in phy (or whatever)
		# See comments above this class definition for a bit more info
		if os.path.exists(os.path.join(self.fname_root, 'cluster_groups.csv')):
			self.cluster_id, self.group = np.loadtxt(os.path.join(self.fname_root, 'cluster_groups.csv'), unpack=True, skiprows=1, dtype=dtype)
		if os.path.exists(os.path.join(self.fname_root, 'cluster_group.tsv')):
			self.cluster_id, self.group = np.loadtxt(os.path.join(self.fname_root, 'cluster_group.tsv'), unpack=True, skiprows=1, dtype=dtype)
		dtype = {'names': ('cluster_id', 'KSLabel'), 'formats': ('i4', 'S10')}
		# 'Raw' labels from a kilosort session
		if os.path.exists(os.path.join(self.fname_root, 'cluster_KSLabel.tsv')):
			self.ks_cluster_id, self.ks_group = np.loadtxt(os.path.join(self.fname_root, 'cluster_KSLabel.tsv'), unpack=True, skiprows=1, dtype=dtype)
		self.spk_clusters = np.squeeze(np.load(os.path.join(self.fname_root, 'spike_clusters.npy')))
		self.spk_times    = np.squeeze(np.load(os.path.join(self.fname_root, 'spike_times.npy')))

	def removeNoiseClusters(self):
		'''
		Restricts analysis to anything that isn't listed as 'noise' in self.group
		'''
		if self.cluster_id is not None:
			self.good_clusters = []
			for id_group in zip(self.cluster_id, self.group):
				if 'noise' not in id_group[1].decode() and 'mua' not in id_group[1].decode():
					self.good_clusters.append(id_group[0])


class OpenEphysBase(object):
	"""
	Base class for openephys anaylsis with data recorded in either the NWB or binary format
	"""
	def __init__(self, pname_root, **kwargs):
		super().__init__()
		self.pname_root = pname_root # top-level directory, typically of form YYYY-MM-DD_HH-MM-SS
		self.settings = None
		self.kilodata = None
		self.rawData = None
		self.xy = None
		self.xyTS = None
		self.recording_start_time = 0
		self.ts = None
		self.ttl_data = None
		self.ttl_timestamps = None
		self.spikeData = None # a list of np.arrays, nominally containing tetrode data in format nspikes x 4 x 40
		self.accelerometerData = None # np.array
		self.settings = None # OESettings.Settings instance
		if ('jumpmax' in kwargs.keys()):
			self.jumpmax = kwargs['jumpmax']
		else:
			self.jumpmax = 100

	def loadKilo(self):
		'''
		Loads a kilosort session
		'''
		kilodata = KiloSortSession(self.pname_root) # pname_root gets walked through and over-written with correct location of kiolsort data
		kilodata.load()
		kilodata.removeNoiseClusters()
		self.kilodata = kilodata

	def __loadSettings__(self):
		'''
		Loads the settings.xml data
		'''
		if self.settings is None:
			import os
			settings = Settings(self.pname_root) # pname_root gets walked through and over-written with correct location of settings.xml
			settings.parse()
			settings.parsePos()
			self.settings = settings

	def __loaddata__(self, **kwargs):
		self.load(self.pname_root, **kwargs)

	def plotXCorrs(self):
		if self.kilodata is None:
			self.loadKilo()
		from ephysiopy.ephys_generic.ephys_generic import SpikeCalcsGeneric
		corriter = SpikeCalcsGeneric(self.kilodata.spk_times)
		corriter.spk_clusters = self.kilodata.spk_clusters
		corriter.plotAllXCorrs(self.kilodata.good_clusters)

	def plotPos(self, jumpmax=None, show=True, **kwargs):
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

		self.__loadSettings__()
		if self.xy is None:
			self.__loaddata__(**kwargs)
		posProcessor = PosCalcsGeneric(self.xy[:,0], self.xy[:,1], ppm=300, cm=True, jumpmax=jumpmax)
		xy, hdir = posProcessor.postprocesspos(self.settings.tracker_params)
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
		Valid kwargs: 'ppm' - this is an integer denoting pixels per metre:
												lower values = more bins in ratemap / SAC
				'clusters' - int or list of ints describing which clusters to plot
				i.e. this overwrites the 'good_clusters' value in self.kilodata
		'''
		if self.kilodata is None:
			self.loadKilo()
		if ( 'ppm' in kwargs.keys() ):
			ppm = kwargs['ppm']
		else:
			ppm = 400
		from ephysiopy.ephys_generic.ephys_generic import PosCalcsGeneric, MapCalcsGeneric
		if self.xy is None:
			self.__loaddata__(**kwargs)
		posProcessor = PosCalcsGeneric(self.xy[:,0], self.xy[:,1], ppm, jumpmax=self.jumpmax)
		import os
		self.__loadSettings__()
		xy, hdir = posProcessor.postprocesspos(self.settings.tracker_params)
		self.hdir = hdir
		spk_times = (self.kilodata.spk_times.T / 3e4) + self.recording_start_time
		mapiter = MapCalcsGeneric(xy, np.squeeze(hdir), posProcessor.speed, self.xyTS, spk_times, plot_type, **kwargs)
		if 'clusters' in kwargs:
			if type(kwargs['clusters']) == int:
				mapiter.good_clusters = np.intersect1d([kwargs['clusters']], self.kilodata.good_clusters)

			else:
				mapiter.good_clusters = np.intersect1d(kwargs['clusters'], self.kilodata.good_clusters)
		else:
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
		Valid kwargs: 'ppm' - this is an integer denoting pixels per metre:
												lower values = more bins in ratemap / SAC
				'clusters' - int or list of ints describing which clusters to plot
				i.e. this overwrites the 'good_clusters' value in self.kilodata
				'save_grid_summary_location' - bool; if present the dictionary returned from
				gridcell.SAC.getMeasures is saved for each cluster - this is passed to MapCalcsGeneric
		'''
		if self.kilodata is None:
			self.loadKilo()
		if ( 'ppm' in kwargs.keys() ):
			ppm = kwargs['ppm']
		else:
			ppm = 400
		from ephysiopy.ephys_generic.ephys_generic import PosCalcsGeneric, MapCalcsGeneric
		if self.xy is None:
			self.__loaddata__(**kwargs)
		posProcessor = PosCalcsGeneric(self.xy[:,0], self.xy[:,1], ppm, jumpmax=self.jumpmax)
		import os
		self.__loadSettings__()
		xy, hdir = posProcessor.postprocesspos(self.settings.tracker_params)
		self.hdir = hdir
		spk_times = (self.kilodata.spk_times.T / 3e4) + self.recording_start_time
		mapiter = MapCalcsGeneric(xy, np.squeeze(hdir), posProcessor.speed, self.xyTS, spk_times, plot_type, **kwargs)
		if 'clusters' in kwargs:
			if type(kwargs['clusters']) == int:
				mapiter.good_clusters = np.intersect1d([kwargs['clusters']], self.kilodata.good_clusters)

			else:
				mapiter.good_clusters = np.intersect1d(kwargs['clusters'], self.kilodata.good_clusters)
		else:
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
		self.__loadSettings__()
		self.settings.parseStimControl()
		if self.kilodata is None:
			self.loadKilo()
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

class OpenEphysNPX(OpenEphysBase):
	"""docstring for OpenEphysNPX"""
	def __init__(self, pname_root):
		super().__init__(pname_root)
		self.path2PosData = None
		self.path2APdata = None
		self.path2LFPdata = None

	def load(self, pname_root=None, experiment_name='experiment1', recording_name='recording1'):
		'''
		Loads data recorded in the OE 'flat' binary format
		See https://open-ephys.atlassian.net/wiki/spaces/OEW/pages/166789121/Flat+binary+format for more

		Parameters
		----------
		pname_root - str - the top level directory that contains the folder 'experiment1'
		(usually) and settings.xml

		recording_name - str - pretty obvious but this is also the directory immediately beneath pname_root
		'''
		self.isBinary = True
		import os
		import re
		pos_t_match = re.compile('Pos_Tracker-[0-9][0-9][0-9].[0-9]')
		APdata_match = re.compile('Neuropix-PXI-[0-9][0-9][0-9].0')
		LFPdata_match = re.compile('Neuropix-PXI-[0-9][0-9][0-9].1')
		sync_message_file = None
		self.recording_start_time = None

		if pname_root is None:
			pname_root = self.pname_root

		for d, c, f in os.walk(pname_root):
			for ff in f:
				if 'data_array.npy' in ff:
					self.path2PosData = os.path.join(d)
				if 'continuous.dat' in ff:
					if APdata_match.search(d):
						self.path2APdata = os.path.join(d)
					if LFPdata_match.search(d):
						self.path2LFPdata = os.path.join(d)
				if 'sync_messages.txt' in ff:
					sync_message_file = os.path.join(d, 'sync_messages.txt')

		if self.path2PosData is not None:
			pos_data = np.load(os.path.join(self.path2PosData, 'data_array.npy'))
			self.xy = pos_data[:,0:2]
			pos_ts = np.load(os.path.join(self.path2PosData, 'timestamps.npy'))
			self.xyTS = pos_ts / 30.0 / 1000.0

		ap_sample_rate = 30000
		n_channels = 384
		trial_length = self.__calcTrialLengthFromBinarySize__(os.path.join(self.path2APdata, 'continuous.dat'), n_channels, ap_sample_rate)
		# Load the start time from the sync_messages file
		if sync_message_file is not None:
			with open(sync_message_file, 'r') as f:
				sync_strs = f.read()
			sync_lines = sync_strs.split('\n')
			for line in sync_lines:
				if 'subProcessor: 0' in line:
					idx = line.find('start time: ')
					start_val = line[idx + len('start time: '):-1]
					tmp = start_val.split('@')
					recording_start_time = float(tmp[0]) / float(tmp[1][0:-1])
		else:
			recording_start_time = self.xyTS[0]
		self.recording_start_time = recording_start_time
		self.ts = np.arange(recording_start_time, trial_length+recording_start_time, 1.0 / ap_sample_rate)

	def __calcTrialLengthFromBinarySize__(self, path2file:str, n_channels=384, sample_rate=30000):
		'''
		Returns the time taken to run the trial (in seconds) based on the size of
		the binary file on disk
		'''
		import os
		status = os.stat(path2file)
		return status.st_size / ( 2.0 * n_channels * sample_rate)

	def plotSpectrogramByDepth(self, nchannels=384, nseconds=100, maxFreq=125, **kwargs):
		import os
		lfp_file = os.path.join(self.path2LFPdata, 'continuous.dat')
		status = os.stat(lfp_file)
		nsamples = int(status.st_size / 2 / nchannels)
		mmap = np.memmap(lfp_file, np.int16, 'r', 0, (nchannels, nsamples), order='F')
		# Load the channel map NB assumes this is in the AP data location and that kilosort was run there
		channel_map = np.squeeze(np.load(os.path.join(self.path2APdata, 'channel_map.npy')))
		lfp_sample_rate = 2500
		data = np.array(mmap[channel_map, 0:nseconds*lfp_sample_rate])
		from ephysiopy.ephys_generic.ephys_generic import EEGCalcsGeneric
		E = EEGCalcsGeneric(data[0, :], lfp_sample_rate)
		E.calcEEGPowerSpectrum()
		spec_data = np.zeros(shape=(data.shape[0], len(E.sm_power[0::50])))
		for chan in range(data.shape[0]):
			E = EEGCalcsGeneric(data[chan, :], lfp_sample_rate)
			E.calcEEGPowerSpectrum()
			spec_data[chan, :] = E.sm_power[0::50]

		x, y = np.meshgrid(E.freqs[0::50], channel_map)
		import matplotlib.colors as colors
		from matplotlib.pyplot import cm
		from mpl_toolkits.axes_grid1 import make_axes_locatable
		fig, spectoAx = plt.subplots()
		spectoAx.pcolormesh(x, y, spec_data, edgecolors='face', cmap='bone',norm=colors.LogNorm())
		spectoAx.set_xlim(0, maxFreq)
		spectoAx.set_ylim(channel_map[0], channel_map[-1])
		spectoAx.set_xlabel('Frequency (Hz)')
		spectoAx.set_ylabel('Channel')
		divider = make_axes_locatable(spectoAx)
		channel_spectoAx = divider.append_axes("top", 1.2, pad = 0.1, sharex=spectoAx)
		meanfreq_powerAx = divider.append_axes("right", 1.2, pad = 0.1, sharey=spectoAx)
		plt.setp(channel_spectoAx.get_xticklabels() + meanfreq_powerAx.get_yticklabels(), visible=False)

		mn_power = np.mean(spec_data, 0)
		cols = iter(cm.rainbow(np.linspace(0,1,(nchannels//60)+1)))
		for i in range(0, spec_data.shape[0], 60):
			c = next(cols)
			channel_spectoAx.plot(E.freqs[0::50], 10*np.log10(spec_data[i, :]/mn_power), c=c, label=str(i))

		channel_spectoAx.set_ylabel('Channel power(dB)')
		channel_spectoAx.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', mode='expand',
			fontsize='x-small', ncol=4)

		freq_inc = 6
		lower_freqs = np.arange(1, maxFreq-freq_inc, freq_inc)
		upper_freqs = np.arange(1+freq_inc, maxFreq, freq_inc)
		cols = iter(cm.nipy_spectral(np.linspace(0,1,len(upper_freqs))))
		mn_power = np.mean(spec_data, 1)
		for freqs in zip(lower_freqs, upper_freqs):
			freq_mask = np.logical_and(E.freqs[0::50]>freqs[0], E.freqs[0::50]<freqs[1])
			mean_power = 10*np.log10(np.mean(spec_data[:, freq_mask],1)/mn_power)
			c = next(cols)
			meanfreq_powerAx.plot(mean_power, channel_map, c=c, label=str(freqs[0]) + " - " + str(freqs[1]))
		meanfreq_powerAx.set_xlabel('Mean freq. band power(dB)')
		meanfreq_powerAx.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', mode='expand',
			fontsize='x-small', ncol=1)
		plt.show()

	def plotPos(self, jumpmax=None, show=True):
		super().plotPos(jumpmax, show)

	def plotMaps(self, plot_type='map', **kwargs):
		super().plotMaps(plot_type, **kwargs)

	def plotMapsOneAtATime(self, plot_type='map', **kwargs):
		super().plotMapsOneAtATime(plot_type, **kwargs)

	def plotEEGPower(self, channel=0):
		super().plotEEGPower(channel)

	def plotSpectrogram(self, nSeconds=30, secsPerBin=2, ax=None, ymin=0, ymax=250):
		super().plotSpectrogram(self, nSeconds, secsPerBin, ax, ymin, ymax)

	def plotPSTH(self):
		super().plotPSTH()

	def plotEventEEG(self):
		super().plotEventEEG()

	def plotWaves(self):
		super().plotWaves()

class OpenEphysNWB(OpenEphysBase):
	'''
	Parameters
	------------
	fname_root:- str
		Should contain the settings.xml file and the .nwb file
	'''

	def __init__(self, pname_root, **kwargs):
		super().__init__(pname_root)
		self.nwbData = None # handle to the open nwb file (HDF5 file object)
		self.rawData = None # np.array holding the raw, continuous recording
		self.recording_name = None # the recording name inside the nwb file ('recording0', 'recording1', etc)
		self.isBinary = False

	def load(self, pname_root: None, session_name=None, recording_name=None, loadraw=False, loadspikes=False, savedat=False):
		'''
		Loads xy pos from binary part of the hdf5 file and data resulting from
		a Kilosort session (see KiloSortSession class above)

		Parameters
		----------
		pname_root : str
			The top level directory, typically the one named YYYY-MM-DD_HH-MM-SS
			NB In the nwb format this directory contains the experiment_1.nwb and settings.xml files
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
		if pname_root is None:
			pname_root = self.pname_root
		if session_name is None:
			session_name = 'experiment_1.nwb'
		self.nwbData = h5py.File(os.path.join(pname_root, session_name), mode='r')
		# Position data...
		if self.recording_name is None:
			if recording_name is None:
				recording_name = 'recording1'
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
			self.__loadSettings__()
			fpgaId = self.settings.fpga_nodeId
			fpgaNode = 'processor' + str(fpgaId) + '_' + str(fpgaId)
			self.ts = np.array(self.nwbData['acquisition']['timeseries'][self.recording_name]['continuous'][fpgaNode]['timestamps'])
			if (loadraw == True):
				self.rawData = np.array(self.nwbData['acquisition']['timeseries'][self.recording_name]['continuous'][fpgaNode]['data'])
				self.settings.parseChannels() # to get the neural data channels
				self.accelerometerData = self.rawData[:,64:]
				self.rawData = self.rawData[:,0:64]
				if (savedat == True):
					data2save = self.rawData[:,0:64]
					data2save.tofile(os.path.join(pname_root, 'experiment_1.dat'))
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
		if ( len(self.ttl_data) > 0 ) and ( len(self.ttl_timestamps) > 0 ):
			data = np.array([self.ttl_data, self.ttl_timestamps])
			if data.shape[0] == 2:
				data = data.T
			np.savetxt(out_fname, data, delimiter='\t')

	def exportPos(self):
		xy = self.plotPos(show=False)
		out = np.hstack([xy.T, self.xyTS[:,np.newaxis]])
		np.savetxt('position.txt', out, delimiter=',', fmt=['%3.3i','%3.3i','%3.3f'])

	def plotPos(self, jumpmax=None, show=True):
		super().plotPos(jumpmax, show)

	def plotMaps(self, plot_type='map', **kwargs):
		super().plotMaps(plot_type, **kwargs)

	def plotMapsOneAtATime(self, plot_type='map', **kwargs):
		super().plotMapsOneAtATime(plot_type, **kwargs)

	def plotEEGPower(self, channel=0):
		super().plotEEGPower(channel)

	def plotSpectrogram(self, nSeconds=30, secsPerBin=2, ax=None, ymin=0, ymax=250):
		super().plotSpectrogram(self, nSeconds, secsPerBin, ax, ymin, ymax)

	def plotPSTH(self):
		super().plotPSTH()

	def plotEventEEG(self):
		super().plotEventEEG()

	def plotWaves(self):
		super().plotWaves()

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
