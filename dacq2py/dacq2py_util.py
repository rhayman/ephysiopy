from __future__ import division
from scipy import signal, stats, ndimage
from datetime import datetime
import os
import re
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
from . import axonaIO
from .tetrode_dict import TetrodeDict
from ephysiopy.ephys_generic import binning
from .fieldcalcs import FieldCalcs
from .spikecalcs import SpikeCalcs
from .eegcalcs import EEGCalcs
from .cluster import Kluster
from . import tintcolours as tcols
from .gridcell import SAC
from itertools import combinations
from mpl_toolkits.axes_grid1 import ImageGrid
import skimage, skimage.morphology, skimage.feature
from collections import OrderedDict

warnings.filterwarnings("ignore",
						message="divide by zero encountered in int_scalars")
warnings.filterwarnings("ignore",
						message="divide by zero encountered in divide")
warnings.filterwarnings("ignore",
						message="invalid value encountered in divide")
warnings.filterwarnings("ignore",
						message="Casting complex values to real discards the imaginary part")
class Trial(axonaIO.IO, SAC, dict):
	'''
	Providesm ethods to plot electrophysiology data acquired using the Axona DACQ recording system
	and methods to extract some measures from that data

	The actual loading of the data is done lazily i.e. only when you ask for
	position data (say plotting the path the animal took in the trial) is the
	position data actually loaded. The class also uses as attibutes several
	instances of subpackages (binning.Ratemap for example) so that the code
	could be made more modular.

	Attributes:
		filename_root : str
			Absolute location on the filesystem of the set of files without a suffix
		basename : str
			Basename of the set of files without a suffix (everything after the last trailing slash)
		EEG : dacq2py.axonaIO.EEG class
			Containing data from .eeg file
		EGF : dacq2py.axonaIO.EEG class
			Containing data from .egf file
		STM : dacq2py.axonaIO.Stim class
			Contains stimulation data (timestamps mostly) and header + some additions work done below
		POS : dacq2py.axonaIO.Pos class
			Contains raw and post-processed position data (xy, dir, speed etc) & header
		TETRODE : extension of Pythons dict class"
			Each value is an instance of dacq2py.axonaIO.Tetrode. Contains
			methods to get cluster spike times, cluster indices etc
		posFilter : dict
			Keys are things like 'speed', 'time'; values are n x 2 arrays of range of values *to keep*
		setheader : dict
			Corresponds to the .set file for the file set. Keys/ values are all strings
		_available_files : list
			All files matching the filename_root + any valid suffix
		metadata : OrderedDict
			Some basic info if the file is an *rh one (see _parseMetaData)
		ratemap : dacq2py.binning.Ratemap class instance

	See Also
	--------
	binning
		Basic binning of data, calculation of bin sizes etc
	eegcalcs
		Contains filters, eeg power spectra methods
	spikecalcs
		Temporal measures of spike trains (firing rates etc) and extracting
		parameters from the waveforms and clusters themselves
	fieldcalcs
		Methods for extracting information from 2D ratemaps mostly but also
		contains some statistical tools (information theoretic measures etc)
	gridcellTrial
		Trial inherits from this at the moment. Includes methods for obtaining
		the spatial autocorrelogram (SAC) (and cross-correlogram) and plotting of the
		SAC

	Examples
	--------
	>>> from dacq2py.dacq2py_util import Trial
	>>> T = Trial(r'/media/robin/data/Dropbox/Science/Recordings/M851/M851_140908t1rh')

	'''

	def __init__(self, filename_root, **kwargs):
		"""
		Parameters
		----------
		filename_root: str
			The absolute filename without any suffix attached
			i.e. C:\\\Robin\\\mytrial

			Note that when RH is using this can be just the trial name as the getFullFile method
			tries to find the trial given the folder layout and the filename - see that method
			for details

		Returns
		-------
		T : object
			a dacq2py_util.Trial object

		Examples
		--------
		>>> T = dacq2py_util.Trial(r'/media/robin/data/Dropbox/Science/Recordings/M851/M851_140908t1rh')
		"""

		# try and intelligently get full filename from just the root
		filename_root = self.getFullFile(filename_root)
		self.basename = os.path.basename(filename_root)
		self.filename_root = filename_root
		self._EEG = None
		self._EGF = None
		self._STM = None
		self._POS = None
		if 'volts' in kwargs:
			useVolts = kwargs['volts']
			self.TETRODE = TetrodeDict(filename_root, volts=useVolts)  # see TETRODE class above
		else:
			self.TETRODE = TetrodeDict(filename_root)
		self._posFilter = None  # a dict used to filter pos
		self._setheader = None
		self.ratemap = None #becomes binning.RateMap instance - see POS getter property below
		self.spikecalcs = SpikeCalcs()
		self.fieldcalcs = FieldCalcs()
		self._isinteractive = 1
		self._figNum = 1
		self._min_spks = 1
		self._available_files = None
		self._getAvailableFiles()
		self.metadata = OrderedDict()
		self.tetrodes = None
		self.clusters = None
		self.pos_weights = None
		if 'cm' in kwargs:
			self.useCm = kwargs['cm']
		else:
			self.useCm = False
		try:
			self._parseMetaData()
		except:
			self.metadata = {'Contents': 'Not an rhayman file'}
		try:
			self.getTsAndCs()
		except:
			pass
		self.eeg_file = 1

	def __repr__(self):
		return '{self.__class__.__name__}({self.filename_root})'.format(self=self)

	def hasFiles(self):
		'''
		Checks for some automated yaml processing (see Dropbox/Science/Analysis/)
		'''

		for i in self.axona_files.iterkeys():
			if os.path.isfile(self.filename_root + i):
				self['has_' + i[1:]] = True
			else:
				self['has_' + i[1:]] = False

	def getFullFile(self, filename):
		'''
		Used to constuct filename_root in __init__

		Parameters
		-------------
		filename : str
			The absolute path the files being analysed here without any suffix
		'''
		if os.path.isdir(r'/home/robin/Dropbox/Science/Recordings'):
			pname, fname = os.path.split(filename)
			if len(pname) == 0:
				defaultDir = r'/home/robin/Dropbox/Science/Recordings'
				animal = filename.split('_')[0]
				filename = os.path.join(defaultDir, animal, filename)
		return filename

	@property
	def setheader(self):
		'''
		Returns
		----------
		self.dict: dict
			Matches contents of .set file with keys and values all mapped as strings
		'''

		if self._setheader is None:
			try:
				self._setheader = self.getHeader(self.filename_root + '.set')
			except IOError:
				self._setheader = None
		return self._setheader

	@setheader.setter
	def setheader(self, value):
		self._setheader = value

	@property
	def ppm(self):
		return self.__ppm

	@ppm.setter
	def ppm(self, value):
		self.__ppm = value
		# Update POS
		self.POS.ppm = value
		# Update Ratemap
		self.ratemap = binning.RateMap(self.POS.xy, self.POS.dir, self.POS.speed, self.pos_weights, self.POS.ppm, self.useCm)

	@property
	def POS(self):
		'''
		Returns
		-----------
		self.POS:
			Contains raw and post-processed position data
		'''

		if self._POS is None:
			try:
				self._POS = axonaIO.Pos(self.filename_root, cm=self.useCm)
				self._POS.postprocesspos()
				self._xlims = (int(self.POS.xy[0,:].min()),
							   int(self.POS.xy[0,:].max()))
				self._ylims = (int(self.POS.xy[1,:].min()),
							   int(self.POS.xy[1,:].max()))
				self.pos_weights = np.ravel(np.ones((1, self.POS.npos), dtype=np.float) / self.POS.pos_sample_rate)
				self.ratemap = binning.RateMap(self.POS.xy, self.POS.dir, self.POS.speed, self.pos_weights, self.POS.ppm, self.useCm)
			except IOError:
				self._POS = None
		return self._POS

	@POS.setter
	def POS(self, value):
		self._POS = value

	@property
	def EEG(self):
		'''
		Returns
		------------
		self.EEG:
			eeg data and header
		'''
		if self._EEG is None:
			try:
				self._EEG = axonaIO.EEG(self.filename_root, eeg_file=self.eeg_file)
				self.pos2eegScale = int(self.EEG.sample_rate /
										self.POS.pos_sample_rate)
			except IOError:
				self._EEG = None
		return self._EEG

	@EEG.setter
	def EEG(self, value):
		self._EEG = value

	@property
	def EGF(self):
		'''
		Returns
		------------
		self.EGF:
			eeg data and header from .egf file
		'''
		if self._EGF is None:
			try:
				self._EGF = axonaIO.EEG(self.filename_root, eeg_file=self.eeg_file, egf=1)
				self.pos2egfScale = int(self.EGF.sample_rate /
										self.POS.pos_sample_rate)
			except IOError:
				self._EGF = None
		return self._EGF

	@EGF.setter
	def EGF(self, value):
		self._EGF = value

	@property
	def STM(self):
		'''
		Returns
		------------
		self.Stim:
			Stimulation data and header + some extras parsed from pos, eeg and set files
		'''
		if self._STM is None:
			try:
				self._STM = axonaIO.Stim(self.filename_root)
				'''
				update the STM dict with some relevant values from the .set file and the headers
				of the eeg and pos files
				'''
				posHdr = self.getHeader(self.filename_root + '.pos')
				eegHdr = self.getHeader(self.filename_root + '.eeg')
				self._STM['posSampRate'] = self.getHeaderVal(posHdr, 'sample_rate')
				self._STM['eegSampRate'] = self.getHeaderVal(eegHdr, 'sample_rate')
				try:
					egfHdr = self.getHeader(self.filename_root + '.egf')
					self._STM['egfSampRate'] = self.getHeaderVal(egfHdr, 'sample_rate')
				except:
					pass
				stim_pwidth = int(self.setheader['stim_pwidth']) / int(1000) # get into ms
				self._STM['off'] = self._STM['on'] + int(stim_pwidth)
				"""
				There are a set of key / value pairs in the set file that
				correspond to the patterns/ protocols specified in the
				Stimulator menu in DACQ. Extract those items now...
				There are five possibe "patterns" that can be used in a trial. Those patterns 
				consist of either "Pause (no stimulation)" or some user-defined stimulation pattern.
				Whether or not one of the five was used is specified in "stim_patternmask_n" where n 
				is 1-5. Confusingly in dacqUSB these 5 things are called "Protocols" accessed from
				the menu Stimulator/Protocols... within that window they are actually called "Phase 1",
				"Phase 2" etc. To keep everything in order it's best to iterate through using a for loop
				as a dict is not guaranteed to be ordered and I cba to use an OrderedDict.
				In dacqUSB nomencalture the pattern is actually the stimulation you 
				want to apply i.e. 10ms pulse every 150ms or whatever. The "pattern" is what is applied
				within every Phase.
				"""
				# phase_info : a dict for each phase that is active
				phase_info = {'startTime': None, 'duration': None, 'name': None, 'pulseWidth': None, 'pulsePause': None};
				stim_dict = {};
				stim_patt_dict = {};
				for k,v in self.setheader.iteritems():
					if k.startswith("stim_patternmask_"):
						if (int(v) == 1):
							# get the number of the phase
							phase_num = k[-1]
							stim_dict['Phase_' + phase_num] = phase_info.copy();
					if k.startswith("stim_patt_"):
						stim_patt_dict[k] = v;
				self.patt_dict = stim_patt_dict
				for k,v in stim_dict.iteritems():
					phase_num = k[-1]
					stim_dict[k]['duration'] = int(self.setheader['stim_patterntimes_' + phase_num])
					phase_name = self.setheader['stim_patternnames_' + phase_num]
					stim_dict[k]['name'] = phase_name
					if not (phase_name.startswith("Pause")):
						# find the matching string in the stim_patt_dict
						for kk,vv in stim_patt_dict.iteritems():
							split_str = vv.split('"');
							patt_name = split_str[1]
							if (patt_name == phase_name):
								ss = split_str[2].split()
								stim_dict[k]['pulseWidth'] = int(ss[0])
								stim_dict[k]['pulsePause'] = int(ss[2])
				# make the dict ordered by Phase number
				self.STM['stim_params'] = OrderedDict(sorted(stim_dict.items()));
			except IOError:
				self._STM = None
		return self._STM

	@STM.setter
	def STM(self, value):
		self._STM = value

	@property
	def posFilter(self):
		'''
		self.posFilter : dict
			Keys are strings such as 'speed', 'time' etc. Values are n x 2 arrays of values *to keep*
		'''
		return self._posFilter

	@posFilter.setter
	def posFilter(self, value):
		"""
		Filters data depending on the filter specified in the dictionary value

		Parameters
		----------
		value : dict
			Filter dict. Legal keys include: 'time', 'dir', 'speed', 'xrange',
			'yrange'. If key is 'time', values must be a n x 2 numpy array that 
			specifies the times to keep in SECONDS. If key is 'dir' values must
			be a two element list/ array that specifies the directions to keep
			in DEGREES NB the values can be singular strings of either 'w', 
			'e', 'n' or 's' which filters for a +/-45 degree range around that
			cardinal direction. If key is 'speed' values are a 2 element list/ 
			array to keep specified in m/s. If key is 'xrange' or 'yrange' 
			values are a two element list/ array that specify the x or y values
			to keep in PIXELS.

		Returns
		-------
		modified dacq2py_util.Trial object: object
			The Trial object is modified in place and all the relevant 
			variables are filtered and changed to numpy masked arrays

		Examples
		--------
		>>> import numpy as np
		>>> T = dacq2py_util.Trial(r'D:\M851\M851_140908t1rh')
		>>> T.posFilter = {'time': np.array([600,1200])}
		"""

		# If masked, remove all masks on all aspects of data
		if np.ma.is_masked(self.POS.speed):
			self.POS.speed.mask = np.ma.nomask
		if np.ma.is_masked(self.POS.dir):
			self.POS.dir.mask = np.ma.nomask
		if np.ma.is_masked(self.POS.xy):
			self.POS.xy.mask = np.ma.nomask
		if np.ma.is_masked(self.EEG.eeg):
			self.EEG.eeg.mask = np.ma.nomask
		if np.ma.is_masked(self.EGF.eeg):
			self.EGF.eeg.mask = np.ma.nomask
		if np.any(self.EEG.EEGphase):
			if np.ma.is_masked(self.EEG.EEGphase):
				self.EEG.EEGphase.mask = np.ma.nomask
		if self.TETRODE:#true if TETRODE dict has entries
			for tet in self.TETRODE.iterkeys():
				if np.ma.is_masked(self.TETRODE[tet].waveforms):
					self.TETRODE[tet].waveforms.mask = np.ma.nomask
					self.TETRODE[tet].spk_ts.mask = np.ma.nomask

		if value is None:
			return

		idx = self.POS.filterPos(value)
		if self.TETRODE:
			for tet in self.TETRODE.iterkeys():
				posSamps = self.TETRODE[tet].getPosSamples()
				common = np.in1d(posSamps, np.nonzero(idx)[1])
				# Mask timestamps first as this is a vector, then expand
				# out the mask array (common)
				self.TETRODE[tet].spk_ts = np.ma.masked_where(common, self.TETRODE[tet].spk_ts)
				common = common[:, None, None]
				common = np.repeat(np.repeat(common, 4, axis=1), 50, axis=-1)
				self.TETRODE[tet].waveforms = np.ma.masked_where(common, self.TETRODE[tet].waveforms)

		self.POS.speed = np.squeeze(np.ma.masked_where(idx, np.expand_dims(self.POS.speed,0)))
		self.POS.dir = np.squeeze(np.ma.masked_where(idx, np.expand_dims(self.POS.dir,0)))
		posMask = np.squeeze(idx)
		posMask = np.vstack((posMask, posMask))
		self.POS.xy = np.ma.masked_where(posMask, self.POS.xy)
		self.EEG.eeg = np.ma.masked_where(np.repeat(np.squeeze(idx),	self.pos2eegScale), self.EEG.eeg)
		if self.EGF:
			self.EGF.eeg = np.ma.masked_where(np.repeat(np.squeeze(idx), self.pos2egfScale), self.EGF.eeg)
		if np.any(self.EEG.EEGphase):
			self.EEG.EEGphase = np.ma.masked_where(np.repeat(np.squeeze(idx), self.pos2eegScale), self.EEG.EEGphase)
		self._posFilter = value

	def print_stim_dict(self):
		'''
		Prints out keys/ values of STM dict
		'''
		for k,v in self.STM.iteritems():
			print(k, v)

	def _filterForStm(self, laser=None):
		'''
		Cycles through the STM dict and fiters for laser on / off periods and
		applies the filter to the pos and eeg data NB tetrode data not dealt with
		yet

		Parameters
		-------------
		laser : bool
			Whether to filter for laser stimulation events
		'''
		if laser is not None:
			times = [0]
			phaseType = []
			for k, d in self.STM['stim_params'].iteritems():
				for kk, v in d.iteritems():
					if 'duration' in kk:
						times.append(v)
					if 'name' in kk:
						phaseType.append(v)
			periods = np.cumsum(times)
			period_bounds = dict.fromkeys(set(phaseType), [])
			for pk in period_bounds.keys():
				bounds = []
				for k, d in self.STM['stim_params'].iteritems():
					if pk == d['name']:
						idx = int(k.split('_')[1])
						bounds.append(periods[idx-1:idx+1])
				period_bounds[pk] = bounds

			for k, v in period_bounds.iteritems():
				if laser == 0:
					if 'Pause' in k:
						self.posFilter = {'time': np.array(v)}
				elif laser == 1:
					if 'Pause' not in k:
						self.posFilter = {'time': np.array(v)}

	def _getAvailableFiles(self):
		self._available_files = glob(self.filename_root + '*')

	def _getMap(self, tetrode=None, cluster=None, var2bin='pos', binsize=3,
				smooth_sz=5, smooth=True, **kwargs):
		'''

		Returns the ratemap (smoothed or unsmoothed) for a given tetrode and
		cluster

		Parameters
		----------
		tetrode : int
				 the tetrode you want to look at
		cluster : int, 1xn array/ list
				 a single number or list (or 1xn array) of the clusters to plot
		binsize : int, optional
				 size of bins. Defaults to 3
		smooth_sz : int
			the width of the smoothing kernel (see **kwargs for more)
		var2bin : str
			(Optional) Defaults to 'pos'. Which variable to bin. Can be either
			'pos', 'dir' or 'speed'. Works with masked arrays
		smooth : bool, optional.
			Defaults to true. Whether to smooth the data or not
		**kwargs : extra arguments include:
					'gaussian' - the smoothing kernel used is gaussian in shape
					not the default boxcar
					'after' - smoothing of the pos and spike maps is done after
					spikes are divided by pos
					'shuffle' - the time in ms by how much to shift the spikes
					by. Used for generated distributions for null hypothesis
					testing

		Returns
		-------------
		rmap : np.array
			The data binned up as requested
		'''
		if 'pos' in var2bin:
			varType = 'xy'
		else:
			varType = var2bin
		if tetrode is None:
			idx = np.arange(0, self.POS.npos)
			mapType = 'pos'
		else:
			idx = self.TETRODE[tetrode].getClustIdx(cluster)
			mapType = 'rate'
		spk_weights = np.bincount(idx, minlength=self.POS.npos)
		if 'shuffle' in kwargs.keys():
			spk_weights = np.roll(spk_weights, int(kwargs['shuffle']) * 50) # * 50 to go from seconds into pos_samples
		if np.ma.is_masked(self.POS.xy):
			mask = ~np.ma.getmask(self.POS.xy[0])
			pos_weights = mask.astype(np.int)
			self.ratemap.pos_weights = pos_weights
			spk_weights[~mask] = 0
		# Update the ratemap instance with arguments fed into this method
		self.ratemap.binsize = binsize
		self.ratemap.smooth_sz = smooth_sz
		if 'cmsPerBin' in kwargs:
			self.ratemap.cmsPerBin = kwargs['cmsPerBin']
		if 'ppm' in kwargs:
			self.ratemap.ppm = kwargs['ppm']
		rmap = self.ratemap.getMap(spk_weights, varType, mapType, smooth)
		return rmap

	def _getPath(self):
		'''
		Returns
		------------
		self.POS.xy : np.array
			The smoothed xy positions filtered appropriately 
		'''
		if np.ma.is_masked(self.POS.xy):
			return self.POS.xy[:, ~self.POS.xy.mask[0, :]]
		return self.POS.xy

	def _getDir(self):
		'''
		Returns
		------------
		self.POS.dir : np.array
			The smoothed directional data filtered appropriately
		'''
		if np.ma.is_masked(self.POS.dir):
			return self.POS.dir[:, ~self.POS.dir.mask[0, :]]
		return self.POS.dir

	def _getFieldLims(self, tetrode, cluster, binsize=3):
		'''
		Returns a labelled matrix of the ratemap for a given cluster on a given
		tetrode. Binsize can be fractional for smaller bins. Uses anything >
		than the half peak rate to select as a field. Data is heavily smoothed

		Parameters
		---------------
		tetrode : int
			The tetrode to examine
		cluster : int
			The cluster identity

		Returns
		----------
		labelled ratemap and the x and y edges of the binned data as a 3-tuple 
		'''
		rmap, (ye, xe) = self._getMap(tetrode, cluster, binsize=binsize)
		rmap[np.isnan(rmap)] = 0.0
		h = int(np.max(rmap.shape) / 2)
		sm_rmap = self.ratemap.blurImage(rmap, h, ftype='gaussian')
		thresh = np.max(sm_rmap.ravel()) * 0.2  # select area > 20% of peak
		# do some image processing magic to get region to keep as field
		distance = ndimage.distance_transform_edt(sm_rmap > thresh)
		mask = skimage.feature.peak_local_max(distance, indices=False,
											  exclude_border=False,
											  labels=sm_rmap > thresh)
		label = ndimage.label(mask)[0]
		w = skimage.morphology.watershed(-distance, label,
										 mask=sm_rmap > thresh)
		label = ndimage.label(w)[0]
		return label, xe, ye

	def _getClusterPhaseVals(self, tetrode, cluster):
		'''
		Returns the phases of the LFP theta a given cluster fired at

		Parameters
		---------------
		tetrode : int
			The tetrode to examine
		cluster : int
			The cluster identity

		Returns
		----------
		eegphase : np.array
			The phase of theta a cluster fired at
		'''
		ts = self.TETRODE[tetrode].getSpkTS()
		ts = ts / (self.TETRODE[tetrode].timebase / self.EEG.sample_rate)
		ts_idx = np.floor(ts[self.TETRODE[tetrode].cut == cluster]).astype(np.int)
		self.EEG.thetaAmpPhase()
		EEGphase = self.EEG.EEGphase[ts_idx]
		return EEGphase

	def _getThetaCycles(self):
		'''
		Return a tuple of indices into the EEG record that denotes the peaks
		and troughs of theta cycles
		'''
		if not self.EEG:
			self.EEG = EEG(self.filename_root)
		sm_eeg = self.EEG.eegfilter()
		df_eeg = np.diff(sm_eeg)
		pts = np.diff((df_eeg > 0).astype(int), 2)
		pts = ((pts == 1).nonzero()[0]).astype(int)
		peaks = pts[sm_eeg[pts] > 0] + 1
		troughs = pts[sm_eeg[pts] < 0] + 2
		return peaks, troughs

	def _getSpikeInCycle(self, peakIdx, spkIdx=None, whichSpk='first'):
		'''
		given an array of spike indices into eeg and indices of peaks in the
		smoothed, theta-filtered eeg signal this returns the first spike in the
		cycle
		whichSpk can be 'first' or 'last'
		'''
		if 'first' in whichSpk:
			side = 'left'
		elif 'last' in whichSpk:
			side = 'right'
		peaks, troughs = self._getThetaCycles()
		if spkIdx is None:
			spkIdx = self.TETRODE[self.tetrode].getSpkTS()
		spk2eeg_idx = (spkIdx / (self.TETRODE[self.tetrode].timebase /
					   self.EEG.sample_rate)).astype(np.int)
		idx = np.searchsorted(peaks, spk2eeg_idx, side=side)
		uniques, unique_indices = np.unique(idx, return_index=True)
		return spk2eeg_idx[unique_indices]

	def _parseMetaData(self):
		'''
		Parses the filename (mine has a standard format) to populate some of
		the objects properties (self.animal_id, self.trial_num etc)
		'''
		pname, fname = os.path.split(self.filename_root)
		self.metadata['Filename'] = fname
		self.metadata['Path'] = pname
		if 'R' in fname[0]:
			self.metadata['Animal'] = 'Rat'
		else:
			self.metadata['Animal'] = 'Mouse'
		self.metadata['Experimenter'] = fname[-2:]
		self.metadata['Animal_id'] = fname.rsplit('_')[0]
		trial_date = self.setheader['trial_date'] + ':' + self.setheader['trial_time']
		self.metadata['Trial_date'] = datetime.strptime(trial_date,
														'%A, %d %b %Y:%H:%M:%S')
		self.metadata['Trial_num'] = int(fname.rsplit('t')[1][0:-2])

	def _set_figure_title(self, fig, tet, clust):
		fig.canvas.set_window_title('Tetrode: {0} Cluster: {1}'.format(tet, clust))

	def _set_ax_title(self, ax, tet, clust):
		ax.set_title('Tetrode: {0}\nCluster: {1}'.format(tet, clust))

	def klustakwik(self, d):
		"""
		Calls two methods below (kluster and getPC) to run klustakwik on
		a given tetrode with nFet number of features (for the PCA)

		Parameters
		----------
		d : dict
			Specifies the vector of features to be used in
			clustering. Each key is the identity of a tetrode (i.e. 1, 2 etc)
			 and the values are the features used to do the clustering for that tetrode (i.e.
			'PC1', 'PC2', 'Amp' (amplitude) etc
		"""

		legal_values = ['PC1', 'PC2', 'PC3', 'PC4', 'Amp',
						'Vt', 'P', 'T', 'tP', 'tT', 'En', 'Ar']
		reg = re.compile(".*(PC).*")  # check for number of principal comps
		# check for any input errors in whole dictionary first
		for i_tetrode in d.keys():
			for v in d[i_tetrode]:
				if v not in legal_values:
					raise ValueError('Could not find %s in %s' % (v, legal_values))
		# iterate through features and see what the max principal component is
		for i_tetrode in d.keys():
			pcs = [m.group(0) for l in d[i_tetrode] for m in [reg.search(l)] if m]
			waves = self.TETRODE[i_tetrode].waveforms
			princomp = None
			if pcs:
				max_pc = []
				for pc in pcs:
					max_pc.append(int(pc[2]))
				num_pcs = np.max(max_pc)  # get max number of prin comps
				princomp = self.TETRODE[i_tetrode].getParam(waves,
										  param='PCA', fet=num_pcs)
				# Rearrange the output from PCA calc to match the 
				# number of requested principal components
				inds2keep = []
				for m in max_pc:
					inds2keep.append(np.arange((m-1)*4, (m)*4))
				inds2keep = np.hstack(inds2keep)
				princomp = np.take(princomp, inds2keep, axis=1)
			out = []
			for value in d[i_tetrode]:
				if 'PC' not in value:
					out.append(self.TETRODE[i_tetrode].getParam(waves, param=value))
			if princomp is not None:
				out.append(princomp)
			out = np.hstack(out)

			c = Kluster(self.filename_root, i_tetrode, out)
			c.make_fet()
			mask = c.get_mask()
			c.make_fmask(mask)
			c.kluster()

	def getcoherence(self, tetrode, cluster, binsize=3, **kwargs):
		"""
		Wrapper for fieldcalcs.coherence - see docs there
		"""
		smthd = self._getMap(tetrode=tetrode, cluster=cluster, var2bin='pos',
							binsize=binsize, smooth_sz=5,
							smooth=True, **kwargs)

		unsmthd = self._getMap(tetrode=tetrode, cluster=cluster, var2bin='pos',
							binsize=binsize, smooth_sz=5,
							smooth=False, **kwargs)

		return self.fieldcalcs.coherence(smthd[0], unsmthd[0])

	def getkldiv(self, tetrode, cluster, binsize=3, **kwargs):
		"""
		Wrapper for fieldcalcs.kldiv - see there for explanation
		"""
		polarMap = self._getMap(tetrode=tetrode, cluster=cluster, var2bin='dir',
							binsize=binsize, smooth_sz=5,
							smooth=True, **kwargs)
		return self.fieldcalcs.kldiv_dir(polarMap[0])

	def getmrv(self, tetrode, cluster, **kwargs):
		'''
		Calculate the mean resultant vector length and direction for a given
		cluster/ cell

		A wrapper for statscalcs.Statscalcs.mean_resultant_vector (see
		statscalcs.py)

		Parameters
		----------
		tetrode : int
			The tetrode to exmaine
		cluster : int
			The cluster to examine

		Returns
		----------
		r : float
			the mean resultant vector length (range = 0-1)
		th : float
			the mean resultant vector direction (in radians)
		'''

		idx = self.TETRODE[tetrode].getClustIdx(cluster)
		angsInRads = np.deg2rad(self.POS.dir[idx])
		from statscalcs import StatsCalcs
		S = StatsCalcs()
		r, th = S.mean_resultant_vector(angsInRads)
		return r, th

	def getcircR(self, tetrode, cluster, **kwargs):
		'''
		Calculate the mean resultant vector length of circular data
		Unlike getmrv (above) this only returns the vector length. This is
		calculated differently (using complex numbers) but is a) faster, b)
		works with binned data and, c) plays nicer/ easier with shuffles of
		the spike train

		Parameters
		---------------
		tetrode : int
			The tetrode to exmaine
		cluster : int
			The cluster to examine
		**kwargs:
			Legal values of interest:
			shuffle: int
			the number of seconds to shift the spike train

		Returns
		----------
		r : float
			the mean resultant vector length (range = 0-1)
		'''

		idx = self.TETRODE[tetrode].getClustIdx(cluster)
		spk_weights = np.bincount(idx, minlength=self.POS.npos)

		if 'shuffle' in kwargs.keys():
			spk_weights = np.roll(spk_weights, int(kwargs['shuffle'] * 50))
		inc = (np.pi*2) / 120.0
		h = self.ratemap._RateMap__binData(np.deg2rad(self.POS.dir), np.arange(0, np.pi*2+inc, inc), spk_weights)
		from statscalcs import StatsCalcs
		S = StatsCalcs()
		R = S.circ_r(h[1][0][0:-1], h[0])
		return R

	def getskaggsInfo(self, tetrode, cluster, binsize=3, **kwargs):
		'''
		Wrapper for fieldcalcs.skaggsInfo see there for docs

		Parameters
		---------------
		tetrode : int
			The tetrode to exmaine
		cluster : int
			The cluster to examine
		binsize : int
			Size of bins in cms
		Returns
		--------------
		bits per spike : float

		Notes
		-----
		binning could be over any single spatial variable (e.g. location, direction, speed).
		'''
		ratemap = self._getMap(tetrode, cluster, binsize=binsize, **kwargs)[0]
		dwelltimes = self._getMap(binsize=binsize, **kwargs)[0]
		ratemap, _, dwelltimes = self.ratemap._RateMap__adaptiveMap(ratemap, dwelltimes)
		return self.fieldcalcs.skaggsInfo(ratemap, dwelltimes)

	def getTsAndCs(self, verbose=False):
		"""
		Prints out the available tetrodes and clusters
		"""
		cut_files = [(f) for f in glob(self.filename_root + '*') if 'cut' in f]
		m = re.compile('(.*)_(.*).cut', re.M|re.I)
		tAndCdict = {}
		if cut_files:
			for f in cut_files:
				tet = int(m.match(f).group(2))
				try:
					data = self.getCut(tet)
					clusters = list(np.unique(data))
					if clusters[0]==0:
						clusters.pop(0)
						if clusters:
							tAndCdict[tet] = clusters
					if verbose:
						print('\nTetrode {0} contains clusters: {1}'.format(tet, clusters))
				except:
					if verbose:
						print('\nTetrode{0} has no cut'.format(tet))
		else:
			pass
		if tAndCdict:
			tets = []
			clusts = []
			for t,c in tAndCdict.items():
				for cc in c:
					tets.append(str(t))
					clusts.append(str(cc))
			'''
			The two fucking stupid lines below are so yaml can
			serialize the object correctly
			'''
			self.tetrodes = map(int,tets)
			self.clusters = map(int,clusts)
			return tAndCdict

	def plotMap(self, tetrode, clusters, ax=None, var2bin='pos', *args, **kwargs):
		"""
		Plots a ratemap for a given tetrode and cluster
		Wrapper for _plotMap() so multiple clusters can be plotted

		Parameters
		----------
		tetrode : int
				 the tetrode you want to look at
		cluster : int, 1xn array/ list
				 a single number or list (or 1xn array) of the clusters to plot
		ax : optional, defaults to None. Which axis to add the plot to; if None
					then a new figure window is produced
		**kwargs :
			extra arguments include:
			'bar' - for use with directional data to produce a polar
			histogram plot
			'add_peak_rate' - bool
			adds the peak rate (to 2 decimal places) to the figure
			binsize : int, optional
				size of bins. Defaults to 3
			smooth_sz : the width of the smoothing kernel (see **kwargs for more)
				var2bin: optional, defaults to 'pos'. Which variable to bin.
				Can be either 'pos', 'dir' or 'speed'. Works with masked
				arrays
			smooth : bool, optional. Defaults to true. Whether to smooth the data or
				not

		Returns
		-------
		ratemap : numpy.ndarray
			depending on whether a directional (1d) or positional (2d) map was
			asked for an ndarray is returned

		Examples
		--------
		>>> T = dacq2py_util.Trial('M845_141003t1rh')
		>>> # Plot the ratemap for cluster 1 on tetrode 1
		>>> T.plotMap(1,1)
		>>> # Add the peak rate to the figure window
		>>> T.plotMap(1,1,add_peak_rate=True)
		>>> # Plot the polar map for same cluster
		>>> T.plotMap(1,1,var2bin='dir')
		>>> # Plot the unsmoothed dwell map for the trial
		>>> T.plotMap(None,None,smooth=False)
		"""

		for key in ('var2bin', 'ax', 'binsize','smooth_sz', 'smooth'):
			if key in kwargs:
				setattr(self, key, kwargs[key])
		if isinstance(clusters, int):
			setattr(self, 'clusters', [clusters])
		elif isinstance(clusters, list):
			setattr(self, 'clusters', clusters)
		elif isinstance(clusters, str):
			if 'all' in clusters:
				tetDict = self.getTsAndCs()
				setattr(self, 'clusters', tetDict[tetrode])
		clusters = getattr(self, 'clusters', None)
#		var2bin = getattr(self, 'var2bin', 'pos')
		ax = getattr(self, 'ax', None)
		binsize = getattr(self, 'binsize', 3)
		smooth_sz = getattr(self.ratemap, 'smooth_sz', 5)
		smooth = getattr(self, 'smooth', True)

		if len(clusters) == 1:
			ncols = 1
			nrows = 1
		elif np.logical_and(len(clusters) > 1, len(clusters) < 6):
			ncols = len(clusters)
			nrows = 1
		else:
			ncols = 5
			nrows = int(np.floor(len(clusters) / 5) + 1)
		if ax is None:
			fig = plt.figure()
			if 'dir' in var2bin:
				ax = fig.add_subplot(nrows, ncols, 1, projection='polar')
			else:
				ax = fig.add_subplot(nrows, ncols, 1)
		axes_out = []
		if clusters is None:
			axes = fig.add_subplot(1, 1, 1)
			ax, ratemap = self._plotMap(None, None, var2bin=var2bin, ax=ax,
						  binsize=binsize, smooth_sz=smooth_sz, smooth=smooth, *args, **kwargs)
			self._set_ax_title(axes, tetrode, clusters)
			axes_out.append(ax)
		if len(clusters) == 1:
			cluster = clusters[0]

			ax, ratemap = self._plotMap(tetrode=tetrode, cluster=cluster, var2bin=var2bin, ax=ax,
					  binsize=binsize, smooth_sz=smooth_sz, smooth=smooth, *args, **kwargs)
			axes = ax
#			# check kwargs to see if we want to add peak rate to axes
			if "add_peak_rate" in kwargs:
				if kwargs['add_peak_rate']:
					ax.annotate('{:.2f}'.format(np.max(ratemap)), (0.9,0.15), \
							xycoords='figure fraction', textcoords='figure fraction', color='k', size=30, weight='bold', ha='center', va='center')


			self._set_ax_title(axes, tetrode, cluster)
			axes_out.append(ax)
		else:
			fig.set_facecolor('w')
			fig.set_frameon(False)
			plt.setp(ax.get_xticklabels(), visible=False)
			plt.setp(ax.get_yticklabels(), visible=False)
			for iax, cluster in enumerate(clusters):
				inax = fig.add_subplot(nrows, ncols, iax+1)
				ax, ratemap = self._plotMap(tetrode=tetrode, cluster=cluster, var2bin=var2bin,
							  binsize=binsize, smooth_sz=smooth_sz, smooth=smooth,
							  ax=inax)
				self._set_ax_title(inax, tetrode, cluster)
				axes_out.append(ax)
		return axes_out

	def _plotMap(self, tetrode=None, cluster=None, ax=None, var2bin='pos', 
				binsize=3, smooth_sz=5, smooth=True, **kwargs):
		"""
		Plots a ratemap for a given tetrode and cluster

		Parameters
		----------
		tetrode : int
			the tetrode you want to look at
		cluster : int, 1xn array/ list
			a single number or list (or 1xn array) of the clusters to plot
		binsize : int, optional
			size of bins. Defaults to 3
		smooth_sz : int
			the width of the smoothing kernel (see **kwargs for more)
		var2bin : optional, defaults to 'pos'. Which variable to bin.
			Can be either 'pos', 'dir' or 'speed'. Works with masked arrays
		smooth : bool
			Defaults to true. Whether to smooth the data or not
		ax : matplotlib.axes
			Defaults to None. Which axis to add the plot to; if None
			then a new figure window is produced
		**kwargs : various
			'bar' - for use with directional data to produce a polar
			histogram plot

		Returns
		-------
		ratemap: ndarray (1d or 2d)
			depending on whether a directional (1d) or positional (2d) map was
			asked for an ndarray is returned
		"""

		rmap = self._getMap(tetrode=tetrode, cluster=cluster, var2bin=var2bin,
							binsize=binsize, smooth_sz=smooth_sz,
							smooth=smooth, **kwargs)
		if rmap[0].ndim == 1:
			# polar plot
			if ax is None:
				fig = plt.figure()
				self._set_figure_title(fig, tetrode, cluster)
				ax = fig.add_subplot(111, projection='polar')
			theta = np.deg2rad(rmap[1][0][1:])
			ax.clear()
			ax.plot(theta, rmap[0])
			ax.set_aspect('equal')
			ax.tick_params(axis='both', which='both', bottom='off', left='off', right='off', top='off', labelbottom='off', labelleft='off', labeltop='off', labelright='off')
			ax.set_rticks([])
			# deal with vmin/ vmax in kwargs
			if 'vmax' in kwargs.keys():
				ax.set_rmax(kwargs['vmax'])
			# See if we should add the mean resultant vector (mrv)
			if 'add_mrv' in kwargs.keys():
				from statscalcs import StatsCalcs
				S = StatsCalcs()
				idx = self.TETRODE[tetrode].getClustIdx(cluster)
				angles = self.POS.dir[idx]
				print('len angles: {}'.format(len(angles)))
				r, th = S.mean_resultant_vector(np.deg2rad(angles))
				ax.hold(True)
				print('r: {}\nth: {}'.format(r,th))
				ax.plot([th, th],[0, r*np.max(rmap[0])],'r')
			ax.set_thetagrids([0, 90, 180, 270])
			ratemap = rmap[0]

		elif rmap[0].ndim == 2:
			if ax is None:
				fig = plt.figure()
				ax = fig.add_subplot(111)
				self._set_figure_title(fig, tetrode, cluster)
			# mask the ratemap where NaNs occur for plotting purposes
			ratemap = np.ma.MaskedArray(rmap[0], np.isnan(rmap[0]), copy=True)
			x, y = np.meshgrid(rmap[1][1][0:-1], rmap[1][0][0:-1][::-1])
			# deal with vmin/ vmax in kwargs
			if 'vmax' in kwargs.keys():
				vmax = kwargs['vmax']
			else:
				vmax = np.max(np.ravel(ratemap))
			ax.pcolormesh(x, y, ratemap, cmap=cm.jet, edgecolors='face', vmax=vmax)
			ax.axis([x.min(), x.max(), y.min(), y.max()])
			ax.set_aspect('equal')
			plt.setp(ax.get_xticklabels(), visible=False)
			plt.setp(ax.get_yticklabels(), visible=False)
			ax.axes.get_xaxis().set_visible(False)
			ax.axes.get_yaxis().set_visible(False)
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
			ax.spines['bottom'].set_visible(False)
			ax.spines['left'].set_visible(False)
		return ax, ratemap

	def plotPath(self, ax=None, clamp=False, label=False, applyStm=False, **kwargs):
		'''
		Plots the animals path during a trial. Default is to limit plot range
		to the min/ max of x/y extent of path

		Parameters
		----------
		ax : matplotlib.Axes
			The axes to plot into. If none a new figure window is created
		clamp : bool
			whether the axes are clamped to self._xlims and self._ylims or not
		applyStm : bool
			Whether to overlay r crosses on the path where the laser events occurred
		'''
		if ax is None:
			fig = plt.figure()
			ax = fig.add_subplot(111)
		else:
			fig = plt.gcf()
		fig.set_facecolor('w')
		xy = self._getPath()
		ax.plot(xy[0], xy[1], color=[0.8627, 0.8627, 0.8627],**kwargs)
		ax.invert_yaxis()
		if applyStm:
			stmTS = self.STM.getPosTS()
			stmXY = xy[:, stmTS.astype(int)]
			ax.plot(stmXY[0], stmXY[1], 'rx', ms=2)
		if clamp:
			ax.set_xlim(self._xlims)
			ax.set_ylim(self._ylims)
		ax.set_aspect('equal')
		if not label:
			plt.setp(ax.get_xticklabels(), visible=False)
			plt.setp(ax.get_yticklabels(), visible=False)

	def plotSpikesOnPath(self, tetrode, clusters, ax=None, clamp=False, **kwargs):
		'''
		Plots the spikes on the path during a trial for a particular tetrode/
		cluster(s)

		Parameters
		----------
		tetrode: int
				the tetrode you want to look at
		cluster : int, 1xn array/ list
				a single number or list (or 1xn array) of the clusters to plot
		clamp : bool, optional
				whether to restrict the plot to the self._xlims and self_ylims
				property
		ax : matplotlib.Axes
			defaults to None. Which axis to add the plot to.
			If None a new figure window is produced

		'''
		if not isinstance(clusters, (np.ndarray, list)):
			if isinstance(clusters, str):
				clusters = self.availableClusters
			else:
				clusters = [clusters]
		xy = self.POS.xy
		for i, clust in enumerate(clusters):
			if ax is None:
				fig = plt.figure()
				ax = fig.add_subplot(111)
			ax.plot(xy[0], xy[1], c=tcols.colours[0], zorder=1)
			idx = self.TETRODE[tetrode].getClustIdx(clust)
			# useful to override default colour scheme for publication figures
			if 'mec' in kwargs.keys():
				mec = kwargs.pop('mec')
			else:
				mec = tcols.colours[clust]
			ax.plot(xy[0, idx], xy[1, idx], 's', c=mec, mec=mec, **kwargs)
			if clamp:
				ax.set_xlim(self._xlims)
				ax.set_ylim(self._ylims)
			ax.set_aspect('equal')
			ax.invert_yaxis()
			plt.tick_params(axis='both', which='both', left='off', right='off',
							bottom='off', top='off')
			plt.setp(ax.get_xticklabels() + ax.get_yticklabels(),
					 visible=False)
		return ax

	def plotRaster(self, tetrode, clusters, ax=None, dt=(-50, 100), prc_max = 0.5, ms_per_bin=1, histtype='count', hist=True, **kwargs):
		"""
		Wrapper for _plotRaster allowing multiple clusters to be plotted in
		separate figure windows

		Parameters
		----------
		tetrode : int
		cluster : int
		dt : 2-tuple
			the window of time in ms to examine zeroed on the event of interest
			i.e. the first value will probably be negative as in the default example
		prc_max : float
			the proportion of firing the cell has to 'lose' to count as
			silent; a float between 0 and 1
		ax - matplotlib.Axes
			the axes to plot into. If not provided a new figure is created
		ms_per_bin : int
			The number of milliseconds in each bin of the raster plot
		histtype : str
			either 'count' or 'rate' - the resulting histogram plotted above the raster plot will
			consist of either the counts of spikes in ms_per_bin or the mean rate
			in ms_per_bin
		"""
		if isinstance(clusters, int):
			clusters = [clusters]
		elif isinstance(clusters, str):
			if 'all' in clusters:
				tetDict = self.getTsAndCs()
				clusters = tetDict[tetrode]
		for cluster in clusters:
			# Calculate the stimulation ratio
			stim_histo = self.getRasterHist(tetrode, cluster, dt=dt, hist=hist)
			mean_stim_spikes = np.sum(stim_histo, 1)
			pre_stim_spks = np.mean(mean_stim_spikes[0:50])
			post_stim_spks = np.mean(mean_stim_spikes[50:60])
			ratio = (post_stim_spks-pre_stim_spks) / (post_stim_spks+pre_stim_spks)
			print("Stimulation ratio = {}".format(ratio))
			self._plotRaster(tetrode=tetrode, cluster=cluster, dt=dt,prc_max=prc_max, ax=ax, ms_per_bin=ms_per_bin,histtype=histtype, **kwargs)
		return ratio

	def _plotRaster(self, tetrode, cluster, dt=(-50, 100), prc_max=0.5, ax=None, ms_per_bin=1, histtype='count', **kwargs):
		"""
		Plots a raster plot for a specified tetrode/ cluster

		Parameters
		----------
		tetrode : int
		cluster : int
		dt : 2-tuple
			the window of time in ms to examine zeroed on the event of interest
			i.e. the first value will probably be negative as in the default example
		prc_max : float
			the proportion of firing the cell has to 'lose' to count as
			silent; a float between 0 and 1
		ax - matplotlib.Axes
			the axes to plot into. If not provided a new figure is created
		ms_per_bin : int
			The number of milliseconds in each bin of the raster plot
		histtype : str
			either 'count' or 'rate' - the resulting histogram plotted above the raster plot will
			consist of either the counts of spikes in ms_per_bin or the mean rate
			in ms_per_bin
		"""

		if 'x1' in kwargs.keys():
			x1 = kwargs.pop('x1')
		else:
			x1 = self.TETRODE[tetrode].getClustTS(cluster)
			x1 = x1 / int(self.TETRODE[tetrode].timebase / 1000.) #in ms
		x1.sort()
		on_good = self.STM.getTS()
		dt = np.array(dt)
		irange = on_good[:, np.newaxis] + dt[np.newaxis, :]
		dts = np.searchsorted(x1, irange)
		y = []
		x = []
		for i, t in enumerate(dts):
			tmp = x1[t[0]:t[1]] - on_good[i]
			x.extend(tmp)
			y.extend(np.repeat(i, len(tmp)))
		if ax is None:
			fig = plt.figure(figsize=(4.0, 7.0))
			self._set_figure_title(fig, tetrode, cluster)
			axScatter = fig.add_subplot(111)
		else:
			axScatter = ax
		axScatter.scatter(x, y, marker='.', s=2, rasterized=False, **kwargs)
		divider = make_axes_locatable(axScatter)
		axScatter.set_xticks((dt[0], 0, dt[1]))
		axScatter.set_xticklabels((str(dt[0]), '0', str(dt[1])))
		axHistx = divider.append_axes("top", 0.95, pad=0.2, sharex=axScatter,
									  transform=axScatter.transAxes)
		scattTrans = transforms.blended_transform_factory(axScatter.transData,
														  axScatter.transAxes)
		stim_pwidth = int(self.setheader['stim_pwidth'])
		axScatter.add_patch(Rectangle((0, 0), width=stim_pwidth/1000., height=1,
							transform=scattTrans,
							color=[0, 0, 1], alpha=0.5))
		histTrans = transforms.blended_transform_factory(axHistx.transData,
														 axHistx.transAxes)
		axHistx.add_patch(Rectangle((0, 0), width=stim_pwidth/1000., height=1,
						  transform=histTrans,
						  color=[0, 0, 1], alpha=0.5))
		axScatter.set_ylabel('Laser stimulation events', labelpad=-18.5)
		axScatter.set_xlabel('Time to stimulus onset(ms)')
		nStms = int(self.STM['num_stm_samples'])
		axScatter.set_ylim(0, nStms)
		# Label only the min and max of the y-axis
		ylabels = axScatter.get_yticklabels()
		for i in range(1, len(ylabels)-1):
			ylabels[i].set_visible(False)
		yticks = axScatter.get_yticklines()
		for i in range(1, len(yticks)-1):
			yticks[i].set_visible(False)

		histColor = [192/255.0,192/255.0,192/255.0]
		histX = axHistx.hist(x, bins=np.arange(dt[0], dt[1] + ms_per_bin, ms_per_bin),
							 color=histColor, alpha=0.6, range=dt, rasterized=True, histtype='stepfilled')
		vals = histX[0]
		bins = histX[1]
		if 'rate' in histtype:
			axHistx.set_ylabel('Rate')
			mn_rate_pre_stim = np.mean(vals[bins[1:] < 0])
			idx = np.logical_and(bins[1:] > 0, bins[1:] < 10).nonzero()[0]
			mn_rate_post_stim = np.mean(vals[idx])
			above_half_idx = idx[(vals[idx] < mn_rate_pre_stim * prc_max).nonzero()[0]]
			half_pre_rate_ms = bins[above_half_idx[0]]
			print('\ntime to {0}% of pre-stimulus rate = {1}ms'.format(*(prc_max * 100, half_pre_rate_ms)))
			print('mean pre-laser rate = {0}Hz'.format(mn_rate_pre_stim))
			print('mean 10ms post-laser rate = {0}'.format(mn_rate_post_stim))
		else:
			axHistx.set_ylabel('Spike count', labelpad=-2.5)
		plt.setp(axHistx.get_xticklabels(),
				 visible=False)
		# Label only the min and max of the y-axis
		ylabels = axHistx.get_yticklabels()
		for i in range(1, len(ylabels)-1):
			ylabels[i].set_visible(False)
		yticks = axHistx.get_yticklines()
		for i in range(1, len(yticks)-1):
			yticks[i].set_visible(False)
		axHistx.set_xlim(dt)
		axScatter.set_xlim(dt)

		return x,y

	def getRasterHist(self, tetrode, cluster, dt=(-50, 100), hist=True):
		'''
		Calculates the histogram of the raster of spikes during a series of events

		Parameters
		----------
		tetrode : int
		cluster : int
		dt : tuple
			the window of time in ms to examine zeroed on the event of interest
			i.e. the first value will probably be negative as in the default example
		hist : bool
			not sure
		'''
		x1 = self.TETRODE[tetrode].getClustTS(cluster)
		x1 = x1 / int(self.TETRODE[tetrode].timebase / 1000.) #in ms
		x1.sort()
		on_good = self.STM.getTS()
		dt = np.array(dt)
		irange = on_good[:, np.newaxis] + dt[np.newaxis, :]
		dts = np.searchsorted(x1, irange)
		y = []
		x = []
		for i, t in enumerate(dts):
			tmp = x1[t[0]:t[1]] - on_good[i]
			x.extend(tmp)
			y.extend(np.repeat(i, len(tmp)))

		if hist:
			nEvents = int(self.STM["num_stm_samples"])
			return np.histogram2d(x, y, bins=[np.arange(dt[0],dt[1]+1,1), np.arange(0,nEvents+1, 1)])[0]
		else:
			return np.histogram(x, bins=np.arange(dt[0],dt[1]+1,1), range=dt)[0]

	def plot_event_EEG(self, eeg_type='egf', dt=(-50, 100), plot=True, ax=None, 
					   evenOnsets=True, **kwargs):
		"""
		Plots out the eeg record following an 'on' event in the log file

		Parameters
		----------
		eeg_type : str
			either 'eeg' or 'egf'
		dt : tuple
			time to look before and after an onset event
		plot : bool
			whether to plot the stimulus-triggered-eeg
		ax : matplotlib.axis
			will plot into this axis if supplied
			(new figure produced if plot is None and ax is None)
		evenOnsets: bool
			if True assume there is supposed to be an even 
			difference between the events in the .stm file. If events are 
			found that have an uneven difference they are thrown out.
			NB The difference is calculated from information gleaned from 
			the trial.STM field. If False this is ignored.
		"""
		on_good = self.STM.getTS()#timestamps in ms
		"""
		Check for inter-stimulus time differences to make sure that the large
		majority (99%) of on pulses are regularly spaced - otherwise issue a warning
		"""
		df = np.diff(np.diff(on_good))
		if np.count_nonzero(df) / float(len(on_good)) * 100 > 1:
			warnings.warn('More than 1% of on events differ in size', UserWarning)
		#check for abnormally large number of stim events and abort
		if len(on_good) > 100000:
			raise Exception('Very large number of stimulation events. Aborting plot_event_EEG')
		#get the eeg data and indices to use
		if 'egf' in eeg_type:
			eeg = self.EGF.eeg
			on_idx = self.STM.getEGFIdx()
			eeg_samps_per_ms = self.EGF.sample_rate / 1000.0
		elif 'eeg' in eeg_type:
			eeg = self.EEG.eeg
			on_idx = self.STM.getEEGIdx()
			eeg_samps_per_ms = self.EEG.sample_rate / 1000.0

		"""
		NB the following conditional assumes there is only one phase of the 
		stimulation that actually contains stim events. If there is more than 
		one then the last one will be the one used
		"""
		df = np.diff(on_good)
		"""
		keep pulsePause here as used lower down to plot multiple Rectangle
		patches in case the dt tuple specifies a range of values higher than
		the pause between stimulation events
		"""
		pulsePause = 0
		if evenOnsets:
			for k, v in self.STM.iteritems():
				if isinstance(v, OrderedDict):
					for kk, vv in v.iteritems():
						for kkk, vvv in vv.iteritems():
							if 'Pause' in kkk:
								if vvv is not None:
									pulsePause = vvv
			pulsePause_ms = pulsePause / 1000#this is the desired
			unequalPausesIdx = np.nonzero(df!=pulsePause_ms)[0]
			on_good = np.delete(on_good, unequalPausesIdx)
			on_idx = np.delete(on_idx, unequalPausesIdx)
		eeg = eeg - np.ma.mean(eeg)
		dt_eeg = eeg_samps_per_ms * np.array(dt)
		rng = np.arange(dt_eeg[0], dt_eeg[1], 1)
		idx = (on_idx[np.newaxis, :] + rng[:, np.newaxis]).astype(int)
		result = np.zeros((len(rng), len(on_good)))
		result = eeg[idx]
		if not plot:
			return result, idx
		else:
			mn = np.mean(result, 1)
			se = np.std(result, 1) / np.sqrt(len(on_good))
			if ax is None:
				fig = plt.figure()
				ax = fig.add_subplot(111)
			else:
				ax = ax
			ax.errorbar(np.linspace(dt[0], dt[1], len(mn)), mn * 1e6,
						yerr=se*1e6, rasterized=False)
			ax.set_xlim(dt)
			axTrans = transforms.blended_transform_factory(ax.transData,
														   ax.transAxes)
			stim_pwidth = int(self.setheader['stim_pwidth'])
			if pulsePause > 0:
				a = np.arange(0, dt[1], pulsePause_ms)
				b = np.arange(0, dt[0], -pulsePause_ms)
				patchStarts = np.unique(np.concatenate((a, b)))
			for p in patchStarts:
				ax.add_patch(Rectangle((p, 0), width=stim_pwidth/1000., height=1,
							 transform=axTrans,
							 color=[1, 1, 0], alpha=0.5))
			ax.set_ylabel('LFP ($\mu$V)')
			ax.set_xlabel('Time(ms)')
			return result

	def plotEventEEGRange(self, eeg_type='egf', stimTrials=[0,1], ax=None, **kwargs):
		"""
		Calls plot_event_eeg with defaults and no plotting and then plots out
		a time period in seconds from x1 to x2 and overlays the correct time in
		seconds on the x-axis - meant for manual inspection of the effect of
		stimulation events on the eeg

		Parameters
		------------
		eeg_type : str
			either 'egf' or 'eeg' although probably no point
			using 'eeg' as sample rate too low
		stimTrials : list
			the stimulation 'trial' to plot, starting at 0
			NB stimulating every 150ms for 10ms for 20 minutes gets
			you 8000 trials
		ax : matplotlib.axis
			the axis to plot into. A new figure is
			produced if this is None
		"""

		result, idx = self.plot_event_EEG(eeg_type=eeg_type, plot=False)
		eeg_samp_rate = self.STM[eeg_type + 'SampRate']
		time_ms = idx / float(eeg_samp_rate / 1000.)
		eeg_blocks = []
		time_blocks = []
		for t in stimTrials:
			eeg_blocks.append(result[:, t])
			time_blocks.append(time_ms[:, t])

		speed_idx = (idx / (eeg_samp_rate / self.POS.pos_sample_rate)).astype(int)
		speed = self.POS.speed[0, np.ravel(speed_idx, 'F')]
		max_speed = np.max(speed)
		speed = np.reshape(speed, idx.shape, 'F')
		# filter the eeg data in the theta and gamma bands
		E = EEGCalcs(self.filename_root)
		eeg = self.EGF.eeg
		eeg = eeg - np.ma.mean(eeg)
		sampRate = self.EGF.sample_rate
		theta_eeg = E.filterWithButter(eeg, 4, 8, sampRate, 2)
		gamma_eeg = E.filterWithButter(eeg, 30, 80, sampRate, 2)

		theta = theta_eeg[np.ravel(idx, 'F')]
		theta = np.reshape(theta, idx.shape, 'F')
		gamma = gamma_eeg[np.ravel(idx, 'F')]
		gamma = np.reshape(gamma, idx.shape, 'F')
		#dt is (-50, 150)
		rectStart = int((eeg_samp_rate / 1000.) * 50)
		rectEnd = int((eeg_samp_rate / 1000.) * 60)
		if ax is None:
			fig = plt.figure()
			ax = fig.add_subplot(111)
		else:
			ax = ax
		ax1 = ax.twinx()
		for block in zip(time_blocks, eeg_blocks, stimTrials):
			ax.plot(block[0], block[1], color=[0.8627, 0.8627, 0.8627])
			ax.hold(True)
			ax.plot(block[0], theta[:, block[2]], 'r')
			ax.plot(block[0], gamma[:, block[2]], 'g')
			ax1.plot(block[0], speed[:, block[2]], 'y')
			ax1.set_ylim(0, np.max(max_speed) * 4)
			axTrans = transforms.blended_transform_factory(ax.transData,
														   ax.transAxes)
			i = block[0][rectStart]
			j = block[0][rectEnd] - block[0][rectStart]
			ax.add_patch(Rectangle((i,0), width=j, height=1,
							 transform=axTrans,
							 color=[41./256, 161./256, 230./256], alpha=0.5))
		ax.set_xlim(time_blocks[0][0], time_blocks[-1][-1])
		ylabels = ax1.yaxis.get_majorticklabels()
		for i,xxx in enumerate(ylabels):
			if i > 1:
				xxx.set_visible(False)
			else:
				xxx.set_color('k')
		yticks = ax1.yaxis.get_major_ticks()
		for i,xxx in enumerate(yticks):
			if i > 1:
				xxx.set_visible(False)

	def adjust_median_speed(self, min_speed=5, plot=True):
		'''
		Parameters
		----------
		min_speed : float
		plot : bool
		'''
		grandMedian = stats.nanmedian(self.POS.speed, 1)
		sortedSpIdx = np.argsort(self.POS.speed)
		sortedSp = np.sort(self.POS.speed)
		indMedian = np.nonzero(sortedSp >= grandMedian)[1][0]
		indFirstOverThresh = np.nonzero(sortedSp >= min_speed)[1][0]
		indLastNotNan = np.nonzero(~np.isnan(sortedSp))[1][-1]
		halfWidth = np.min([indMedian-indFirstOverThresh, indLastNotNan-indMedian])
		if plot:
			fig = plt.figure()
			ax = fig.add_subplot(111)
			maxSp = sortedSp[0, indLastNotNan]
			L = sortedSp.shape[1]
			rect = Rectangle(xy=(0, indMedian-halfWidth), width=maxSp, height=indMedian+halfWidth/2, color='b', alpha=0.5)
			ax.add_patch(rect)
			ax.plot(sortedSp[0, 0:indLastNotNan], np.arange(indLastNotNan), 'k', lw=2)
			ax.set_xlabel('Speed (cm/s)')
			ax.set_ylabel('Cumulative number of samples')
			if indLastNotNan != L:
				ax.plot((0, maxSp), (indLastNotNan+1, indLastNotNan+1), 'r-')
				ax.plot((0, maxSp), (L, L), 'r-')
			ax.set_xlim(0, maxSp)
			ax.set_ylim(0, L)
			ax.plot((0, maxSp), (indMedian, indMedian), 'b', lw=1)
			ax.plot((grandMedian, grandMedian), (0, indMedian), 'b-')
			ax.plot(grandMedian, indMedian, 'bo', ms=12)
			ax.plot((0, maxSp), (indFirstOverThresh, indFirstOverThresh), 'b', lw=1)
			ax.plot((min_speed, min_speed), (0, indFirstOverThresh), 'b--')
			ax.plot(min_speed, indFirstOverThresh, 'bo', ms=12)
		return sortedSpIdx[indMedian-halfWidth:indMedian+halfWidth]

	def plotRateVSpeed(self, tetrode, cluster, minSpeed=0.0, maxSpeed = 40.0, 
					   sigma=3.0, shuffle=False, nShuffles=100, plot=False, ax=None,
					   verbose=False, getShuffledData=False, getData=False, **kwargs):
		'''
		Plots the instantaneous firing rate of a cell against running speed
		Also outputs a couple of measures as with Kropff et al., 2015; the
		Pearsons correlation and the depth of modulation (dom) - see below for
		details

		Parameters
		-------------------
		tetrode : int
			the tetrode to use
		cluster : int
			the cluster to use
		minSpeed : float
			speeds below this value are masked and not used
		maxSpeed : float
			speeds above this value are masked and not used
		sigma : float
			the standard deviation of the gaussian used to smooth the spike
			train
		shuffle : bool, default False
			Whether to calculate the significance of the speed score or not
			This is done by calculating the correlation between speed and
			the shuffled spike train for nShuffles where the shuffles are only allowed with the
			window (trial_start + minTime) : (trial_end - minTime). Default is
			30 seconds as with Kropff et al., 2015. Default False
		nShuffles : int
			How many times to perform the shuffle. Defaults to 100 as with
			Kropff et al., 2015
		plot : bool
			Whether to plot output or not. Defaults to False
		'''

		speed = self.POS.speed.ravel()
		# Calculate histogram to see how much is accounted for in each bin
		if np.nanmax(speed) < maxSpeed:
			maxSpeed = np.nanmax(speed)
			if verbose:
				print('Capping speed to max in data: {:.2f}'.format(maxSpeed))
		spd_bins = np.arange(minSpeed, maxSpeed, 1.0)
		# Construct the mask
		speed_filt = np.ma.MaskedArray(speed)
		speed_filt = np.ma.masked_where(speed_filt < minSpeed, speed_filt)
		speed_filt = np.ma.masked_where(speed_filt > maxSpeed, speed_filt)
		spk_sm = self._getTimeSmoothedSpikes(tetrode, cluster, sigma)
		spk_sm = np.ma.MaskedArray(spk_sm, mask=np.ma.getmask(speed_filt))

		# res is the basic correlation between running speed and instantaneous
		# firing rate
		res = stats.mstats.pearsonr(spk_sm, speed_filt)
		if shuffle:
			duration = self.POS.npos / self.POS.pos_sample_rate
			shuffles = np.linspace(30, duration-30, nShuffles)
			shuffled_rs = []
			for time in shuffles:
				shuffled_spks = self._getTimeSmoothedSpikes(tetrode, cluster, sigma, time)
				shuffled_rs.append(stats.mstats.pearsonr(shuffled_spks, speed_filt)[0])
			prob = np.array([.90, .95, .99])
			qtiles = stats.mstats.mquantiles(shuffled_rs, prob)
			if verbose:
				print("Running speed vs firing rate correlation (PPMC): {0}".format(res[0]))
				print("The {0} percentiles are {1}".format(prob*100, qtiles))
		spd_dig  = np.digitize(speed_filt, spd_bins, right=True)
		mn_rate = np.array([np.ma.mean(spk_sm[spd_dig==i]) for i in range(0,len(spd_bins))])
		if plot:
			if ax is None:
				fig = plt.figure()
				ax = fig.add_subplot(111)
			ax.plot(spd_bins, mn_rate * self.POS.pos_sample_rate, 'k')
			ax.set_xlim(spd_bins[0], spd_bins[-1])
			ax.set_ylabel("Firing rate(Hz)")
			ax.set_xlabel("Speed(cm/s)")
			ylabels = ax.get_yticklabels()
			for i in range(1, len(ylabels)-1):
				ylabels[i].set_visible(False)
			yticks = ax.get_yticklines()
			for i in range(1, len(yticks)-1):
				yticks[i].set_visible(False)
			xlabels = ax.get_xticklabels()
			for i in range(1, len(xlabels)-1):
				xlabels[i].set_visible(False)
			xticks = ax.get_xticklines()
			for i in range(1, len(xticks)-1):
				xticks[i].set_visible(False)
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
			ax.yaxis.set_ticks_position('left')
			ax.xaxis.set_ticks_position('bottom')
			if "add_peak_rate" in kwargs:
				if kwargs['add_peak_rate']:
					ax.annotate('{:.2f}'.format(np.max(res[0])), (0.15,0.9), \
							xycoords='axes fraction', textcoords='axes fraction', color='k', size=30, weight='bold', ha='center', va='center')

		if getData:
			return res[0], spd_bins, mn_rate * self.POS.pos_sample_rate
		if getShuffledData:
			return res[0], shuffled_rs
		else:
			return res[0]

	def plotRollingCorrRateVSpeed(self, tetrode, cluster, minSpeed=2.0,
								  sigma=3.0, **kwargs):
		'''
		Plots the rolling correlation of instantaneous firing rate of a given
		cell against running speed

		Parameters
		----------
		tetrode : int
		cluster : int
		minSpeed : float
		sigma : float
			The width of the smoothing kernel applied to the spike train to smooth it
		'''
		speed_filt = self.POS.speed.ravel()
		#filter for low speeds
		lowSpeedIdx = speed_filt < minSpeed
		spk_sm = self._getTimeSmoothedSpikes(tetrode, cluster, sigma)
		windowSize = 50
		runningCorr = np.ones_like(spk_sm)
		for i in range(len(spk_sm)):
			runningCorr[i] = stats.pearsonr(spk_sm[i:i+windowSize],
											  speed_filt[i:i+windowSize])[0]
		speed_filt = np.ma.MaskedArray(speed_filt, lowSpeedIdx)
		spk_sm = np.ma.MaskedArray(spk_sm, lowSpeedIdx)
		# mask the running correlation where there is no rate (ie the cell fails
		# to fire)
		new_mask = np.ma.mask_or(lowSpeedIdx, spk_sm==0)
		runningCorr = np.ma.MaskedArray(runningCorr, new_mask)
		fig, ax = plt.subplots()
		fig.subplots_adjust(right=0.75)
		ax2 = ax.twinx()
		ax3 = ax.twinx()
		ax2.spines["right"].set_position(("axes", 1.2))
		ax3.set_frame_on(True)
		ax3.patch.set_visible(False)
		for sp in ax.spines.values():
			sp.set_visible(False)
		ax3.spines["right"].set_visible(True)

		p1, = ax.plot(speed_filt, 'b')
		p2, = ax2.plot(spk_sm, 'r')
		p3, = ax3.plot(runningCorr, 'k')

		ax.set_xlim(0, len(speed_filt))
		ax.set_ylim(0, np.max(speed_filt))
		ax2.set_ylim(0, np.max(spk_sm))
		ax3.set_ylim(-1, 1)

		ax.set_ylabel('Speed(cm/s)')
		ax2.set_ylabel('Instantaneous firing rate(Hz)')
		ax3.set_ylabel('Running correlation')

		ax.yaxis.label.set_color(p1.get_color())
		ax2.yaxis.label.set_color(p2.get_color())
		ax3.yaxis.label.set_color(p3.get_color())

		tkw = dict(size=4, width=1.5)
		ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
		ax2.tick_params(axis='y', colors=p2.get_color(), **tkw)
		ax3.tick_params(axis='y', colors=p3.get_color(), **tkw)
		ax.tick_params(axis='x', **tkw)


	def _getTimeSmoothedSpikes(self, tetrode, cluster, sigma=3.0, shuffle=None):
		'''
		Returns a spike train the same length as num pos samples that has been
		smoothed in time with a gaussian kernel M in width and standard deviation
		equal to sigma

		Parameters
		--------------
		tetrode : int
			the tetrode to use
		cluster : int
			the cluster to use
		sigma : float
			the standard deviation of the gaussian used to smooth the spike
			train
		'''

		x1 = self.TETRODE[tetrode].getClustIdx(cluster)
		spk_sm = self.spikecalcs.smoothSpikePosCount(x1, self.POS.npos, sigma, shuffle)
		return spk_sm

	def plotFreqVSpeed(self, minSp=5, maxSp=50, spStep=5, ax=None, laserFilter=None, **kwargs):
		'''
		Plots running speed vs eeg frequencies and does linear regression. Also adds position sample histogram
		TODO: filter out negative frequencies - do this as default in EEG class
		Parameters
		----------
		minSp : int
			speeds below this are ignored
		maxSp : int
			speeds above this are ignored
		spStep : int
			the bin width for speed
		ax : matplotlib.axes
			the axes in which to plot
		laser : int or None
			whether to filter for laser on/ off events
			None means no filtering at all
			1 means laser is on and data is filtered for on periods
			0 means filter for laser off periods

		'''

		sp = np.ma.compressed(self.POS.speed)
		if laserFilter:
			eeg = self.EEG.eeg
			EE = EEGCalcs(self.filename_root, thetaRange=[6,12])
			if 'dip' in kwargs:
				d = kwargs['dip']
			else:
				d = 15.0
			if 'width' in kwargs:
				w = kwargs['width']
			else:
				w = 0.125
			if 'stimFreq' in kwargs:
				sf = kwargs['stimFreq']
			else:
				sf = 6.66
			fx = EE.filterForLaser(E=eeg, width=w, dip=d, stimFreq=sf)#filters out laser stimulation artifact
			fxx = self.EEG.eegfilter(fx)
			self.EEG.thetaAmpPhase(fxx)#filters for theta
			freq = self.EEG.EEGinstfreq
		else:
			try:
				freq = self.EEG.EEGinstfreq
			except:
				self.EEG.thetaAmpPhase()
				freq = self.EEG.EEGinstfreq
		freq[freq<0] = np.nan
		sp_bins = np.arange(minSp, maxSp, spStep)
		sp_dig = np.digitize(sp, sp_bins)
		freq = np.reshape(freq, (self.POS.npos, self.EEG.sample_rate/self.POS.pos_sample_rate))
		if np.ma.is_masked(self.POS.speed):
			mask = np.ma.getmask(self.POS.speed)
			mask = np.tile(mask.T, self.EEG.sample_rate/self.POS.pos_sample_rate)
			freq = np.ma.MaskedArray(freq, mask=mask)
		mn_freq = np.nanmean(freq, 1)
		mn_freq = np.ma.compressed(mn_freq)
		X = [mn_freq[sp_dig==i] for i in range(len(sp_bins))]
		# remove any nans which will screw plt.boxplots ability to calculate means
		# and do the boxplot correctly
		for i,x in enumerate(X):
			idx = ~np.isfinite(x)
			X[i] = np.delete(x,np.nonzero(idx))
		if ax is None:
			fig = plt.figure()
			fig.set_facecolor('w')
			ax = plt.gca()
		else:
			fig = plt.gcf()
			fig.set_facecolor('w')
		# set up some properties for the elements in the box plot
		bprops = {'c': [0.8627, 0.8627, 0.8627]}
		wprops = {'c': [0.8627, 0.8627, 0.8627]}
		ax.boxplot(X, positions=sp_bins, boxprops=bprops, whiskerprops=wprops)
		medians = np.array([stats.nanmedian(x) for x in X])
		nan_idx = np.isnan(medians)
		slope, intercept, r_value, p_value, std_err = stats.linregress(sp_bins[~nan_idx], medians[~nan_idx])
		minFreq = np.min(medians[~nan_idx]) - 1.0
		maxFreq = np.max(medians[~nan_idx]) + 1.0
		ax.set_ylim(minFreq, maxFreq)
#        ax.set_xlim(0, sp_bins[-1])
#		ylims = np.array(ax.get_ylim())
		xlims = np.array(ax.get_xlim())
		res = stats.theilslopes(medians[~nan_idx], sp_bins[~nan_idx], 0.90)
		ax.plot([0,xlims[1]], (res[1], res[1] + (res[0] * sp_bins[-1])), 'r-')
		ax.plot([0,xlims[1]], (res[1], res[1] + (res[2] * sp_bins[-1])), 'r--')
		ax.plot([0,xlims[1]], (res[1], res[1] + (res[3] * sp_bins[-1])), 'r--')
#        ax.plot([0,xlims[1]], (intercept, intercept + (sp_bins[-1] * slope)), 'k--', lw=2)
		ax.set_ylabel('Frequency(Hz)')
		ax.set_xlabel('Speed (cm/s)')
		ax.set_title('Intercept: {0:.3f}    Slope: {1:.5f}'.format(intercept, slope))
		# add the right-hand y-axis and format
		ax1 = ax.twinx()
		# get a histogram of speed to be plotted against the right-hand y-axis
		h,e = np.histogram(np.ma.compressed(sp), bins=len(sp_bins)*10, range=(0, sp_bins[-1]))
		ax1.bar(e[0:-1], h, color=[0.6667, 0.6667, 0], linewidth=0, align='edge')
		ax1.set_ylim(0, np.max(h) * 4) # reduce the 'height' of the secondary plot
#        ax1.set_xlim(0, sp_bins[-1]+spStep)
		ax1.set_ylabel('Position samples', color=[0.6667, 0.6667, 0])
		ax1.yaxis.set_label_coords(1.1,.15)
		ylabels = ax1.yaxis.get_majorticklabels()
		for i,xxx in enumerate(ylabels):
			if i > 1:
				xxx.set_visible(False)
			else:
				xxx.set_color([0.6667, 0.6667, 0])
		yticks = ax1.yaxis.get_major_ticks()
		for i,xxx in enumerate(yticks):
			if i > 1:
				xxx.set_visible(False)
		return ax, intercept, slope

	def plotPhaseOfFiring(self, tetrode, cluster, ax=None, **kwargs):
		"""
		Plots the phase of firing of a given cluster as a histogram

		Parameters
		----------
		tetrode : int
		cluster : int
		ax : matplotlib.Axes
		"""

		phase = self._getClusterPhaseVals(tetrode, cluster)
		if ax is None:
			fig = plt.figure()
			ax = fig.add_subplot(211)
			ax2 = fig.add_subplot(212)
		# make the plot like the Somogyi figures!
		fig.set_facecolor('#203C8A')
		phase = np.hstack((phase, phase + (2*np.pi)))
		ax2.hist(phase, bins=120, range=(-np.pi, 3*np.pi), color='w', histtype='stepfilled')
		t = np.arange(-np.pi, 3 * np.pi, 0.1)
		ax.plot(t, np.sin(t), 'w')
		ax.annotate('180', xy=(-np.pi-0.2, 0), xycoords='data', ha='right', va='center',
					color='w', fontsize=20)
		ax.set_axis_bgcolor('#203C8A')
		ax.set_ylim(-1.1, 1.1)
		ax.axis('off')
		ax2.set_axis_bgcolor('#203C8A')
		plt.axis('off')

	def plotPhaseInField(self, tetrode, cluster, ax=None, **kwargs):
		'''
		Plots theta phase of spikes in a place field (found using _getFieldLims)
		as individual colours for each run through the field
		TODO: broken
		Parameters
		----------
		tetrode : int
		cluster : int
		ax : matplotlib.Axes
		'''
		if not self.EEG:
			self.EEG = EEG(self.filename_root)
		self.EEG.thetaAmpPhase()
		self.EEG.EEGphase = np.rad2deg(self.EEG.EEGphase)
		runs_to_keep, spk_in_run, run_duration = self.getFieldRuns(tetrode, cluster)
		if ax is None:
			ax = plt.gca()
		else:
			ax = ax
		for spks in spk_in_run:
			ax.plot(self.POS.xy[0,spks], self.EEG.EEGphase[spks * self.pos2eegScale]+180,'.')
		ax.set_title(self.filename_root.split('\\')[-1] + ' cluster ' + str(cluster) + ' on tetrode ' + str(tetrode))
		plt.show()

	def plotSpectrogram(self, eegType='eeg', ymin=0, ymax=50, ax=None, secsPerBin=2,
						laser=False, width=0.125, dip=15.0):
		'''
		Plots a spectrogram of the LFP of the whole trial

		Parameters
		--------------
		eegType : str
			Whether to do use .eeg file or .egf file. Defaults to eeg
		ymin / ymax : int
			Minimum/ maximum frequency (y-axis) to plot
		ax : matplotlib.pyplot.axis]
			Which axis to add the plot to. If None a new figure window is produced
		secsPerBin : int
			Size of the x-axis bins
		laser : bool
			Whether to filter the eeg for laser stimulation events
		width/ dip : float
			Parameters for Kaisser filter in eegcalcs.EEGCalcs - see there
			for definition

		Returns
		------------
		Plots the spectrogram
		'''

		if 'eeg' in eegType:
			E = self.EEG.eeg
			if np.ma.is_masked(E):
				E = E.compressed()
			Fs = self.EEG.sample_rate
		elif 'egf' in eegType:
			E = self.EGF.eeg
			if np.ma.is_masked(E):
				E = E.compressed()
			Fs = self.EGF.sample_rate

		EE = EEGCalcs(self.filename_root,thetaRange=[6,12])
		if laser:
			'''
			Split the eeg into the parts where the laser is on and off
			and then reassemble for the spectrogram
			NB this assumes the laser comes on at 600s for 20 minutes
			and then goes off
			'''
			mask = np.ones_like(E).astype(bool)

			mask[600*int(Fs):1800*int(Fs)] = False
			# filter
#			import pdb
#			pdb.set_trace()
			fx = EE.filterForLaser(E=E[~mask], width=width, dip=dip)
			# reassemble
			Etmp = np.zeros_like(E)
			Etmp[~mask] = fx
			Etmp[mask] = E[mask]
			fx = Etmp

		else:
			fx = E
		nperseg = int(Fs * secsPerBin)
		freqs, times, Sxx = signal.spectrogram(fx, Fs, nperseg=nperseg)
#		Sxx_sm = self.ratemap.blurImage(Sxx, (secsPerBin*2)+1)
		Sxx_sm = Sxx
		x, y = np.meshgrid(times, freqs)
		if ax is None:
			plt.figure()
			ax = plt.gca()
			im = ax.pcolormesh(x, y, Sxx_sm, edgecolors='face', cmap='RdBu',norm=colors.LogNorm())
		im = ax.pcolormesh(x, y, Sxx_sm, edgecolors='face', norm=colors.LogNorm())
		ax.set_xlim(times[0], times[-1])
		ax.set_ylim(ymin, ymax)
		ax.set_xlabel('Time(s)')
		ax.set_ylabel('Frequency(Hz)')
		if laser:
			ax.vlines(600,ymin,ymax)
			ax.vlines(1800,ymin,ymax)

			ax.set_xticks((0, 600, 1800, 2400))
			ax.set_xticklabels((str(0), str(600), str(1800), str(2400)))
		return freqs, times, Sxx, im

	def plotEEGPower(self, E=None, eegType='eeg', smthKernelSigma=0.1875,
					freqBand=(6,12), outsideBand=(3,125), s2nWdth=2, xmax=125, 
					ymax=None, plot=True, ax=None, **kwargs):
		'''
		Plots the eeg power spectrum. Annotates graph around theta frequency band.

		Parameters
		-------------
		E : numpy.array
			(Optional) numEEGSamples sized numpy array of raw eeg signal amplitude.
		eegType : str
			(Optional) Either 'eeg' or 'egf'. The .eeg file type to use. Defaults to 'eeg'
		smthKernelSigma : float
			(Optional) number of points in the output window for gaussian filtering of eeg. This
			value is multipled by the binsPerHz which comes from the length of the fft (derived from nextpow2 for speed).
		freqBand : two-tuple
			(Optional) the theta-band to examine.
		outsideBand : two-tuple
			(Optional): frequencies outside these values are ignored. NOT IMPLEMENTED.
		s2nWdth : int
			(Optional) Determines the width of the window to calculate the signal-to-noise ratio.
		xmax : int
			(Optional) Maximum x-value (frequency) to plot to. Defaults to 125
		ymax : int
			(Optional) Maximum y-value to plot to. Defaults to None so plots full range
		plot : bool
			(Optional) Whether to produce a plot
		ax : matplotlib.pyplot.axis instance
			(Optional) The axis to plot in to.

		Returns
		-------------
		ax : matplotlib.pyplot.axis instance
			The axis containing the plot.
		'''

		if E is None:
			if 'eeg' in eegType:
				E = self.EEG.eeg
				freqBand = (self.EEG.x1, self.EEG.x2)
				if np.ma.is_masked(E):
					E = E.compressed()
				sample_rate = self.EEG.sample_rate
			elif 'egf' in eegType:
				E = self.EGF.eeg
				freqBand = (self.EEG.x1, self.EEG.x2)
				if np.ma.is_masked(E):
					E = E.compressed()
				sample_rate = self.EGF.sample_rate
		else:
			if np.ma.is_masked(E):
				E = E.compressed()
			sample_rate = kwargs['sample_rate']
		nqLim = 0
		nqLim = sample_rate / 2
		origLength = len(E)
		fftLength = 2 ** self.EEG.nextpow2(origLength).astype(int)
		freqs, power = signal.periodogram(E, fs=sample_rate, return_onesided=True, nfft=fftLength)
		fftHalfLength = fftLength / 2+1
		# calculate the number of points in the gaussian window - gleaned from gaussian_filter1d
		# which lives in scipy/ndimage/filters.py
		binsPerHz = (fftHalfLength-1) / nqLim
		kernelSigma = smthKernelSigma * binsPerHz
		smthKernelWidth = 2 * int(4.0 * kernelSigma + 0.5) + 1
		gaussWin = signal.gaussian(smthKernelWidth, kernelSigma)
		# smooth the power
		sm_power = signal.fftconvolve(power, gaussWin, 'same')
		# normalize the smoothed power by the length of the fft
		sm_power = sm_power / np.sqrt(len(sm_power))
		# calculate some metrics
		spectrumMaskBand = np.logical_and(freqs>freqBand[0], freqs<freqBand[1])
		bandMaxPower = np.max(sm_power[spectrumMaskBand])
		maxBinInBand = np.argmax(sm_power[spectrumMaskBand])
		bandFreqs = freqs[spectrumMaskBand]
		freqAtBandMaxPower = bandFreqs[maxBinInBand]
		# find power in windows around peak, divide by power in rest of spectrum
		# to get SNR
		spectrumMaskPeak = np.logical_and(freqs>freqAtBandMaxPower-s2nWdth/2, freqs < freqAtBandMaxPower + s2nWdth/2)
		snr = np.nanmean(sm_power[spectrumMaskPeak]) / np.nanmean(sm_power[~spectrumMaskPeak])
		# collect all the following keywords into a dict for output
		dictKeys = ('sm_power','freqs', 'spectrumMaskPeak', 'power','freqBand',
		'freqAtBandMaxPower', 'bandMaxPower', 'xmax', 'ymax', 'snr', 'kernelSigma', 'binsPerHz')
		outDict = dict.fromkeys(dictKeys,np.nan)
		for thiskey in outDict.keys():
			outDict[thiskey] = locals()[thiskey]# neat trick: locals is a dict that holds all locally scoped variables
		if plot:
			if ax is None:
				plt.figure()
				ax = plt.gca()
			ax.plot(freqs, power, alpha=0.5, color=[0.8627, 0.8627, 0.8627])
			# ax.hold(1)
			ax.plot(freqs, sm_power)
			r = Rectangle((freqBand[0],0), width=np.diff(freqBand)[0], height=np.diff(ax.get_ylim())[0], alpha=0.25, color='r', ec='none')
			ax.add_patch(r)
			ax.set_xlim(0,xmax)
			ax.set_ylim(0, bandMaxPower / 0.8)
			ax.set_xlabel('Frequency')
			ax.set_ylabel('Power')
			ax.text(x = freqBand[1] / 0.9, y = bandMaxPower, s = str(freqAtBandMaxPower)[0:4], fontsize=20)
		return ax

	def plotClusterSpace(self, tetrode, clusters=None, ax=None, bins=256,**kwargs):
		'''
		Plots the cluster space for the given tetrode

		Parameters
		----------
		tetrode : int
			the tetrode cluster space to plot
		clusters : int or list or np.array
			the clusters to colour in
		ax : matplotlib.pyplot.axis
			the axis to plot into
		bins : int
			the number of bins to use in the histogram
		**kwargs :
			can include a param keyword for the parameter to construct the
			histogram from - this defaults to amplitude ('Amp') but can be any
			valid key in the getParam method of the Tetrode class

		Returns
		-------
		fig: handle to figure window
		'''

		if clusters is not None and not isinstance(clusters, (np.ndarray, list)):
			clusters = [clusters]  # ie needs to be iterable
		waves = self.TETRODE[tetrode].waveforms
		if self.TETRODE[tetrode].volts:
			waves = (waves * 128) / self.TETRODE[tetrode].scaling[:, np.newaxis]
			waves = waves.astype(int)
		cutfile = self.TETRODE[tetrode].cut

		if cutfile is not None:
			cutfile = np.array(cutfile)
		if 'param' in kwargs.keys():
			param = kwargs['param']
		else:
			param = 'Amp'
		amps = self.TETRODE[tetrode].getParam(waves, param=param)
		bad_electrodes = np.setdiff1d(np.array(range(4)),np.array(np.sum(amps,0).nonzero())[0])
		cmap = np.tile(tcols.colours[0],(bins,1))
		cmap[0] = (1,1,1)
		cmap = colors.ListedColormap(cmap)
		cmap._init()
		alpha_vals = np.ones(cmap.N+3)
		alpha_vals[0] = 0
		cmap._lut[:,-1] = alpha_vals
		cmb = combinations(range(4),2)
		if 'figure' in kwargs.keys():
			fig = kwargs.pop('figure')
		else:
			fig = plt.figure()
		if ax is None:
			ax = fig.add_subplot(111)
		else:
			ax = ax
		ax.axis('off')
#        fig = plt.gcf()
		rect = ax.get_position().bounds
		grid = ImageGrid(fig, rect, nrows_ncols= (2,3), axes_pad=0.1)
		if 'Amp' in param:
			myRange = [[0,256],[0,256]]
		else:
			myRange = None
		for i, c in enumerate(cmb):
			if c not in bad_electrodes:
				H = np.histogram2d(amps[:,c[0]], amps[:,c[1]], range = myRange, bins=bins)
				grid[i].imshow(H[0], cmap=cmap, interpolation='nearest')
				if clusters is not None:
					for thisclust in clusters:
						if 'clustColour' in kwargs.keys():
							clustColour = kwargs['clustColour']
						else:
							clustColour = tcols.colours[thisclust]
						clustidx = (cutfile==thisclust).nonzero()[0]
						H = np.histogram2d(amps[clustidx,c[0]],amps[clustidx,c[1]], range=myRange, bins=bins)
						H = H[0]
						H = signal.convolve2d(H, np.ones((3, 3)), mode='same')
						clustCMap = np.tile(clustColour,(bins,1))
						clustCMap[0] = (1,1,1)
						clustCMap = colors.ListedColormap(clustCMap)
						clustCMap._init()
						clustCMap._lut[:,-1] = alpha_vals
						grid[i].imshow(H, cmap=clustCMap, interpolation='nearest')
			s = str(c[0]+1) + ' v ' + str(c[1]+1)
			grid[i].text(0.05,0.95, s, va='top', ha='left', size='small', color='k', transform=grid[i].transAxes)
			grid[i].set_xlim([0,bins])
			grid[i].set_ylim([0,bins])
			grid[i].tick_params(axis='both', which='both', left='off', right='off',
							bottom='off', top='off')
		plt.setp([a.get_xticklabels() for a in grid], visible=False)
		plt.setp([a.get_yticklabels() for a in grid], visible=False)
		return fig

	def plotXCorr(self, tetrode, clusters, ax=None, Trange=(-500,500), bins=None, annotate=True, **kwargs):
		'''
		Plots the temporal autocorrelogram (defaults to +/- 500ms)
		TODO: needs to be able to take in two tetrodes & make sure Trange in ms

		Parameters
		----------
		tetrode : int
		clusters : int or list
		ax : matplotlib.Axes
			The axes to plot into. If None a new figure window is created
		TRange : two-tuple
			The range over which to examine the events. Zero time is the occurance of the event
		bins : int
			The number of bins to assign the data to
		annotate : bool
			Whether to add the cluster identities to the figure axis
		**kwargs
			if 'add_peak_rate' is in the kwargs then that is also added to the axes
		'''
		if isinstance(clusters, (np.ndarray, list, int)):
			clusters = [clusters]
		if isinstance(tetrode, (np.ndarray, list, int)):
			tetrode = [tetrode]
		duration = np.diff(Trange)
		if bins is None:
			bins = 201
		if ax is None:
			fig = plt.figure()
			ax = fig.add_subplot(111)
		if len(clusters) == 1:
			cluster_a = cluster_b = clusters[0]
		elif len(clusters) == 2:
			cluster_a = clusters[0]
			cluster_b = clusters[1]
		if len(tetrode) == 1:
			tetrode_a = tetrode[0]
			tetrode_b = None
		elif len(tetrode) == 2:
			tetrode_a = tetrode[0]
			tetrode_b = tetrode[1]
		Trange = np.array(Trange)
		timebase = self.TETRODE[tetrode_a].timebase
		x1 = self.TETRODE[tetrode_a].getClustTS(cluster_a) / (timebase/1000)
		if tetrode_b is None:
			if cluster_b is None:
				x2 = x1
				cluster_b = cluster_a
			else:
				x2 = self.TETRODE[tetrode_a].getClustTS(cluster_b) / (timebase/1000)
		else:
			x2 = self.TETRODE[tetrode_b].getClustTS(cluster_b) / (timebase/1000)
		if self.posFilter:
			idx = np.nonzero(~self.POS.xy.mask[0])[0] # indices to keep
			x1PosSamp = (x1 / (1000 / self.POS.pos_sample_rate)).astype(int)
			x1 = x1[np.in1d(x1PosSamp, idx)]
			if cluster_b is not None:
				x2PosSamp = (x2 / (1000 / self.POS.pos_sample_rate)).astype(int)
				x2 = x2[np.in1d(x2PosSamp, idx)]
		y = self.spikecalcs.xcorr(x1, x2, Trange=Trange)
		h = ax.hist(y[y != 0], bins=bins, range=Trange, color='k', histtype='stepfilled')
		ax.set_xlim(Trange)
		if annotate:
			if cluster_b is None:
				cond_rate = np.count_nonzero(y == 0) / np.float(duration)
				ax.text(0.55, .9, "{0:.4}".format(str(cond_rate)), ha='center', va='center',
						transform=ax.transAxes)
			else:
				if np.logical_or((tetrode_a == tetrode_b), tetrode_b is None):
					if (cluster_a == cluster_b):
						#autocorr being done so get theta modulation
						modIdx = self.spikecalcs.thetaModIdx(x1)
						ax.set_title('Cluster {0} vs Cluster {1}\ntheta modulation: {2:.4f}'.format(cluster_a, cluster_b, modIdx))
						if "add_peak_rate" in kwargs:
							if kwargs['add_peak_rate']:
								ax.annotate('{:.2f}'.format(np.max(modIdx)), (0.15,0.9), \
										xycoords='axes fraction', textcoords='axes fraction', color='k', size=30, weight='bold', ha='center', va='center')

	#                    ax.set_title('Cluster ' + str(cluster_a) + ' vs Cluster ' + str(cluster_b) +'\ntheta modulation=' + str(modIdx))
				else:
					ax.set_title('Cluster ' + str(cluster_a) + ' vs Cluster ' + str(cluster_b))
		ax.set_xlabel('Time(ms)')
		ax.set_xticks((Trange[0], 0, Trange[1]))
		ax.set_xticklabels((str(Trange[0]), '0', str(Trange[1])))
		ax.tick_params(axis='both', which='both', left='off', right='off',
							bottom='off', top='off')
		ax.set_yticklabels('')
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.spines['left'].set_visible(False)
		ax.xaxis.set_ticks_position('bottom')
		return ax, h

	def getThetaModIdx(self, tetrode, cluster):
		'''
		Calculates the theta modulation index of a clusters autocorrelogram
		as the difference between the first trough and second peak of the
		autocorrelogram (actually the difference over their sum)

		Parameters
		--------------
		tetrode : int
			The tetrode the cluster is on
		cluster : int
			The cluster identity
		Returns
		-------------
		thetaModulation : int
			The depth of theta modulation
		'''
		x1 = self.TETRODE[tetrode].getClustTS(cluster) / float(self.TETRODE[tetrode].timebase) * 1000
		if self.posFilter:
			idx = np.nonzero(~self.POS.xy.mask[0])[0] # indices to keep
			x1PosSamp = (x1 / (1000 / self.POS.pos_sample_rate)).astype(int)
			x1 = x1[np.in1d(x1PosSamp, idx)]
		thetaMod = self.spikecalcs.thetaModIdx(x1)
		return thetaMod

	def getThetaModIdx2(self, tetrode, cluster):
		'''
		Wrapper for thetaModIdxV2 in spikecalcs.py

		Parameters
		--------------
		tetrode : int
			The tetrode the cluster is on
		cluster : int
			The cluster identity
		Returns
		-------------
		thetaModulation : int
			The depth of theta modulation
		'''

		x1 = self.TETRODE[tetrode].getClustTS(cluster) / float(self.TETRODE[tetrode].timebase) * 1000
		if self.posFilter:
			idx = np.nonzero(~self.POS.xy.mask[0])[0] # indices to keep
			x1PosSamp = (x1 / (1000 / self.POS.pos_sample_rate)).astype(int)
			x1 = x1[np.in1d(x1PosSamp, idx)]
		thetaMod = self.spikecalcs.thetaModIdxV2(x1)
		return thetaMod

	def plotWaveforms(self, tetrode, clusters, ax=None, **kwargs):
		"""
		Plots spike waveforms on all four wires for a given tetrode/ cluster
		The units for the plots are *real* in the sense that the x-axis is in
		ms and the y-axis is in micro-volts. The axes limits are set up so the 
		ratio between the x and y axes is 100

		Parameters
		----------
		tetrode : int
		clusters : int or list
		ax : matplotlib.Axes
			the axes to plot into. If None a new figure window is created.
		"""
		waves = self.TETRODE[tetrode].waveforms
		clust_idx = self.TETRODE[tetrode].cut == clusters
		clust_waves = waves[clust_idx, :, :]
		gains = self.TETRODE[tetrode].gains
		samps_per_spike = int(self.TETRODE[tetrode].header['samples_per_spike'])
		clust_waves = clust_waves * 1e6 # now in uv
		ADC_scale = int(self.setheader['ADC_fullscale_mv'])
		axes_scales = (ADC_scale / gains.astype(float)) * 1000 # axes limits in uv
		if ~np.any(clust_idx):
			return
		if ax is None:
			fig = plt.figure()
			ax = fig.add_subplot(111)
		else:
			fig = kwargs['figure']
			ax = ax
		ax.axis('off')
		rect = ax.get_position().bounds
		x = np.linspace(0, 0.001, samps_per_spike)
		if 'clustColour' in kwargs.keys():
			clustColour = kwargs['clustColour']
		else:
			if clusters is None:
				clustColour = tcols.colours[0]
			else:
				clustColour = tcols.colours[clusters]
		grid = ImageGrid(fig, rect, nrows_ncols= (1, 4), axes_pad=0.1, add_all=True, share_all=True)
		for i in range(4):
			lc = LineCollection(list(zip(x,y) for y in np.squeeze(clust_waves[:, i, :])))
			lc.set_rasterized(True)
			lc.set_color(clustColour)
			grid[i].add_collection(lc)
			grid[i].plot(x, np.squeeze(np.mean(clust_waves[:, i, :], 0)), 'w-')
			grid[i].set_aspect(2.5e-6*(ADC_scale/1000.))
			grid[i].set_xlim(0, 0.001)
			grid[i].set_ylim(-axes_scales[i], axes_scales[i])
			grid[i].set_rasterized(True)
			grid[i].tick_params(axis='both', which='both', left='off', right='off',
							bottom='off', top='off')
			grid[i].text(0.9,0.95, str(i+1), va='top', ha='right', size='small', color='k', transform=grid[i].transAxes)

		plt.setp([a.get_xticklabels() for a in grid], visible=False)
		plt.setp([a.get_yticklabels() for a in grid], visible=False)


	def plotSAC(self, tetrode, clusters, ax=None, binsize=3, **kwargs):
		"""
		Plots the spatial autocorrelogram of the given tetrode/ cluster

		Parameters
		----------
		tetrode : int
		cluster : int
		ax : matplotlib.pyplot.axis
			plots into this axis
		binsize : int
			size of bins (cms)

		See Also
		--------
			plotFullSAC

		"""
		if ax is None:
			fig = plt.figure()
			ax = fig.add_subplot(111)
		rmap = self._getMap(tetrode=tetrode, cluster=clusters, binsize=binsize, **kwargs)[0]
		nodwell = ~np.isfinite(rmap)
		ret = self.autoCorr2D(rmap, nodwell)
		ax.imshow(ret, interpolation='nearest', origin='lower')
		ax.set_aspect('equal')
		plt.setp(ax.get_xticklabels(), visible=False)
		plt.setp(ax.get_yticklabels(), visible=False)
		return ax

	def plotFullSAC(self, tetrode, clusters, ax=None, binsize=3, limit2mask=False, plot=True, **kwargs):
		"""
		Plots the full SAC ie including the edges and colours the central area in colour
		and the outlying bits (outside the mask area used to calculate gridness) in black and white

		Parameters
		----------
		tetrode : int
		cluster : int
		ax : matplotlib.pyplot.axis
			plots into this axis
		binsize : int
			size of bins (cms)

		"""
		if plot:
			if ax is None:
				fig = plt.figure()
				ax = fig.add_subplot(111)
		if 'step' in kwargs.keys():
			step = kwargs.pop('step')
		else:
			step = 30
		rmap = self._getMap(tetrode=tetrode, cluster=clusters, binsize=binsize, limit2mask=limit2mask, **kwargs)[0]
		nodwell = np.isnan(rmap)
		ret = self.autoCorr2D(rmap, nodwell)
		dct = self.getMeasures(ret, step=step)
		if 'gaussian' in kwargs.keys():
			kwargs.pop('gaussian')
		if plot:
			print('\nGridness: {0}\nOrientation: {1}\nScale: {2}'.format(dct['gridness'], dct['orientation'], dct['scale']))
			self.show(ret, dct, ax=ax, **kwargs)
		return dct

	def getFieldRuns(self, tetrode, cluster, binsize=3):
		'''
		Extracts the runs through a place field of a given cluster on a given
		tetrode and returns the indices of the runs that are kept (defaults to
		at least 5 spikes needing to be fired) and the indices of the spikes in
		the run and the duration of each run

		Parameters
		----------
		tetrode : int
		cluster : int
		binsize : int
			size of bins (cms)

		Returns
		-------
		data : tuple
			The runs retained, the spikes in the run and the run duration
		'''

		# label is a mask of the place field - this could be hijacked to cover the whole track
		label, xe, ye = self._getFieldLims(tetrode, cluster, binsize)
		S = skimage.measure.regionprops(label)
		areas = [s['area'] for s in S]# get the biggest field
		bigFieldIdx = np.argmax(areas)
		bigFieldProps = S[bigFieldIdx]
		binCoords = bigFieldProps.coords
		min_field_edge = np.min(binCoords[:,1])
		max_field_edge = np.max(binCoords[:,1])
		x_coord_field_min = xe[min_field_edge]
		x_coord_field_max = xe[max_field_edge]
		xy = self._getPath()
		x_field_bool = np.logical_and(xy[0] > x_coord_field_min,
								  xy[0] < x_coord_field_max)
		# find the runs with spikes
		run_indices = x_field_bool.nonzero()
		# get a list of runs
		runs = np.array_split(run_indices[0], np.where(np.diff(run_indices[0])>10)[0]+1)
		# there might be short runs through the field so calculate the min distance
		# across the largest part of the field - done so this method can hopefully
		# account for runs in open fields as well as linear tracks
		ppb = int(self.POS.header['pixels_per_metre']) / 100. # pixels per bin
		dist2CrossField = bigFieldProps['major_axis_length'] * ppb
		# get the spike indices into position data
		idx = self.TETRODE[tetrode].getClustIdx(cluster)
		runs_to_keep = []
		spks_in_run = []
		run_duration = []
		for run in runs:
			if np.nansum(np.hypot(np.diff(xy[0,run]),np.diff(xy[1,run]))) > (dist2CrossField / 2): # be conservative and take 1/2 dist
				if np.intersect1d(run, idx).any():
					if len(np.intersect1d(run, idx)) > self._min_spks:# if there are >5 spikes keep run
						runs_to_keep.append(run)
						spks_in_run.append(np.intersect1d(run, idx))
						run_duration.append((run[-1] - run[0])/float(self.POS.pos_sample_rate))
		return runs_to_keep, spks_in_run, run_duration

	def tortuosity(self, xy=None):
		'''
		Parameters
		-----------
		xy - numpy.array
			2xm matrix of xy positions. Default is None so will use this
			instances xy array in POS

		Returns
		--------
		tortuosity : float
			tortuosity index calculated as follows:
			T = sum(path_segment / segment_straight_line) / n_segments
			n_segments is the number of one second segments per trial
		'''

		if xy is None:
			xy = self._getPath()
		T = np.zeros(int(np.shape(xy)[1]/50))
		idx = 0
		for i in xrange(0, xy.shape[1]-50, 50):
			straight_line = np.hypot(xy[0,i] - xy[0,i+50], xy[1,i] - xy[1,i+50])
			path_segment = np.nansum(np.hypot(np.diff(xy[0,i:i+50]),np.diff(xy[1,i:i+50])))
			T[idx] = path_segment / straight_line
			idx += 1
		toobigbool = T > 100
		T = np.delete(T, toobigbool.nonzero())
		zerobool = T==0
		T = np.delete(T, zerobool.nonzero())
		T = np.delete(T, np.isinf(T).nonzero())
		T = np.delete(T, np.isnan(T).nonzero())
		return np.sum(T) / len(T)

	def getThigmotaxisIndex(self):
		'''
		Currently fucked
		Calculates the ratio of time spent in the middle of the environment
		to the amount of time spent in the central part
		'''
		dwellmap = self._getMap(smooth=False)[0] # unsmoothed dwell map
		# simply calculate the sums in the corners and see if this 
		# goes above some threshold
		corner_sz = 3
		tl = dwellmap[0:corner_sz, 0:corner_sz]
		tr = dwellmap[0:corner_sz, -corner_sz:]
		bl = dwellmap[-corner_sz:, 0:corner_sz]
		br = dwellmap[-corner_sz:, -corner_sz:]
		corner_dwell = np.sum([tl, tr, bl, br])

		if corner_dwell > 20:
			shape = 'square'

		else:
			shape = 'circle'

	def getBorderScore(self, tetrode, cluster, debug=False, **kwargs):
		'''
		Calculates the border score in a similar way to how the Moser group did
		but can also deal with circular environments as well as square ones

		Wrapper for fieldcalcs getBorderScore - see there for docs

		Parameters
		----------
		tetrode : int
		cluster : int
		debug : bool

		See Also
		--------
		fieldcalcs.FieldCalcs.getBorderScore
		'''

		A = self._getMap(tetrode, cluster, **kwargs)[0]
		dwellmap = self._getMap(smooth=None)[0]
		# simply calculate the sums in the corners and see if this
		# goes above some threshold
		corner_sz = 3
		tl = dwellmap[0:corner_sz, 0:corner_sz]
		tr = dwellmap[0:corner_sz, -corner_sz:]
		bl = dwellmap[-corner_sz:, 0:corner_sz]
		br = dwellmap[-corner_sz:, -corner_sz:]
		corner_dwell = np.sum([tl, tr, bl, br])

		A_rows, A_cols = np.shape(A)

		if corner_dwell > 20:
			shape = 'square'
		else:
			shape = 'circle'
		return self.fieldcalcs.getBorderScore(A, shape=shape, debug=debug)

	def plotDirFilteredRmaps(self, tetrode, cluster, maptype='rmap', **kwargs):
		'''
		Plots out directionally filtered ratemaps for the tetrode/ cluster

		Parameters
		----------
		tetrode : int
		cluster : int
		maptype : str
			Valid values include 'rmap', 'polar', 'xcorr'
		'''
		inc = 8.0
		step = 360/inc
		dirs_st = np.arange(-step/2, 360-(step/2), step)
		dirs_en = np.arange(step/2, 360, step)
		dirs_st[0] = dirs_en[-1]

		if 'polar' in maptype:
			fig, axes = plt.subplots(nrows=3, ncols=3, subplot_kw={'projection': 'polar'})
		else:
			fig, axes = plt.subplots(nrows=3, ncols=3)
		ax0 = axes[0][0] # top-left
		ax1 = axes[0][1] # top-middle
		ax2 = axes[0][2] # top-right
		ax3 = axes[1][0] # middle-left
		ax4 = axes[1][1] # middle
		ax5 = axes[1][2] # middle-right
		ax6 = axes[2][0] # bottom-left
		ax7 = axes[2][1] # bottom-middle
		ax8 = axes[2][2] # bottom-right

		max_rate = 0
		for d in zip(dirs_st, dirs_en):
			self.posFilter = {'dir': (d[0], d[1])}
			if 'polar' in maptype:
				rmap = self._getMap(tetrode=tetrode, cluster=cluster, var2bin='dir')[0]
			elif 'xcorr' in maptype:
				x1 = self.TETRODE[tetrode].getClustTS(cluster) / (96000/1000)
				rmap = self.spikecalcs.xcorr(x1, x1, Trange=np.array([-500, 500]))
			else:
				rmap = self._getMap(tetrode=tetrode, cluster=cluster)[0]
			if np.nanmax(rmap) > max_rate:
				max_rate = np.nanmax(rmap)

		from collections import OrderedDict
		dir_rates = OrderedDict.fromkeys(dirs_st, None)

		for d in zip(dirs_st, dirs_en, [ax5,ax2,ax1,ax0,ax3,ax6,ax7,ax8]):
			self.posFilter = {'dir': (d[0], d[1])}
			npos = np.count_nonzero(np.ma.compressed(~self.POS.dir.mask))
			print("npos = {}".format(npos))
			nspikes = np.count_nonzero(np.ma.compressed(~self.TETRODE[tetrode].getClustSpks(cluster).mask[:,0,0]))
			print("nspikes = {}".format(nspikes))
			dir_rates[d[0]] = nspikes# / (npos/50.0)
			if 'spikes' in maptype:
				self.plotSpikesOnPath(tetrode, cluster, ax=d[2], markersize=4)
			elif 'rmap' in maptype:
				self._plotMap(tetrode, cluster, ax=d[2], vmax=max_rate)
			elif 'polar' in maptype:
				self._plotMap(tetrode, cluster, var2bin='dir', ax=d[2], vmax=max_rate)
			elif 'xcorr' in maptype:
				self.plotXCorr(tetrode, cluster, ax=d[2])
				x1 = self.TETRODE[tetrode].getClustTS(cluster) / (96000/1000)
				print("x1 len = {}".format(len(x1)))
				dir_rates[d[0]] = self.spikecalcs.thetaBandMaxFreq(x1)
				d[2].set_xlabel('')
				d[2].set_title('')
				d[2].set_xticklabels('')
			d[2].set_title("nspikes = {}".format(nspikes))
		self.posFilter = None
		if 'spikes' in maptype:
			self.plotSpikesOnPath(tetrode, cluster, ax=ax4)
		elif 'rmap' in maptype:
			self._plotMap(tetrode, cluster, ax=ax4)
		elif 'polar' in maptype:
			self._plotMap(tetrode, cluster, var2bin='dir', ax=ax4)
		elif 'xcorr' in maptype:
			self.plotXCorr(tetrode, cluster, ax=ax4)
			ax4.set_xlabel('')
			ax4.set_title('')
			ax4.set_xticklabels('')
		return dir_rates