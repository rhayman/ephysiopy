# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 15:01:45 2014

@author: Robin
"""
import numpy as np
from scipy import interpolate, signal, spatial, misc, stats, ndimage, interpolate, optimize
import matplotlib
import os
import re
from glob import glob
import subprocess
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
from matplotlib.transforms import offset_copy
from matplotlib.widgets import Button, RadioButtons
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
from ephysiopy import dacq2py
from ephysiopy.dacq2py import tintcolours as tcols
from astropy.convolution import convolve
from sklearn.decomposition import PCA
from scikits import bootstrap
from ephysiopy.dacq2py import gridcell
from itertools import combinations
from mpl_toolkits.axes_grid1 import ImageGrid
import skimage, skimage.morphology, skimage.feature
from ephysiopy.dacq2py.dacq2py_util import Trial
from ephysiopy.dacq2py import spikecalcs

class LTrack(Trial):
	'''
	Class definition
	'''
	def __init__(self, fname_root):
		self.fname_root = fname_root
		self._figNum = None
		self._min_spks = 1
		self._getAvailableFiles()
		for fs in self._available_files:
			if 'log' in fs:
				self._logData = self._getLogData()
		self.POS = dacq2py.Pos(fname_root) # load pos by default
		self.POS.postprocesspos() # and postprocess
		self._xlims = (int(self.POS.header['window_min_x']), int(self.POS.header['window_max_x']))
		self._ylims = (int(self.POS.header['window_min_y']), int(self.POS.header['window_max_y']))
		try:
			self.EEG = dacq2py.EEG(fname_root)
#            self.EEG.thetaAmpPhase()
		except IOError:
			self.EEG = None
		self.setheader = self.POS.getHeader(self.fname_root + '.set')
		if 'minX' in self.setheader['comments']:
			minmaxRE = re.compile('minX\s+=\s+(\d+)\D\s+maxX\s+=\s+(\d+)')
			m = minmaxRE.findall(self.setheader['comments'])
			self._laser_edges = (int(m[0][0]), int(m[0][1]))
		else:
			self._laser_edges = None
		self.TETRODE = {} # a dict to hold all tetrodes - only loaded as and when needed
		self.LOG = {}
		self.spikecalcs = spikecalcs()
		self.posFilter = None # can be a dict used to filter pos e.g. {'dir': (225, 315)}
		
		
		self._ptSize = 40
		self._lines = None
		self._path = None
		self._pathSpks = None
		self._eeglines = None
		self._eeglines2 = None
		self._eegspks = None
		self._eegfirst = None
		self._ppText = None
		self._posText = None
		self._eegText = None
		self._pts = None
		self._firstSpks = None
		self._regressRunLine = None
		self._pathFirstSpks = None
		self.pp_patch = None
		self.pos_patch = None
		self.eeg_patch = None
		self._xticks = None
		self._runIdx = 0
		

	def _getClusterXPos(self, tetrode, cluster):
		'''
		Returns the x pos of a given cluster taking account of any masking
		'''
		idx = self.TETRODE[tetrode].getClustIdx(cluster)
		x = self.POS.xy[0, idx]
		return x

	def plotPhaseVsX(self, tetrode, cluster, ax=None):
		'''
		Plots phase of theta cycle at which a given cluster on a given tetrode
		fired against x position (used for linear track analysis).
		Pos filters are applied if present
		'''
		if not self.EEG:
			self.EEG = dacq2py.EEG(self.filename_root)
		x_pos = self._getClusterXPos(tetrode, cluster)
		x_pos = np.ma.hstack((x_pos, x_pos))  # duplicate for 720 degree plot
		x_phase = self._getClusterPhaseVals(tetrode, cluster)
		x_phase = np.ma.hstack((x_phase, x_phase + (2*np.pi)))
		if ax is None:
			ax = plt.gca()
		else:
			ax = ax
		ax.plot(x_pos, x_phase, '.')
		ax.set_xlim(self._xlims)
			
	def plotPPperRun(self, tetrode, cluster):
		self.__checkTetrodeLoaded__(tetrode)
		if not self.EEG:
			self.EEG = dacq2py.EEG(self.fname_root)
		self.EEG.sm_eeg = self.EEG.eegfilter()
		idx = self.TETRODE[tetrode].getClustIdx(cluster)
		
		# remove any pos filtering to get peaks and troughs
		old_filter = self.posFilter
#        old_x_mask = 
		self.posFilter = None
		peaks, troughs = self._getThetaCycles()
		clust_ts = self.TETRODE[tetrode].getClustTS(cluster)
		firstSpk = self._getSpikeInCycle(peaks, clust_ts)
		self.firstSpkEEGIdx = firstSpk
		self.firstSpkPosIdx = firstSpk / 5
		self.posFilter = old_filter
		
		t = self._getClusterPhaseVals(tetrode, cluster)
		x = self._getClusterXPos(tetrode, cluster)
		label, xe, _ = self._getFieldLims(tetrode, cluster)
		xInField = xe[label.nonzero()[1]]
		mask = np.logical_and(x > np.min(xInField), x < np.max(xInField)) # same length as idx (nSpikes in cluster)
		# combine this mask with the possibly pre-existing one for the x and phase data if pos filtering has been applied
		x = np.ma.masked_where(~mask, x)
		t = np.ma.masked_where(~mask, t)
		
		# keep x between -1 and +1
		self.mnx = np.ma.mean(x)
		xn = x - self.mnx
		self.mxx = np.ma.max(np.ma.abs(xn))
		x = xn / self.mxx
		self.x = x
		self.t = t
		self.idx = idx
		
		slope, intercept = self.circRegress(x, t)
		rho, p, rho_boot, p_shuff, ci = self.circCircCorrTLinear(x, t)
		self.fig, self.ax = plt.subplots(num=self._figNum, figsize=(8.27, 11.69))
		
		plt.subplots_adjust(bottom=0.3, top=0.85)
		self.ptcolor = tcols.colours[0]
		self.pts = self.ax.scatter(x, t, color = self.ptcolor, s=self._ptSize)
		self.ax.plot((-1, 1), (-slope + intercept, slope + intercept), 'r', lw=3)
		
		plt.axis([-1,1,-np.pi,np.pi])
		
		runs, spks, durations = self.getFieldRuns(tetrode, cluster)
		self.runs = runs
		self.spks = spks
		# path axis
		self.posAx = self.fig.add_axes([0.125, 0.02, 0.775, 0.1])
		self.posAx.plot(self.POS.xy[0], self.POS.xy[1], color=self.ptcolor)
		self.posAx.set_xlim((np.min(xInField),np.max(xInField)))
		self.posAx.set_ylim(self._ylims)
#        self.posAx.set_aspect('equal')
		plt.setp(self.posAx.get_xticklabels(), visible=False)
		plt.setp(self.posAx.get_yticklabels(), visible=False)
		# eeg axis
		self.eegaxis = self.fig.add_axes([0.125,0.15,0.775,0.1])
		self.eegaxis.set_xlim((np.min(xInField),np.max(xInField)))
		plt.setp(self.eegaxis.get_xticklabels(), visible=False)
		plt.setp(self.eegaxis.get_yticklabels(), visible=False)

		# next and previous buttons 
		self.nextax = plt.axes([0.5,0.9,0.1,0.05])
		self.nextbutton = Button(self.nextax, 'Next', color='r')
		self.nextbutton.on_clicked(self.nextRun)
		
		self.prevax = plt.axes([0.3,0.9,0.1,0.05])
		self.prevbutton = Button(self.prevax, 'Prev', color='r')
		self.prevbutton.on_clicked(self.prevRun)
		plt.show()
		
		self.update()
		
	def clear(self, ax, lines):
		if ax is not None:
			for line in lines:
				line.remove()
			ax.figure.canvas.draw_idle()
		
	def update(self):
		pos_v = np.intersect1d(self.spks[self._runIdx],self.idx)
		this_idx = np.searchsorted(self.idx, pos_v)
		v = np.intersect1d(self.firstSpkPosIdx, pos_v)
		firstSpkIdx = np.searchsorted(self.idx, v)
#        self.clear(self._path, self._lines)
		if self._lines is not None:
			for line in self._lines:
				line.remove()
		if self._pts is not None:
			self._pts.remove()
		if self._firstSpks is not None:
			self._firstSpks.remove()
		if self._path is not None:
			for line in self._path:
				line.remove()
		if self._pathSpks is not None:
			for line in self._pathSpks:
				line.remove()
		if self._eeglines is not None:
			for line in self._eeglines:
				line.remove()
		if self._eeglines2 is not None:
			for line in self._eeglines2:
				line.remove()
		if self._eegspks is not None:
			for line in self._eegspks:
				line.remove()
		if self._eegfirst is not None:
			for line in self._eegfirst:
				line.remove()
		if self._ppText is not None:
			for txt in self._ppText:
				txt.remove()
		if self._posText is not None:
			for txt in self._posText:
				txt.remove()
		if self._eegText is not None:
			for txt in self._eegText:
				txt.remove()
		if self._regressRunLine is not None:
			for line in self._regressRunLine:
				line.remove()
		if self._pathFirstSpks is not None:
			for line in self._pathFirstSpks:
				line.remove()
		if self._xticks is not None:
			for line in self._xticks:
				line.remove()
		if self.pp_patch is not None:
			self.pp_patch.remove()
		if self.pos_patch is not None:
			self.pos_patch.remove()
		if self.eeg_patch is not None:
			self.eeg_patch.remove()
		self._lines = self.ax.plot(self.x[this_idx], self.t[this_idx],'k',lw=1)
		self._pts = self.ax.scatter(self.x[this_idx], self.t[this_idx], c='b',s=self._ptSize)
		self._firstSpks = self.ax.scatter(self.x[firstSpkIdx], self.t[firstSpkIdx], c='r',s=self._ptSize)
		# plot the regeression line through the x-phase values in this run
		x = self.x[this_idx]
		t = self.t[this_idx]
#        mnx = np.ma.mean(x)
#        xn = x - mnx
#        mxx = np.ma.max(np.ma.abs(xn))
#        x = xn / mxx
		slope, intercept = self.circRegress(x, t)
		rho, p, rho_boot, p_shuff, ci = self.circCircCorrTLinear(x, t)
		self._regressRunLine = self.ax.plot((-1, 1), (-slope + intercept, slope + intercept), 'k--', lw=2)
		
		transOffset = offset_copy(self.ax.transData, fig=self.fig, x=5, y=5, units='dots')
		self._ppText = []
		for s,x,y in zip(range(len(this_idx)), self.x[this_idx], self.t[this_idx]):
			self._ppText.append(self.ax.text(x, y, s='%d' % (int(s)), fontsize=15, transform=transOffset))
		
		this_run = self.runs[self._runIdx]
		these_spks = self.spks[self._runIdx]
		self._path = self.posAx.plot(self.POS.xy[0,this_run],self.POS.xy[1,this_run],lw=3,c='k')
		self._pathSpks = self.posAx.plot(self.POS.xy[0,these_spks],self.POS.xy[1,these_spks],'s', c='b', mec='b')
		self._pathFirstSpks = self.posAx.plot(self.POS.xy[0,v],self.POS.xy[1,v],'s', c='r', mec='r')
		
		transOffset = offset_copy(self.posAx.transData, fig=self.fig, x=0, y=15, units='dots')
		
		self._posText = []
		for s,x,y in zip(range(len(these_spks)), self.POS.xy[0,these_spks], self.POS.xy[1,these_spks]):
			self._posText.append(self.posAx.text(x, y, s='%d' % (int(s)), fontsize=11, transform=transOffset))

		m = interpolate.interp1d(np.linspace(0,len(this_run), len(this_run)), self.POS.xy[0,this_run])
		x_new = m(np.linspace(0, len(this_run), len(self.EEG.sm_eeg[this_run[0] * self.pos2eegScale:this_run[-1] * self.pos2eegScale])))
		self._eeglines = self.eegaxis.plot(x_new, self.EEG.eeg[this_run[0]*5:this_run[-1]*5], c=[0.8627, 0.8627, 0.8627], lw=5)
		self._eeglines2 = self.eegaxis.plot(x_new, self.EEG.sm_eeg[this_run[0] * self.pos2eegScale:this_run[-1] * self.pos2eegScale], c='k')
		self._eegspks = self.eegaxis.plot(self.POS.xy[0,these_spks], self.EEG.sm_eeg[these_spks * self.pos2eegScale], '|', ms=9, mew=3, c='b')
		self._eegfirst = self.eegaxis.plot(self.POS.xy[0,v], self.EEG.sm_eeg[v * self.pos2eegScale], '|', ms=9, mew=3, c='r')
		
		transOffset = offset_copy(self.eegaxis.transData, fig=self.fig, x=0, y=13, units='dots')
		
		self._eegText = []
		for s,x,y in zip(range(len(these_spks)), self.POS.xy[0,these_spks], self.EEG.sm_eeg[these_spks * self.pos2eegScale]):
			self._eegText.append(self.eegaxis.text(x, y, s='%d' % (int(s)), fontsize=15, transform=transOffset))
		
		# add xticks/ lines at intervals of 1/10 seconds to the eeg plot
		self._xticks = []
		axtrans = transforms.blended_transform_factory(self.eegaxis.transData, self.eegaxis.transAxes)
		for xx in x_new[0:-1:int(self.EEG.sample_rate/10)]:
			self._xticks.append(self.eegaxis.add_line(Line2D((xx,xx),(-1,1),c='b', ls='--',transform=axtrans)))


		# add the actual laser on/off position to the plots  as some kind of box        
		if self._laser_edges is not None:
			log_data = self._getLogData()
			log_ts = log_data['ts']  / 20
			log_state = log_data['state']
			laser_on_pos_idx = log_ts[log_state=='on']
			this_run_on = np.intersect1d(laser_on_pos_idx, this_run)
			start_x = np.min(self.POS.xy[0,this_run_on])
			end_x = np.max(self.POS.xy[0,this_run_on])
			left_edge = (start_x - self.mnx) / self.mxx
			right_edge = (end_x - self.mnx) / self.mxx
			laserAxTrans = transforms.blended_transform_factory(self.ax.transData, self.ax.transAxes)
			self.pp_patch = self.ax.add_patch(Rectangle((left_edge,0), width=np.diff((left_edge,right_edge)), 
								   height=1, transform = laserAxTrans, color=[1,1,0], alpha=0.5))
			laserAxTrans = transforms.blended_transform_factory(self.posAx.transData, self.posAx.transAxes)
			self.pos_patch = self.posAx.add_patch(Rectangle((start_x,0), width=np.ptp((start_x, end_x)), 
								   height=1, transform = laserAxTrans, color=[1,1,0], alpha=0.5))
			laserAxTrans = transforms.blended_transform_factory(self.eegaxis.transData, self.eegaxis.transAxes)
			self.eeg_patch = self.eegaxis.add_patch(Rectangle((start_x,0), width=np.ptp((start_x, end_x)), 
								   height=1, transform = laserAxTrans, color=[1,1,0], alpha=0.5))
		
		
		self.eegaxis.figure.canvas.draw_idle()
		self.posAx.figure.canvas.draw_idle()
		self.ax.figure.canvas.draw_idle()
		
	def prevRun(self, event):
		if self._runIdx <= 0:
			self._runIdx = 0
		else:
			self._runIdx -= 1
		self.update()
	def nextRun(self, event):
		if self._runIdx == len(self.runs):
			self._runIdx = len(self.runs)
		else:
			self._runIdx += 1
		self.update()


class LinearTrackTrial(Trial):
	def plotLTSpikesRateMap(self, tetrode, cluster, clamp=True, ax=None):
		'''
		====
		plotLTSpikesRateMap
		====

		Definition: plotSpikesOnPath(tetrode, clusters, clamp=True)

		----

		Plots the spikes on the path during a trial for a particular tetrode/
		cluster(s) with the ratemap below and a histogram of the spikes vs x
		above the spike / pos plot + a kernel smoothed density estimate of the
		smoothed spike/ pos data. Only does this for a single cluster (not multiple)

		Parameters
		----------
		tetrode: integer
				 the tetrode you want to look at
		cluster: integer, 1xn array/ list
				 a single number or list (or 1xn array) of the clusters to plot
		clamp:   bool, optional
				 whether to restrict the plot to the self._xlims and self_ylims
				 property

		'''
		xy = self._getPath()
		fig = plt.figure(self._figNum)

		if ax is None:
			ax = fig.add_subplot(1, 1, 1)

		idx = self.TETRODE[tetrode].getClustIdx(cluster)
		ax.plot(xy[0], xy[1], c=tcols.colours[0], zorder=1)
		ax.plot(xy[0, idx], xy[1, idx], 's', c=tcols.colours[cluster],
				mec=tcols.colours[cluster])
		ax.set_xlim(self._xlims)
		ax.set_ylim(self._ylims)
		ax.set_aspect('equal')

		divider = make_axes_locatable(ax)
		axHistx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)

		binsize = self.__getBinUnits__()
		if np.ma.is_masked(self.POS.xy[0]):
			mask = ~np.ma.getmask(self.POS.xy[0])
			mask = mask.nonzero()[0]
			idx = idx[np.in1d(idx, mask)]

		axHistx.hist(self.POS.xy[0, idx], bins=self.binsize[1],
					 range=self._xlims)[0]
		# calculate and plot the kernel smoothed density estimate
		kde = stats.gaussian_kde(self.POS.xy[0, idx])
		# calculate the amount to normalise kde by
		norm = len(self.POS.xy[0, idx]) * (np.float(np.ptp(self._xlims)) / binsize[1])
		grid = np.linspace(self._xlims[0], self._xlims[1], binsize[1])
		z = kde.evaluate(grid)
		# normalise kde by the number of data points and the bin width
		z = z * norm
		axHistx.plot(grid, z, 'r')

		axRmap = divider.append_axes("bottom", 1.2, pad=0.1, sharex=ax, sharey=ax)
		rmap, ye, xe = self._getMap(tetrode, cluster)
		axRmap.imshow(rmap, extent=(xe[0], xe[-1], ye[0], ye[-1]),
					  interpolation='nearest', origin='lower')

		plt.setp(axHistx.get_xticklabels() + axHistx.get_yticklabels(),
				 visible=False)
		plt.setp(ax.get_xticklabels() + ax.get_yticklabels(), visible=False)
		plt.setp(axRmap.get_xticklabels() + axRmap.get_yticklabels(),
				 visible=False)
		plt.show()

	def plotLTSpikesOnPath(self, tetrode, cluster, figNum=1):
		'''
		plotts a n row by 3 column plot:
		1st column is both directions of running
		2nd column is leftward runs
		3rd column is rightward runs
		'''
		self._figNum = figNum
		old_filter = self.posFilter
		self.posFilter = None
		self.plotSpikesOnPath(tetrode, cluster, colNum=1)
		self.posFilter = {'dir': (315, 45)}
		self.plotSpikesOnPath(tetrode, cluster, colNum=2)
		self.posFilter = {'dir': (135, 225)}
		self.plotSpikesOnPath(tetrode, cluster, colNum=3)
		self.posFilter = old_filter
