# -*- coding: utf-8 -*-
"""
Created on Mon May 23 16:27:31 2016

@author: robin
"""
import numpy as np
from ephysiopy.dacq2py.dacq2py_util import Trial
from ephysiopy.dacq2py.eegcalcs import EEGCalcs
import matplotlib.pyplot as plt
import yaml
import os

class OptoClusterSummary(Trial):
	"""
	docs
	"""
	def __init__(self, filename_root, *args, **kwargs):
		super(OptoClusterSummary, self).__init__(filename_root, *args, **kwargs)

	def __repr__(self):
	   return "{0}.{1}(filename_root=r\'{2}\')".format(self.__class__.__module__, self.__class__.__name__, self.filename_root)

	def plotLaserFilteredSpectrogram(self, eeg_file='eeg', ymax=50, ax=None, secsPerBin=2, **kwargs):
		pass

	def plotFiringRateChange(self, tet, clust, n=50):
		'''
		Produces a bar graph of the firing rate in the n ms before laser onset
		and 50 ms after laser offset
		TODO: THIS IS BROKEN
		'''
		spkTimes_ms = self.TETRODE[tet].getClustTS(clust) / (self.TETRODE[tet].timebase / 1000)
		spkTimes_ms.sort()
		stm_timebase = int(self.STM['timebase'].split()[0])
		stm_on = self.STM['on'] / float(stm_timebase) * 1000
		stm_on.sort()
		stm_off = self.STM['off'] / float(stm_timebase) * 1000
#        stm_off.sort()
		Trange_to_onset = np.array([-n, 0])
		Trange_to_offset = np.array([0, n])
		irange_to_onset = spkTimes_ms[:, np.newaxis] + Trange_to_onset[np.newaxis, :]
		irange_to_offset = spkTimes_ms[:, np.newaxis] + Trange_to_offset[np.newaxis, :]
		dts_to_onset = np.searchsorted(stm_on, irange_to_onset)
		dts_to_offset = np.searchsorted(stm_off, irange_to_offset)
		counts_to_onset = []
		counts_to_offset = []
		for i, t in enumerate(dts_to_onset):
			counts_to_onset.extend(stm_on[t[0]:t[1]] - spkTimes_ms[i])
		for i, t in enumerate(dts_to_offset):
			counts_to_offset.extend(stm_off[t[0]:t[1]] - spkTimes_ms[i])

		print('\nMean firing rate {0}ms before laser = {1:.2f} Hz'.format(n, np.mean(np.abs(counts_to_offset))))
		print('Mean firing rate {0}ms after laser  = {1:.2f} Hz'.format(n, np.mean(np.abs(counts_to_onset))))

		return counts_to_onset, counts_to_offset

	def getFiringRateDuringLaser(self, tet, clust, n=50):
		'''
		Returns a number which is the difference divided by the sum of the mean firing rate
		in the 50ms (or n ms) prior to laser onset and the firing rate during the stim period
		'''
		spkTimes_ms = self.TETRODE[tet].getClustTS(clust) / (self.TETRODE[tet].timebase / 1000)
		spkTimes_ms.sort()
		stm_timebase = int(self.STM['timebase'].split()[0])
		stm_on = self.STM['on'] / float(stm_timebase) * 1000
		stm_on.sort()
		stm_off = self.STM['off'] / float(stm_timebase) * 1000
#        stm_off.sort()
		Trange_to_onset = np.array([-n, 0])
		Trange_to_offset = np.array([0, 10])
		irange_to_onset = spkTimes_ms[:, np.newaxis] + Trange_to_onset[np.newaxis, :]
		irange_to_offset = spkTimes_ms[:, np.newaxis] + Trange_to_offset[np.newaxis, :]
		dts_to_onset = np.searchsorted(stm_on, irange_to_onset)
		dts_to_offset = np.searchsorted(stm_on, irange_to_offset)
		counts_to_onset = []
		counts_to_offset = []
		for i, t in enumerate(dts_to_onset):
			counts_to_onset.extend(stm_on[t[0]:t[1]] - spkTimes_ms[i])
		for i, t in enumerate(dts_to_offset):
			counts_to_offset.extend(stm_on[t[0]:t[1]] - spkTimes_ms[i])
		before_rate = np.mean(np.abs(counts_to_offset))
		after_rate = np.mean(np.abs(counts_to_onset))
		change = float(before_rate - after_rate) / float(before_rate + after_rate)
		return before_rate, after_rate,change

	def plotAllPhasesXCorr(self, tet, clust, savename=None):
		fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(5.845, 4.135))
		ax0 = axes[0]
		ax1 = axes[1]
		ax2 = axes[2]
		self._filterForStm(laser=0)
		# Split the posFilter into A and A'
		phaseA = self.posFilter['time'][0,:] / 50
		phaseAprime = self.posFilter['time'][1,:] / 50

		self.posFilter = {'time': phaseA}
		ax0, xcorr0 = super(OptoClusterSummary, self).plotXCorr(tet, clust, ax=ax0, annotate=False)
		self.posFilter = {'time': phaseAprime}
		ax2, xcorr2 = super(OptoClusterSummary, self).plotXCorr(tet, clust, ax=ax2, annotate=False)
		self._filterForStm(laser=1)
		ax1, xcorr1 = super(OptoClusterSummary, self).plotXCorr(tet, clust, ax=ax1, annotate=False)
		A_vs_Aprime = self.fieldcalcs.corr_maps(xcorr0[0], xcorr2[0])
		A_vs_B = self.fieldcalcs.corr_maps(xcorr0[0], xcorr1[0])
		Aprime_vs_B = self.fieldcalcs.corr_maps(xcorr2[0], xcorr1[0])
		arrow_args = dict(arrowstyle='|-|', connectionstyle="Bar,armA=2,armB=2,fraction=.2")
		ax1.annotate('', xy=(0.5, 1.05), xycoords=ax0.transAxes,
				  xytext=(0.5,1.05), textcoords=ax1.transAxes, arrowprops=arrow_args)
		ax2.annotate('', xy=(0.5, 1.05), xycoords=ax1.transAxes,
				  xytext=(0.5,1.05), textcoords=ax2.transAxes, arrowprops=arrow_args)
		arrow_args = dict(arrowstyle='|-|', connectionstyle="Bar,armA=2,armB=2,fraction=-.1")
		ax2.annotate('', xy=(0.5, -0.025), xycoords=ax0.transAxes,
				  xytext=(0.5,-0.025), textcoords=ax2.transAxes, arrowprops=arrow_args)
		ax0.text(1.0,1.4,'{:.2f}'.format(A_vs_B),transform=ax0.transAxes)
		ax1.text(1.0,1.4,'{:.2f}'.format(Aprime_vs_B),transform=ax1.transAxes)
		ax1.text(0.5,-.4,'{:.2f}'.format(A_vs_Aprime),transform=ax1.transAxes,ha='center')
		if savename is not None:
			fig.savefig(savename)
		return fig

	def plotAllPhases(self, tet, clusts, savename=None, var2bin='pos',vmax='auto'):
		'''
		Plots all phases i.e. A, B A'
		'''
		if 'dir' in var2bin:
			fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(5.845, 4.135),subplot_kw={"projection":"polar"})
		else:
			fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(5.845, 4.135))
		ax0 = axes[0]
		ax1 = axes[1]
		ax2 = axes[2]
		self._filterForStm(laser=0)
		# Split the posFilter into A and A'
		phaseA = self.posFilter['time'][0,:] / 50
		phaseAprime = self.posFilter['time'][1,:] / 50

		# normalizing the ratemaps has to be done in a two-pass operation
		# first-pass - get ratemap
		self.posFilter = {'time': phaseA}
		rmap0 = self._getMap(tet, clusts)[0]
		self.posFilter = {'time': phaseAprime}
		rmap2 = self._getMap(tet, clusts)[0]
		self._filterForStm(laser=1)
		rmap1 = self._getMap(tet, clusts)[0]
		# get the max value in all ratemaps to scale the others
		if 'auto' in vmax:
			vmax = np.nanmax([rmap0.ravel(), rmap1.ravel(), rmap2.ravel()])
			vmax1 = vmax
			vmax2 = vmax
		else:
			vmax = np.nanmax(rmap0.ravel())
			vmax1 = np.nanmax(rmap1.ravel())
			vmax2 = np.nanmax(rmap2.ravel())
#		vmax=10.0
		print('vmax = {}'.format(vmax))
		self.posFilter = {'time': phaseA}
		ax0, rmap0 = super(OptoClusterSummary, self)._plotMap(tet, clusts, ax=ax0, var2bin=var2bin, binsize=3, smooth_sz=5, smooth=True,vmax=vmax,add_mrv=True)
		self.posFilter = {'time': phaseAprime}
		ax2, rmap2 = super(OptoClusterSummary, self)._plotMap(tet, clusts, ax=ax2, var2bin=var2bin, binsize=3, smooth_sz=5, smooth=True,vmax=vmax1,add_mrv=True)
		self._filterForStm(laser=1)
		ax1, rmap1 = super(OptoClusterSummary, self)._plotMap(tet, clusts, ax=ax1, var2bin=var2bin, binsize=3, smooth_sz=5, smooth=True,vmax=vmax2,add_mrv=True)

		# annotate the figure with arrows and correlation values
		# get the correlation values
		A_vs_Aprime = self.fieldcalcs.corr_maps(rmap0, rmap2)
		A_vs_B = self.fieldcalcs.corr_maps(rmap0, rmap1)
		Aprime_vs_B = self.fieldcalcs.corr_maps(rmap2, rmap1)
		arrow_args = dict(arrowstyle='|-|', connectionstyle="Bar,armA=2,armB=2,fraction=.2")
		ax1.annotate('', xy=(0.5, 1.05), xycoords=ax0.transAxes,
				  xytext=(0.5,1.05), textcoords=ax1.transAxes, arrowprops=arrow_args)
		ax2.annotate('', xy=(0.5, 1.05), xycoords=ax1.transAxes,
				  xytext=(0.5,1.05), textcoords=ax2.transAxes, arrowprops=arrow_args)
		arrow_args = dict(arrowstyle='|-|', connectionstyle="Bar,armA=2,armB=2,fraction=-.1")
		ax2.annotate('', xy=(0.5, -0.025), xycoords=ax0.transAxes,
				  xytext=(0.5,-0.025), textcoords=ax2.transAxes, arrowprops=arrow_args)
		ax0.text(1.0,1.4,'{:.2f}'.format(A_vs_B),transform=ax0.transAxes)
		ax1.text(1.0,1.4,'{:.2f}'.format(Aprime_vs_B),transform=ax1.transAxes)
		ax1.text(0.5,-.4,'{:.2f}'.format(A_vs_Aprime),transform=ax1.transAxes,ha='center')
		if savename is not None:
			fig.savefig(savename)
		return fig

	def plotLaserPhasesRatemap(self, tet, clusts, var2bin='pos', binsize=3,
				smooth_sz=5, smooth=True, laser=None, **kwargs):
		fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(5.845, 4.135))
		self._filterForStm(laser=0)
		ax1, rmap1 = super(OptoClusterSummary, self)._plotMap(tet, clusts, ax=ax1, var2bin='pos', binsize=3, smooth_sz=5, smooth=True)
		ax1.set_xlabel('Laser off')
		self._filterForStm(laser=1)
		ax2, rmap2 = super(OptoClusterSummary, self)._plotMap(tet, clusts, ax=ax2, var2bin='pos', binsize=3, smooth_sz=5, smooth=True)
		ax2.set_xlabel('Laser on')

	def plotLaserPhasesSpikesOnPath(self, tet, clusts):
		fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(5.845, 4.135))
		self._filterForStm(laser=0)
		ax1 = super(OptoClusterSummary, self).plotSpikesOnPath(tet, clusts, ax=ax1)
		ax1.set_xlabel('Laser off')
		self._filterForStm(laser=1)
		ax2 = super(OptoClusterSummary, self).plotSpikesOnPath(tet, clusts, ax=ax2)
		ax2.set_xlabel('Laser on')

	def plotLaserPhasesSpeedVsRate(self, tet, clusts):
		fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(5.845, 4.535))
		self._filterForStm(laser=0)
		super(OptoClusterSummary, self).plotRateVSpeed(tet, clusts, plot=True, ax=ax1)
		ax1.set_xlabel('Laser off')
		self._filterForStm(laser=1)
		super(OptoClusterSummary, self).plotRateVSpeed(tet, clusts, plot=True, ax=ax2)
		ax2.set_xlabel('Laser on')
		ax2.set_ylabel('')

	def plotAllSpeedVsRate(self, tet, clust):
		'''
		plots the speed vs rate graphs for all three sections of a stim
		trial on the same line graph
		'''
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.135, 5.845))
		self._filterForStm(laser=0)
		# Split the posFilter into A and A'
		phaseA = self.posFilter['time'][0,:] / 50
		phaseAprime = self.posFilter['time'][1,:] / 50
		self.posFilter = {'time': phaseA}
		res, spd_binsA, mn_rateA = self.plotRateVSpeed(tet,clust,plot=False, getData=True)
#		ax.plot(spd_binsA, mn_rateA, 'k')
		self.posFilter = {'time': phaseAprime}
		res, spd_binsAp, mn_rateAp = self.plotRateVSpeed(tet,clust,plot=False, getData=True)
#		ax.plot(spd_binsAp, mn_rateAp, 'k--')
		self._filterForStm(laser=1)
		res, spd_binsB, mn_rateB = self.plotRateVSpeed(tet,clust,plot=False, getData=True)
		line0, = ax.plot(spd_binsA, mn_rateA, 'k')
		line1, = ax.plot(spd_binsAp, mn_rateAp, 'k--')
		line2, = ax.plot(spd_binsB, mn_rateB, 'b--')

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
		ax.legend((line0,line2,line1),['Laser off (1)','Laser on','Laser off (2)'],loc=2)
		return fig

	def plotLaserPhasesSAC(self, tetrode, clusters, ax=None, binsize=3, **kwargs):
		fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(5.845, 4.135))
		self._filterForStm(laser=0)
		ax1 = super(OptoClusterSummary, self).plotSAC(tetrode, clusters, ax=ax1, binsize=3, **kwargs)
		ax1.set_xlabel('Laser off')
		self._filterForStm(laser=1)
		ax2 = super(OptoClusterSummary, self).plotSAC(tetrode, clusters, ax=ax2, binsize=3, **kwargs)
		ax2.set_xlabel('Laser on')

	def plotLaserPhasesFreqVSpeed(self, minSp=5, maxSp=50, spStep=5, ax=None, laserFilter=None, **kwargs):
		'''
		Plots both phases for an opto trial ie laser on *and* off on same axes
		Scales axes so y is same on both
		Note that kwargs can contain values that affect the filtering of the EEG
		such as 'width', 'dip' and 'stimFreq' which will change the coefficients of
		the Kaisser filter
		'''
		fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(4.135, 5.845))
		self._filterForStm(laser=0)
		ax1, intercept, slope = super(OptoClusterSummary, self).plotFreqVSpeed(minSp=minSp, maxSp=maxSp, spStep=spStep, ax=ax1, laserFilter=None, **kwargs)
		ax1.set_title('Laser off\nIntercept: {0:.3f}    Slope: {1:.5f}'.format(intercept, slope))
		self._filterForStm(laser=1)
		sf = self.getStimFreq()
		ax2, intercept, slope = super(OptoClusterSummary, self).plotFreqVSpeed(minSp=minSp, maxSp=maxSp, spStep=spStep, ax=ax2, laserFilter=True, stimFreq=sf, **kwargs)
		ax2.set_title('Laser on\nIntercept: {0:.3f}    Slope: {1:.5f}'.format(intercept, slope))
		ax1ylims = ax1.get_ylim()
		ax2ylims = ax2.get_ylim()
		ylims = (np.min((ax1ylims, ax2ylims)), np.max((ax1ylims, ax2ylims)))
		ax1.set_ylim(ylims)
		ax2.set_ylim(ylims)
		ax1.set_xlabel('')
		ax1.tick_params(axis='x',which='both', bottom='off', labelbottom='off')

	def plotFreqVSpeed(self, minSp=5, maxSp=50, spStep=5, ax=None, laser=None, **kwargs):
		'''
		Overrides the same method in the Trial class by filtering for
		presence/ absence of laser stimulation and modifying the title
		of the resulting plot to say whether laser was on /off / 
		not present (nothing added to title)
		'''
		self._filterForStm(laser)
		ax, intercept, slope = super(OptoClusterSummary, self).plotFreqVSpeed(minSp=minSp, maxSp=maxSp, spStep=spStep, ax=ax, **kwargs)
		if laser == 1:
			ax.set_title('Laser on\nIntercept: {0:.3f}    Slope: {1:.5f}'.format(intercept, slope))
		elif laser == 0:
			ax.set_title('Laser off\nIntercept: {0:.3f}    Slope: {1:.5f}'.format(intercept, slope))
		else:
			ax.set_title('Intercept: {0:.3f}    Slope: {1:.5f}'.format(intercept, slope))
		return ax

	def plotLaserPhasesEEGPower(self, width=0.125, dip=15.0, laserFilter=True):
		'''
		Overrides the same method in the Trial class by filtering for
		presence/ absence of laser stimulation and modifying the title
		of the resulting plot to say whether laser was on /off / 
		not present (nothing added to title)
		'''
		EE = EEGCalcs(self.filename_root,thetaRange=[6,12])
		self._filterForStm(laser=0)
		if np.ma.is_masked(self.EEG.eeg):
			eeg = np.ma.compressed(self.EEG.eeg)
		else:
			eeg = self.EEG.eeg
#        if laserFilter:
#            sf = self.getStimFreq()
#            fx = EE.filterForLaser(E=eeg, width=width, dip=dip, stimFreq=sf)
#        else:
#            fx = eeg
		fx = eeg
		fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(4.135, 5.845))
		ax1 = super(OptoClusterSummary, self).plotEEGPower(E=fx, sample_rate=250.0, freqBand=(6,12), ax=ax1)
		ax1.set_title("Laser off\nwidth = {0}, dip = {1}".format(width, dip))
		ax1.set_xlabel('')
		ax1.tick_params(axis='x',which='both', bottom='off', labelbottom='off')

		self._filterForStm(laser=1)
		if np.ma.is_masked(self.EEG.eeg):
			eeg = np.ma.compressed(self.EEG.eeg)
		else:
			eeg = self.EEG.eeg
		if laserFilter:
			sf = self.getStimFreq()
			fx = EE.filterForLaser(E=eeg, width=width, dip=dip, stimFreq=sf)
		else:
			fx = eeg
		ax2 = super(OptoClusterSummary, self).plotEEGPower(E=fx, sample_rate=250.0, freqBand=(6,12), ax=ax2)
		ax2.set_title("Laser on\nwidth = {0}, dip = {1}".format(width, dip))

	def getStimFreq(self):
		'''
		Looks in the STM dict and calculates the stimulation frequency in Hz
		Note that at the moment this only returns the first phase that isn't a
		'Pause (no stimulation)' phase i.e. if multiple stimulation phases are 
		used this (and other functions) will need re-writing
		'''
		if self.STM:
			for phase in self.STM['stim_params']:
				if 'Pause' not in self.STM['stim_params'][phase]['name']:
					pulsePause = self.STM['stim_params'][phase]['pulsePause']
					pulseWidth = self.STM['stim_params'][phase]['pulseWidth']
					stimFreq = (float(pulseWidth) / pulsePause) * 100
					return stimFreq

	def read_yaml(self):
		"""
		method to read data in from a yaml file
		"""
		yaml_file = r'/media/robin/data/Dropbox/Science/Analysis/Mouse optogenetics/SST_Cre_grid_cell_project/grid_cell.yaml'
		stream = open(yaml_file, 'r')
		yaml_data = yaml.load_all(stream)
		return yaml_data

	def plot(self):
		"""
		main plotting method
		"""
		savedir = r'/home/robin/Desktop'
		plot_items = (['plotClusterSpace', 'plotWaveforms', 'plotXCorr',
					  'plotEEGPower', 'plotMap', 'plotFullSAC',
					  'plotRaster', 'plot_event_EEG'])
		n_rows = 4
		n_cols = 2
		for i, tet_clust in enumerate(zip(self.tetrodes, self.clusters)):
			fig = plt.figure(figsize=(4.135, 11.6))  # A4 landscape
			for idx, item in enumerate(plot_items):
				ax = fig.add_subplot(n_rows, n_cols, idx+1, frame_on=True)
				ax.axis('off')
				rect = ax.get_position().bounds
				if 'XCorr' in item:
					ax.axis('on')
				if np.logical_or('Cluster' in item, 'Wave' in item):
					ax.axis(frame_on=False)
					eval('self.' + item + '(tetrode=tet_clust[0], clusters=tet_clust[1], ax=ax, figure=fig)')
				elif np.logical_or('Raster' in item, 'EEG' in item):
					ax.axis(frame_on=True)
					ax.axis('on')
					eval("self." + item + "(tetrode=tet_clust[0], clusters=tet_clust[1],  ax=ax)")
				else:
					eval('self.' + item + '(tetrode=tet_clust[0], clusters=tet_clust[1], ax=ax)')
			fig.text(0.5, 0.95, (os.path.split(self.filename_root)[-1] + '\nTetrode: ' +
								str(tet_clust[0]) + ' Cluster: ' + str(tet_clust[1])),
								transform = fig.transFigure, fontsize=20,
								va='top', ha='center')
			fig.subplots_adjust(hspace=0.5, wspace=0.5)
			fig.savefig(os.path.join(savedir, (os.path.split(self.filename_root)[-1] + '_t' + str(tet_clust[0])
												+ '_c' + str(tet_clust[1]) + '.png')))
		plt.show()