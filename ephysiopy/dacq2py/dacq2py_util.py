import os
import warnings
from collections import OrderedDict

import matplotlib.pylab as plt
from ephysiopy.common.ephys_generic import PosCalcsGeneric
from ephysiopy.dacq2py import axonaIO
from ephysiopy.dacq2py.tetrode_dict import TetrodeDict
from ephysiopy.visualise.plotting import FigureMaker

warnings.filterwarnings("ignore", message="divide by zero encountered in int_scalars")
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings(
    "ignore",
    message="Casting complex values to real\
        discards the imaginary part",
)


class AxonaTrial(FigureMaker):
    def __init__(self, fileset_root: str, **kwargs):
        # fileset_root here is the fully qualified path to the .set file
        assert os.path.exists(fileset_root)
        self.fileset_root = fileset_root
        self.common_name = os.path.splitext(fileset_root)[0]
        self.__settings = None  # will become a dict ~= the .set file
        self.__ppm = None
        self.xy = None
        self.xyTS = None
        self.__EEG = None
        self.__EGF = None
        self.TETRODE = TetrodeDict(self.common_name, volts=True)
        self.__STM = None
        self.ttl_data = None
        self.__ttl_timestamps = None
        self.recording_start_time = 0
        self.data_loaded = False

    def load(self, *args, **kwargs):
        """
        Minially, there should be at least a .set file
        Other files (.eeg, .pos, .stm, .1, .2 etc) are essentially optional

        """
        if self.settings is not None:
            print("Loaded .set file")
        # Give ppm a default value from the set file...
        self.ppm = int(self.settings["tracker_pixels_per_metre"])
        # ...with the option to over-ride
        if "ppm" in kwargs:
            self.ppm = kwargs["ppm"]

        # ------------------------------------
        # ------------- Pos data -------------
        # ------------------------------------
        if self.xy is None:
            try:
                AxonaPos = axonaIO.Pos(self.common_name)
                P = PosCalcsGeneric(
                    AxonaPos.led_pos[:, 0],
                    AxonaPos.led_pos[:, 1],
                    cm=True,
                    ppm=self.ppm,
                )
                P.postprocesspos(tracker_params={"AxonaBadValue": 1023})
                self.xy = P.xy
                self.xyTS = AxonaPos.ts - AxonaPos.ts[0]
                self.dir = P.dir
                self.speed = P.speed
                self.pos_sample_rate = AxonaPos.getHeaderVal(
                    AxonaPos.header, "sample_rate"
                )

                print("Loaded .pos file")
            except IOError:
                print("Couldn't load the pos data")

    @property
    def ppm(self):
        return self.__ppm

    @ppm.setter
    def ppm(self, value):
        self.__ppm = value

    @property
    def settings(self):
        if self.__settings is None:
            try:
                from ephysiopy.dacq2py.axonaIO import IO

                settings_io = IO()
                self.__settings = settings_io.getHeader(self.fileset_root)
            except IOError:
                print(".set file not loaded")
                self.__settings = None
        return self.__settings

    @settings.setter
    def settings(self, value):
        self.__settings = value

    @property
    def EEG(self):
        if self.__EEG is None:
            try:
                self.__EEG = axonaIO.EEG(self.common_name)
            except IOError:
                print("Could not load EEG file")
                self.__EEG = None
        return self.__EEG

    @EEG.setter
    def EEG(self, value):
        self.__EEG = value

    @property
    def EGF(self):
        if self.__EGF is None:
            try:
                self.__EGF = axonaIO.EEG(self.common_name, egf=1)
            except IOError:
                print("Could not load EGF file")
                self.__EGF = None
        return self.__EGF

    @EGF.setter
    def EGF(self, value):
        self.__EGF = value

    @property
    def ttl_timestamps(self):
        return self.STM["on"]

    @ttl_timestamps.setter
    def ttl_timestamps(self, value):
        self.__ttl_timestamps = value

    @property
    def STM(self):
        """
        Returns
        -------
        ephysiopy.dacq2py.axonaIO.Stim:
            Stimulation data and header + some extras parsed from pos, eeg
            and set files
        """
        if self.__STM is None:
            try:
                self.__STM = axonaIO.Stim(self.common_name)
                """
                update the STM dict with some relevant values from
                the .set file and the headers of the eeg and pos files
                """
                from ephysiopy.dacq2py.axonaIO import IO

                io = IO()
                posHdr = io.getHeader(self.common_name + ".pos")
                eegHdr = io.getHeader(self.common_name + ".eeg")
                self.__STM["posSampRate"] = io.getHeaderVal(posHdr, "sample_rate")
                self.__STM["eegSampRate"] = io.getHeaderVal(eegHdr, "sample_rate")
                try:
                    egfHdr = io.getHeader(self.common_name + ".egf")
                    self.__STM["egfSampRate"] = io.getHeaderVal(egfHdr, "sample_rate")
                except Exception:
                    pass
                # get into ms
                self.settings
                stim_pwidth = int(self.settings["stim_pwidth"]) / int(1000)
                self.__STM["off"] = self.__STM["on"] + int(stim_pwidth)
                setattr(self, "ttl_timestamps", self.__STM["on"])
                """
                There are a set of key / value pairs in the set file that
                correspond to the patterns/ protocols specified in the
                Stimulator menu in DACQ. Extract those items now...
                There are five possibe "patterns" that can be used in a trial.
                Patterns consist of either "Pause (no stimulation)" or some
                user-defined stimulation pattern.
                Whether or not one of the five was used is specified in
                "stim_patternmask_n" where n is 1-5.
                Confusingly in dacqUSB these 5 things are called "Protocols"
                accessed from the menu Stimulator/Protocols...
                Within that window they are actually called "Phase 1",
                "Phase 2" etc.

                In dacqUSB nomencalture the pattern is actually the
                stimulation you want to apply i.e. 10ms pulse every 150ms
                or whatever. The "pattern" is what is applied
                within every Phase.
                """
                # phase_info : a dict for each phase that is active
                phase_info_keys = [
                    "startTime",
                    "duration",
                    "name",
                    "pulseWidth",
                    "pulseRatio",
                ]
                phase_info = dict.fromkeys(phase_info_keys, None)
                stim_dict = {}
                stim_patt_dict = {}
                for k, v in self.settings.items():
                    if k.startswith("stim_patternmask_"):
                        if int(v) == 1:
                            # get the number of the phase
                            phase_num = k[-1]
                            stim_dict["Phase_" + phase_num] = phase_info.copy()
                    if k.startswith("stim_patt_"):
                        stim_patt_dict[k] = v
                self.patt_dict = stim_patt_dict
                for k, v in stim_dict.items():
                    phase_num = k[-1]
                    stim_dict[k]["duration"] = int(
                        self.settings["stim_patterntimes_" + phase_num]
                    )
                    phase_name = self.settings["stim_patternnames_" + phase_num]
                    stim_dict[k]["name"] = phase_name
                    if not (phase_name.startswith("Pause")):
                        # find the matching string in the stim_patt_dict
                        for _, vv in stim_patt_dict.items():
                            split_str = vv.split('"')
                            patt_name = split_str[1]
                            if patt_name == phase_name:
                                ss = split_str[2].split()
                                stim_dict[k]["pulseWidth"] = int(ss[0])
                                stim_dict[k]["pulsePause"] = int(ss[2])
                # make the dict ordered by Phase number
                self.STM["stim_params"] = OrderedDict(sorted(stim_dict.items()))
            except IOError:
                self.__STM = None
        return self.__STM

    @STM.setter
    def STM(self, value):
        self.__STM = value

    def plotSummary(self, tetrode: int, cluster: int, **kwargs):
        ts = self.TETRODE.get_spike_samples(tetrode, cluster)  # in seconds
        ax = self.makeSummaryPlot(ts, **kwargs)
        plot = True
        if "plot" in kwargs:
            plot = kwargs.pop("plot")
        if plot:
            plt.show()
        return ax

    def plotSpikesOnPath(self, tetrode=None, cluster=None, **kwargs):
        ts = None
        if tetrode is not None:
            ts = self.TETRODE.get_spike_samples(tetrode, cluster)  # in seconds
        plot = True
        if "plot" in kwargs:
            plot = kwargs.pop("plot")
        ax = self.makeSpikePathPlot(ts, **kwargs)
        if plot:
            plt.show()
        return ax

    def plotRateMap(self, tetrode, cluster, **kwargs):
        ts = self.TETRODE.get_spike_samples(tetrode, cluster)  # in seconds
        ax = self.makeRateMap(ts)
        plot = True
        if "plot" in kwargs:
            plot = kwargs.pop("plot")
        if plot:
            plt.show()
        return ax

    def plotHDMap(self, tetrode, cluster, **kwargs):
        ts = self.TETRODE.get_spike_samples(tetrode, cluster)  # in seconds
        ax = self.makeHDPlot(ts, ax=None, **kwargs)
        plot = True
        if "plot" in kwargs:
            plot = kwargs.pop("plot")
        if plot:
            plt.show()
        return ax

    def plotSAC(self, tetrode, cluster, **kwargs):
        ts = self.TETRODE.get_spike_samples(tetrode, cluster)  # in seconds
        ax = self.makeSAC(ts, ax=None, **kwargs)
        plot = True
        if "plot" in kwargs:
            plot = kwargs.pop("plot")
        if plot:
            plt.show()
        return ax

    def plotSpeedVsRate(self, tetrode, cluster, **kwargs):
        ts = self.TETRODE.get_spike_samples(tetrode, cluster)  # in seconds
        ax = self.makeSpeedVsRatePlot(ts, ax=None, **kwargs)
        plot = True
        if "plot" in kwargs:
            plot = kwargs.pop("plot")
        if plot:
            plt.show()
        return ax

    def plotSpeedVsHeadDirection(self, tetrode, cluster, **kwargs):
        ts = self.TETRODE.get_spike_samples(tetrode, cluster)  # in seconds
        ax = self.makeSpeedVsHeadDirectionPlot(ts, ax=None, **kwargs)
        plot = True
        if "plot" in kwargs:
            plot = kwargs.pop("plot")
        if plot:
            plt.show()
        return ax

    def plotEEGPower(self, eeg_type="eeg", **kwargs):
        from ephysiopy.common.ephys_generic import EEGCalcsGeneric

        if "eeg" in eeg_type:
            E = EEGCalcsGeneric(self.EEG.sig, self.EEG.sample_rate)
        elif "egf" in eeg_type:
            E = EEGCalcsGeneric(self.EGF.sig, self.EGF.sample_rate)
        power_res = E.calcEEGPowerSpectrum()
        ax = self.makePowerSpectrum(
            power_res[0],
            power_res[1],
            power_res[2],
            power_res[3],
            power_res[4],
            **kwargs
        )
        plot = True
        if "plot" in kwargs:
            plot = kwargs.pop("plot")
        if plot:
            plt.show()
        return ax

    def plotXCorr(self, tetrode, cluster, **kwargs):
        ts = self.TETRODE.get_spike_samples(tetrode, cluster)  # in seconds
        ax = self.makeXCorr(ts)
        plot = True
        if "plot" in kwargs:
            plot = kwargs.pop("plot")
        if plot:
            plt.show()
        return ax

    def plotRaster(self, tetrode, cluster, **kwargs):
        ts = self.TETRODE.get_spike_samples(tetrode, cluster)  # in seconds
        self.ttl_timestamps = self.STM["on"]
        ax = self.makeRaster(ts, **kwargs)
        return ax

    '''
    def klustakwik(self, d):
        """
        Calls two methods below (kluster and getPC) to run klustakwik on
        a given tetrode with nFet number of features (for the PCA)

        Parameters
        ----------
        d : dict
            Specifies the vector of features to be used in clustering.
            Each key is the identity of a tetrode (i.e. 1, 2 etc)
            and the values are the features used to do the clustering for
            that tetrode (i.e. 'PC1', 'PC2', 'Amp' (amplitude) etc
        """
        from ephysiopy.common.spikecalcs.SpikeCalcsAxona import getParam
        from ephysiopy.dacq2py.cluster import Kluster

        legal_values = ['PC1', 'PC2', 'PC3', 'PC4', 'Amp',
                        'Vt', 'P', 'T', 'tP', 'tT', 'En', 'Ar']
        reg = re.compile(".*(PC).*")  # check for number of principal comps
        # check for any input errors in whole dictionary first
        for i_tetrode in d.keys():
            for v in d[i_tetrode]:
                if v not in legal_values:
                    raise ValueError('Could not find %s in %s' % (
                        v, legal_values))
        # iterate through features and see what the max principal component is
        for i_tet in d.keys():
            pcs = [m.group(0) for t in d[i_tet] for m in [reg.search(t)] if m]
            waves = self.TETRODE[i_tet].waveforms
            princomp = None
            if pcs:
                max_pc = []
                for pc in pcs:
                    max_pc.append(int(pc[2]))
                num_pcs = np.max(max_pc)  # get max number of prin comps
                princomp = getParam(
                    waves, param='PCA', fet=num_pcs)
                # Rearrange the output from PCA calc to match the
                # number of requested principal components
                inds2keep = []
                for m in max_pc:
                    inds2keep.append(np.arange((m-1)*4, (m)*4))
                inds2keep = np.hstack(inds2keep)
                princomp = np.take(princomp, inds2keep, axis=1)
            out = []
            for value in d[i_tet]:
                if 'PC' not in value:
                    out.append(
                        getParam(waves, param=value))
            if princomp is not None:
                out.append(princomp)
            out = np.hstack(out)

            c = Kluster(self.common_name, i_tet, out)
            c.make_fet()
            mask = c.get_mask()
            c.make_fmask(mask)
            c.kluster()
    '''
