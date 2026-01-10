import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.signal import argrelextrema
from skimage.segmentation import watershed
from ephysiopy.io.recording import AxonaTrial
from ephysiopy.common.rhythmicity import LFPOscillations
from ephysiopy.common.phasecoding import get_bad_cycles
from ephysiopy.common.statscalcs import (
    CircStatsResults,
    circCircCorrTLinear,
    circRegress,
    RegressionResults,
)
from ephysiopy.common.utils import VariableToBin, BinnedData, flatten_list
from ephysiopy.common.fieldcalcs import (
    filter_for_speed,
    filter_runs,
    fancy_partition,
    simple_partition,
)
from ephysiopy.common.fieldproperties import fieldprops, LFPSegment, FieldProps
from ephysiopy.phase_precession.config import phase_precession_config

subaxis_title_fontsize = 10


class phasePrecessionND(object):
    """
    Performs phase precession analysis for single unit data

    Mostly a rip-off of code written by Ali Jeewajee for his paper on
    2D phase precession in place and grid cells [1]_

    .. [1] Jeewajee A, Barry C, Douchamps V, Manson D, Lever C, Burgess N.
        Theta phase precession of grid and place cell firing in open
        environments.
        Philos Trans R Soc Lond B Biol Sci. 2013 Dec 23;369(1635):20120532.
        doi: 10.1098/rstb.2012.0532.

    Parameters
    ----------
    T : AxonaTrial (or OpenEphysBase eventually)
        The trial object holding position, LFP, spiking and ratemap stuff
    cluster : int
        the cluster to examine
    channel : int
        The channel the cluster was recorded on
    pp_config : dict
        Contains parameters for running the analysis.
        See phase_precession_config dict in ephysiopy.common.eegcalcs
    regressors : list
        A list of the regressors to use in the analysis

    Attributes
    ----------
    orig_xy : np.ndarray
        The original position data
    spike_ts : np.ndarray
        The spike timestamps
    regressors : dict
        A dictionary containing the regressors and their values
    alpha : float
        The alpha value for hypothesis testing
    hyp : int
        The hypothesis to test
    conf : bool
        Whether to calculate confidence intervals
    eeg : np.ndarray
        The EEG signal
    min_theta : int
        The minimum theta frequency
    max_theta : int
        The maximum theta frequency
    filteredEEG : np.ndarray
        The filtered EEG signal
    phase : np.ndarray
        The phase of the EEG signal
    phaseAdj : np.ma.MaskedArray
        The adjusted phase of the EEG signal as a masked array
    spike_times_in_pos_samples : np.ndarray
        The spike times in position samples (vector with length = npos)
    spk_weights : np.ndarray
        The spike weights (vector with length = npos)
    """

    def __init__(
        self,
        T: AxonaTrial,
        cluster: int,
        channel: int,
        pp_config: dict = phase_precession_config,
        regressors=None,
        **kwargs,
    ):
        if not T.PosCalcs:
            T.load_pos_data()
        if not T.EEGCalcs:
            T.load_lfp()
        if not T.RateMap:
            T.initialise()

        self.trial = T
        self.cluster = cluster
        self.channel = channel
        self._regressors = regressors
        self._binning_var = kwargs.get("var2bin", VariableToBin.XY)

        # ---------- Set up the parameters ----------
        # this adds, as attributes, the parameters defined in the
        # pp_config dictionary
        self.update_config(pp_config)
        self.update_config(kwargs)

        # ratemap params...
        self.trial.RateMap.smooth_sz = self.field_smoothing_kernel_len

        # values for the regression
        self.nshuffles = 1000
        self.alpha = 0.05
        self.hyp = 0
        self.conf = True

        # LFP params...
        self.eeg = T.EEGCalcs.sig
        self.lfp_fs = T.EEGCalcs.fs
        L = LFPOscillations(self.eeg, self.lfp_fs)
        self.min_theta = pp_config["min_theta"]
        self.max_theta = pp_config["max_theta"]
        FP = L.getFreqPhase(self.eeg, [self.min_theta, self.max_theta], 2)
        self.filteredEEG = FP.filt_sig
        self.phase = FP.phase
        self.phaseAdj = np.ma.MaskedArray

        # Some spiking params...
        spk_times_in_pos_samples = T.get_binned_spike_times(cluster, channel)
        spk_times_in_pos_samples = np.ravel(spk_times_in_pos_samples).astype(int)
        spk_weights = np.bincount(
            spk_times_in_pos_samples, minlength=len(T.PosCalcs.xyTS)
        )
        self.spike_times_in_pos_samples = spk_times_in_pos_samples
        self.spk_weights = spk_weights

        self.spike_ts = T.get_spike_times(cluster, channel)

    @property
    def binning_var(self):
        return self._binning_var

    @binning_var.setter
    def binning_var(self, val):
        self._binning_var = val

    @property
    def regressors(self):
        return self._regressors

    @regressors.setter
    def regressors(self, val):
        self._regressors = val

    @property
    def spike_eeg_idx(self):
        return (self.spike_ts * self.lfp_fs).astype(int)

    def update_config(self, pp_config):
        """Update the relevant pp_config values"""
        [
            setattr(self, attribute, pp_config[attribute])
            for attribute in pp_config.keys()
            if attribute in pp_config
        ]

    def do_regression(self, **kwargs):
        """
        Wrapper function for doing the actual regression which has multiple
        stages.

        Specifically here we partition fields into sub-fields, get a bunch of
        information about the position, spiking and theta data and then
        do the actual regression.

        **kwargs
            do_plot : bool
                whether to plot the results of the regression(s)
            ax : matplotlib.mat
                The axes to plot into

        """
        do_plot = kwargs.get("plot", False)
        ax = kwargs.get("ax", None)

        if "binned_data" in kwargs.keys():
            binned_data = kwargs.get("binned_data")
        else:
            if self.binning_var.value == VariableToBin.XY.value:
                binned_data = self.trial.get_rate_map(self.cluster, self.channel)
            elif self.binning_var.value == VariableToBin.X.value:
                binned_data = self.trial.get_linear_rate_map(
                    self.cluster, self.channel, var_type=self.binning_var
                )

        # split into runs
        # method = kwargs.get("method", "field")
        field_properties = self.get_pos_props(binned_data, self.binning_var, **kwargs)

        # get theta cycles, amplitudes, phase etc
        field_properties = self.get_theta_props(field_properties)

        reg_results = self.get_phase_reg_per_field(field_properties)

        self.do_correlation(reg_results, plot=do_plot, ax=ax)

    def get_pos_props(
        self,
        binned_data: BinnedData = None,
        var_type: VariableToBin = VariableToBin.XY,
        **kwargs,
    ) -> list:
        """
        Uses the output of fancy_partition and returns vectors the same
        length as pos.

        Parameters
        ----------
        binned_data - BinnedData
            optional BinnedData instance. Will be calculated here
            if not given

        var_type - VariableToBin
            defines if we are dealing with 1- or 2D data essentially

        **kwargs - keywords
            valid kwargs:
                field_threshold - see fancy_partition()
                field_threshold_percent - see fancy_partition()
                area_threshold - see fancy_partition()
        Returns
        -------
        list of FieldProps
            A list of FieldProps instances
            (see ephysiopy.common.fieldcalcs.FieldProps)
        """
        if var_type.value == VariableToBin.XY.value:
            posdata = self.trial.PosCalcs.xy
        elif var_type.value == VariableToBin.X.value:
            posdata = self.trial.PosCalcs.xy[0]
        elif var_type.value == VariableToBin.PHI.value:
            posdata = self.trial.PosCalcs.phi

        self.binning_var = var_type

        if binned_data is None:
            if var_type.value == VariableToBin.XY.value:
                binned_data = self.trial.get_rate_map(self.cluster, self.channel)
            elif var_type.value == VariableToBin.X.value:
                binned_data = self.trial.get_linear_rate_map(
                    self.cluster, self.channel, var_type=var_type
                )
            elif var_type.value == VariableToBin.PHI.value:
                binned_data = self.trial.get_linear_rate_map(
                    self.cluster, self.channel, var_type=var_type
                )

        # user might want to override the values for partitioning
        # the field based on mean rate for example so add the
        # option to provide as kwargs

        field_threshold_percent = kwargs.get(
            "field_threshold_percent", self.field_threshold_percent
        )
        field_threshold = kwargs.get("field_threshold", self.field_threshold)
        area_threshold = kwargs.get("area_threshold", self.area_threshold)

        partition_method = kwargs.get("partition_method", "fancy")
        print(f"Partitioning fields using the {partition_method} method")

        if partition_method == "simple":
            _, _, labels, _ = simple_partition(
                binned_data,
                rate_threshold_prc=field_threshold_percent,
            )
        else:
            _, _, labels, _ = fancy_partition(
                binned_data,
                field_threshold_percent,
                field_threshold,
                area_threshold,
            )

        # The large number of bins combined with the super-smoothed ratemap
        # will lead to fields labelled with lots of small holes in. Fill those
        # gaps in here and calculate the perimeter of the fields based on that
        # labelled image
        labels, _ = ndimage.label(ndimage.binary_fill_holes(labels))

        # This is the main call to get the field properties
        # for each field found in the ratemap

        method = kwargs.pop("method", "field")

        field_props = fieldprops(
            labels,
            binned_data,
            self.trial.get_spike_times(self.cluster, self.channel),
            posdata,
            sample_rate=self.trial.PosCalcs.sample_rate,
            method=method,
            **kwargs,
        )

        field_props = filter_runs(
            field_props,
            ["duration", "n_spikes"],
            [np.greater, np.greater_equal],
            [self.minimum_allowed_run_duration, 2],
        )
        field_props = filter_for_speed(field_props, self.minimum_allowed_run_speed)

        # Smooth the runs before calculating other metrics
        [
            f.smooth_runs(
                self.ifr_smoothing_constant,
                self.spatial_lowpass_cutoff,
                self.pos_sample_rate,
            )
            for f in field_props
        ]

        self.field_properties = field_props

        print("Filtered runs after position processing...")
        [print(f) for f in field_props]

        return field_props

    def get_theta_props(self, field_props: list[FieldProps]):
        """
        Processes the LFP data and inserts into each run within each field
        a segment of LFP data that has had its phase and amplitude extracted
        as well as some other data

        Parameters
        ----------
        field_props : list[FieldProps]
            A list of FieldProps instances

        Returns
        -------
        list of FieldProps
            The amended list with LFP data added to each run for each field

        """
        # phase = np.ma.MaskedArray(self.phase, mask=True)
        # get indices of spikes into eeg
        # spkEEGIdx = self.spike_eeg_idx
        # spkPhase = phase.copy()
        # unmask the valid entries
        # spkPhase.mask[spkEEGIdx] = False

        # cycleLabel, phaseAdj = get_cycle_labels(
        #     spkPhase, self.allowed_min_spike_phase)
        minima = argrelextrema(self.phase, np.less)[0]
        markers = np.bincount(minima, minlength=len(self.phase))
        markers = np.cumsum(markers)
        cycleLabel = watershed(self.phase, markers=markers)
        isNegFreq = np.diff(np.unwrap(self.phase)) < 0
        isNegFreq = np.append(isNegFreq, isNegFreq[-1])
        # phaseAdj = phaseAdj.data

        isBad = get_bad_cycles(
            self.filteredEEG,
            isNegFreq,
            cycleLabel,
            self.min_power_percent_threshold,
            self.min_theta,
            self.max_theta,
            self.lfp_fs,
        )
        self.bad_cycles = isBad
        self.cycleLabel = cycleLabel
        lfp_to_pos_ratio = int(self.lfp_fs / self.pos_sample_rate)
        spike_times = self.trial.get_spike_times(self.cluster, self.channel)

        for field in field_props:
            for run in field.runs:
                lfp_slice = slice(
                    run.slice.start * lfp_to_pos_ratio,
                    run.slice.stop * lfp_to_pos_ratio,
                )
                lfp_segment = LFPSegment(
                    run,
                    field.label,
                    run.label,
                    lfp_slice,
                    spike_times=spike_times,
                    mask=isBad[lfp_slice],
                    signal=self.eeg[lfp_slice],
                    filtered_signal=self.filteredEEG[lfp_slice],
                    phase=self.phase[lfp_slice],
                    cycle_label=cycleLabel[lfp_slice],
                    sample_rate=self.lfp_fs,
                )
                run.lfp = lfp_segment

        # filter again as the lfp data might have masked
        # some spikes that can lead to runs with no spikes
        # in them

        field_props = filter_runs(
            field_props,
            ["n_spikes"],
            [np.greater_equal],
            [2],
        )
        self.field_properties = field_props

        print("Filtered runs after theta processing...")
        [print(f) for f in field_props]

        return field_props

    def get_phase_reg_per_field(self, fp: list[FieldProps], **kwargs) -> dict:
        """
        Extracts the phase and all regressors for all runs through each
        field separately

        Parameters
        ----------
        fp : list
            A list of FieldProps instances

        Returns
        -------
        dict
            two-level dictionary holding regression results per field
            first level keys are field number
            second level are the regressors (current_dir etc)
            items in the second dict are the regression results
        """
        regressors = kwargs.get("regressors", self.regressors)

        results = dict.fromkeys([f.label for f in fp])
        for field in fp:
            results[field.label] = {}
            results[field.label]["phase"] = np.array(
                flatten_list([run.lfp.mean_spiking_var().ravel() for run in field.runs])
            )
            for regressor in regressors:
                match regressor:
                    case "pos_timeInRun":
                        vals = field.mean_spiking_var("cumulative_time")
                    case "spk_numWithinRun":
                        vals = field.mean_spiking_var("spike_num_in_run")
                    case "pos_d_normed_x":
                        vals = field.mean_spiking_var("normed_x")
                    case "pos_d_currentdir":
                        vals = field.mean_spiking_var("current_direction")
                    case "pos_d_cum":
                        vals = field.mean_spiking_var("cumulative_distance")
                    case "pos_exptdRate_cum":
                        vals = field.mean_spiking_var("expected_spikes")
                    case "pos_instFR":
                        vals = field.mean_spiking_var("instantaneous_firing_rate")

                results[field.label][regressor] = vals
        return results

    def do_correlation(
        self, phase_regressors: dict, **kwargs
    ) -> list[RegressionResults]:
        """
        Do the regression(s) for each regressor in the phase_regressors dict,
        optionally plotting the results of the regression

        Parameters
        ----------
        phase_regressors : dict
            Dictionary with keys as field label (1,2 etc), each key contains a
            dictionary with keys 'phase' and optional nummbers of regressors
        plot : bool
            Whether to plot the regression results

        Notes
        -----
        This collapses across fields and does the regression for all
        phase and regressor values
        """
        # extract field label and the list of the regressors
        field_ids = list(phase_regressors.keys())
        regressors = list(phase_regressors[field_ids[0]].keys())
        regressors = [r for r in regressors if "phase" not in r]

        # extract the phase
        phase = [phase_regressors[field]["phase"] for field in field_ids]
        phase = np.array(flatten_list(phase))

        results = []

        for reg in regressors:
            i_reg = np.concatenate(
                flatten_list([phase_regressors[f][reg] for f in field_ids])
            )
            slope, intercept = circRegress(i_reg, phase)

            mn_reg = np.mean(i_reg)
            i_reg -= mn_reg
            mxx = np.max(np.abs(i_reg)) + np.spacing(1)
            i_reg /= mxx

            theta = np.mod(np.abs(slope) * i_reg, 2 * np.pi)

            result = circCircCorrTLinear(
                theta, phase, self.nshuffles, self.alpha, self.hyp, self.conf
            )
            result.slope = slope
            result.intercept = intercept
            print(f"\n{reg}:\n{result}")

            R = RegressionResults(reg, phase, i_reg, result)
            results.append(R)

            if kwargs.get("plot", False):
                self.plot_regressor(reg, i_reg, phase, result)

        return results

    def plot_regressor(
        self,
        regressor: str,
        vals: np.ndarray,
        pha: np.ndarray,
        result: CircStatsResults,
        ax=None,
    ):
        """
        Plot the regressor against the phase

        Parameters
        ----------
        regressor : str
            The regressor to plot
        ax : matplotlib.axes.Axes
            The axes to plot on

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plot
        """
        assert regressor in self.regressors
        if ax is None:
            fig = plt.figure(figsize=(3, 5))
            ax = fig.add_subplot(111)
        else:
            ax = ax
        slope = result.slope
        intercept = result.intercept
        mm = (0, -4 * np.pi, -2 * np.pi, 2 * np.pi, 4 * np.pi)
        for m in mm:
            ax.plot((-1, 1), (-slope + intercept + m, slope + intercept + m), "r", lw=3)
            ax.plot(vals, pha + m, "k.")
        ax.set_xlim(-1, 1)
        xtick_locs = np.linspace(-1, 1, 3)
        ax.set_xticks(xtick_locs, list(map(str, xtick_locs)))
        ax.set_yticks(sorted(mm), ["-4π", "-2π", "0", "2π", "4π"])
        ax.set_ylim(-2 * np.pi, 4 * np.pi)
        title_str0 = f"{regressor} vs phase: slope = {slope:.2f}"
        title_str1 = f"\nintercept = {intercept:.2f}"
        title_str2 = f"\np_shuffled = {result.p_shuffled:.2f}"
        title_str = title_str0 + title_str1 + title_str2
        ax.set_title(title_str, fontsize=subaxis_title_fontsize)
        ax.set_ylabel("Phase", fontsize=subaxis_title_fontsize)
        ax.set_xlabel("Normalised position", fontsize=subaxis_title_fontsize)
        return ax
