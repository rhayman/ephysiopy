import numpy as np

"""
A dictionary containing parameters for the phase precession analysis
"""
phase_precession_config = {
    "pos_sample_rate": 50,
    "lfp_sample_rate": 250,
    "cms_per_bin": 1,  # bin size gets calculated in Ratemap
    "ppm": 445,
    "field_smoothing_kernel_len": 7,
    "field_smoothing_kernel_sigma": 5,
    # minimum firing rate - values below this are discarded (turned to 0)
    "field_threshold": 0.5,
    # field threshold percent - fed into fieldcalcs.local_threshold as prc
    "field_threshold_percent": 20,
    # fractional limit for restricting fields size
    "area_threshold": 0.01,
    # making the bins_per_cm value <1 leads to truncation of xy values
    # on unit circle
    "bins_per_cm": 1,
    "convert_xy_2_cm": True,
    # defines start/ end of theta cycle
    "allowed_min_spike_phase": np.pi,
    # percentile power below which theta cycles are rejected
    "min_power_percent_threshold": 0,
    # theta bands for min / max cycle length
    "min_theta": 6,
    "max_theta": 12,
    # kernel length for smoothing speed (boxcar)
    "speed_smoothing_window_len": 15,
    # cm/s - original value = 2.5; lowered for mice
    "minimum_allowed_run_speed": 0.5,
    "minimum_allowed_run_duration": 1,  # in seconds
    "min_spikes": 1,  # min allowed spikes per run
    # instantaneous firing rate (ifr) smoothing constant
    "ifr_smoothing_constant": 1.0 / 3,
    "spatial_lowpass_cutoff": 3,
    "ifr_kernel_len": 1,  # ifr smoothing kernal length
    "ifr_kernel_sigma": 0.5,
    "bins_per_second": 50,  # bins per second for ifr smoothing
}


"""
A list of the regressors that can be used in the phase precession analysis
"""
all_regressors = [
    "spk_numWithinRun",
    "pos_exptdRate_cum",
    "pos_instFR",
    "pos_timeInRun",
    "pos_d_cum",
    "pos_d_meanDir",
    "pos_d_currentdir",
    "spk_thetaBatchLabelInRun",
]
