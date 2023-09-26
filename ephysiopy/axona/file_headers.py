from dataclasses import dataclass, field, fields
from abc import ABC

'''
The only exception to lots of the common headers etc in 
the Axona file collection is the cut file, dealt with here

When converting data from KiloSort/ OE to Axona format there
might be a problem with the number of clusters. Axona limits 
you to 31 (including the 0 cluster), where KS can go into the
hundreds

If we're dealing with tetrode data though it's unlikely that
nclusters in reality will go much above 30. It might be 
necessary to rename the clusters on a per tetrode basis to
limit the range of cluster values within a tetrode to 0-30
'''


def make_cut_header(n_clusters: int = 31,
                    n_channels: int = 4,
                    n_params: int = 2):
    cut_header = [('n_clusters', n_clusters),
                  ('n_channels', n_channels),
                  ('n_params', n_params),
                  ('times_used_in_Vt', '    0'*n_channels)]
    return dict(cut_header)


def make_cluster_cut_entries(n_clusters: int = 31,
                             n_channels: int = 4,
                             n_params: int = 2):
    n_zeros = n_channels * n_params
    output = ""
    for c in range(n_clusters):
        output = \
               output + " cluster: " + str(c) + " center:" + "   0"*n_zeros + \
               "\n" + ' '*15 + "min:" + "   0"*n_zeros + \
               "\n" + ' '*15 + "max:" + "   0"*n_zeros + "\n"
    return output


common_entries = [
    ('trial_date', None), ('trial_time', None),
    ('experimenter', None), ('comments', None), ('duration', None)
]


def make_common_entries():
    return dict(common_entries)


@dataclass
class AxonaHeader(ABC):
    common: dict = field(default_factory=make_common_entries)

    def __setattr__(self, name, value):
        for f in fields(self):
            if issubclass(f.type, dict):
                object.__setattr__(self, name, value)
            else:
                return super().__setattr__(name, value)

    def print(self):
        for f in fields(self):
            if f.repr:
                if issubclass(f.type, dict):
                    for k, v in getattr(self, f.name).items():
                        if v is None:
                            v = ""
                        print(f"{k} {v}")
                else:
                    print(f"{f.name}")

# --------------------- pos headers --------------------


pos_entries = [
    ('min_x', None), ('max_x', None), ('min_y', None),
    ('max_y', None), ('window_min_x', None), ('window_max_x', None),
    ('window_min_y', None), ('window_max_y', None),
    ('sample_rate', None),
    ('pixels_per_metre', None), ('num_pos_samples', None),
    ('sw_version', '1.2.2.1'),
    ('num_colours', '4'), ('timebase', '50.0 hz'),
    ('bytes_per_timestamp', '4'),
    ('EEG_samples_per_position', '5'), ('bearing_colour_1', '210'),
    ('bearing_colour_2', '30'), ('bearing_colour_3', '0'),
    ('bearing_colour_4', '0'),
    ('pos_format', 't,x1,y1,x2,y2,numpix1,numpix2'),
    ('bytes_per_coord', '2')
]


def make_pos_entries():
    return dict(pos_entries)


@dataclass
class PosHeader(AxonaHeader):
    '''
    Empty .pos header class for Axona
    '''
    pos: dict = field(default_factory=make_pos_entries)


@dataclass
class CutHeader(AxonaHeader):
    common: dict = field(default_factory=make_cut_header)

# --------------------- eeg/ egf headers --------------------


lfp_entries = [
    ('sw_version', '1.1.0'),
    ('num_chans', '1'),
    ('sample_rate', None),
    ('bytes_per_sample', None)
]


@dataclass
class LFPHeader(AxonaHeader):

    _n_samples: str = field(default=None, repr=False)

    @property
    def n_samples(self):
        if self.lfp_entries['sample_rate'] is not None:
            if '4800' in self.lfp_entries['sample_rate']:
                return self.lfp_entries['num_EGF_samples']
            else:
                return self.lfp_entries['num_EEG_samples']

    @n_samples.setter
    def n_samples(self, value):
        if '4800' in self.lfp_entries['sample_rate']:
            self.lfp_entries['num_EGF_samples'] = value
        else:
            self.lfp_entries['num_EEG_samples'] = value


eeg_entries = [
    ('sample_rate', "250 hz"),
    ('num_EEG_samples', None),
    ('EEG_samples_per_position', '5'),
    ('bytes_per_sample', '1')
]


def make_eeg_entries():
    return {**dict(lfp_entries), **dict(eeg_entries)}


@dataclass
class EEGHeader(LFPHeader):
    lfp_entries: dict = field(
        default_factory=make_eeg_entries)


egf_entries = [
    ('sample_rate', "4800 hz"),
    ('num_EGF_samples', None),
    ('bytes_per_sample', '2')
]


def make_egf_entries():
    return {**dict(lfp_entries), **dict(egf_entries)}


@dataclass
class EGFHeader(LFPHeader):
    lfp_entries: dict = field(
        default_factory=make_egf_entries)


# --------------------- tetrode headers --------------------

tetrode_entries = [
    ('num_spikes', None),
    ('sw_version', '1.1.0'),
    ('num_chans', '4'),
    ('timebase', '96000'),
    ('bytes_per_timestamp',  '4'),
    ('samples_per_spike', '50'),
    ('sample_rate', '48000 Hz'),
    ('bytes_per_sample', '1'),
    ('spike_format', 't,ch1,t,ch2,t,ch3,t,ch4')
]


def make_tetrode_entries():
    return dict(tetrode_entries)


@dataclass
class TetrodeHeader(AxonaHeader):
    tetrode_entries: dict = field(
        default_factory=make_tetrode_entries)


# ------------------------------ set header ----------------------
# append 0 - 63 to each of these
entries_to_number = [
    ('gain_ch_', '0'),
    ('filter_ch_', '0'),
    ('a_in_ch_', '0'),
    ('b_in_ch_', '0'),
    ('mode_ch_', '0'),
    ('filtresp_ch_', '0'),
    ('filtkind_ch_', '0'),
    ('filtfreq1_ch_', '0'),
    ('filtfreq2_ch_', '0'),
    ('filtripple_ch_', '0'),
    ('filtdcblock_ch_', '0'),
    ('dispmode_ch_', '0'),
    ('channame_ch_', '0')
]

# append 1-64 for these
entries_to_number_one_indexed = [
    ('EEG_ch_', '0'),
    ('saveEEG_ch_', '0'),
    ('BPFEEG_ch_', '0')
]

# append 1-16 to these
entries_to_number_to_sixteen = [
    ('collectMask_', '0'),
    ('stereoMask_', '0'),
    ('monoMask_', '0'),
    ('EEGmap_', '0')
]
# append 1-3 to these
entries_to_number_to_three = [
    ('BPFrecord', '0'),
    ('BPFbit', '0'),
    ('BPFEEGin', '0')
]

# Replace the 1 with 1-4
entries_to_replace_one = [
    ('colmap_1_rmin', '0'),
    ('colmap_1_rmax', '0'),
    ('colmap_1_gmin', '0'),
    ('colmap_1_gmax', '0'),
    ('colmap_1_bmin', '0'),
    ('colmap_1_bmax', '0'),
    ('colactive_1', '0')
]

# append 1-9
entries_to_number_to_nine = [
    ('slot_chan_', '0')
]

# replace X with 1-17 and Y with 0-9
entries_groups = [
    ('groups_X_Y', '0')
]

singleton_entries = [
    ('second_audio', '0'),
    ('default_filtresp_hp', '0'),
    ('default_filtkind_hp', '0'),
    ('default_filtfreq1_hp', '0'),
    ('default_filtfreq2_hp', '0'),
    ('default_filtripple_hp', '0'),
    ('default_filtdcblock_hp', '0'),
    ('default_filtresp_lp', '0'),
    ('default_filtkind_lp', '0'),
    ('default_filtfreq1_lp', '0'),
    ('default_filtfreq2_lp', '0'),
    ('default_filtripple_lp', '0'),
    ('default_filtdcblock_lp', '0'),
    ('notch_frequency', '0'),
    ('ref_0', '0'),
    ('ref_1', '0'),
    ('ref_2', '0'),
    ('ref_3', '0'),
    ('ref_4', '0'),
    ('ref_5', '0'),
    ('ref_6', '0'),
    ('ref_7', '0'),
    ('trigger_chan', '0'),
    ('selected_slot', '0'),
    ('sweeprate', '0'),
    ('trig_point', '0'),
    ('trig_slope', '0'),
    ('threshold', '0'),
    ('leftthreshold', '0'),
    ('rightthreshold', '0'),
    ('aud_threshold', '0'),
    ('chan_group', '0'),
    ('BPFsyncin1', '0'),
    ('BPFrecordSyncin1', '0'),
    ('BPFunitrecord', '0'),
    ('BPFinsightmode', '0'),
    ('BPFcaladjust', '0'),
    ('BPFcaladjustmode', '0'),
    ('rawRate', '0'),
    ('RawRename', '0'),
    ('RawScope', '0'),
    ('RawScopeMode', '0'),
    ('lastfileext', '0'),
    ('lasttrialdatetime', '0'),
    ('lastupdatecheck', '0'),
    ('useupdateproxy', '0'),
    ('updateproxy', '0'),
    ('updateproxyid', '0'),
    ('updateproxypw', '0'),
    ('contaudio', '0'),
    ('mode128channels', '0'),
    ('modeanalog32', '0'),
    ('EEGdisplay', '0'),
    ('lightBearing_1', '0'),
    ('lightBearing_2', '0'),
    ('lightBearing_3', '0'),
    ('lightBearing_4', '0'),
    ('artefactReject', '0'),
    ('artefactRejectSave', '0'),
    ('remoteStart', '0'),
    ('remoteChan', '0'),
    ('remoteStop', '0'),
    ('remoteStopChan', '0'),
    ('endBeep', '0'),
    ('recordExtin', '0'),
    ('recordTracker', '0'),
    ('showTracker', '0'),
    ('trackerSerial', '0'),
    ('serialColour', '0'),
    ('recordVideo', '0'),
    ('dacqtrackPos', '0'),
    ('stimSerial', '0'),
    ('recordSerial', '0'),
    ('useScript', '0'),
    ('script', '0'),
    ('postProcess', '0'),
    ('postProcessor', '0'),
    ('postProcessorParams', '0'),
    ('sync_out', '0'),
    ('syncRate', '0'),
    ('autoTrial', '0'),
    ('numTrials', '0'),
    ('trialPrefix', '0'),
    ('autoPrompt', '0'),
    ('saveEGF', '0'),
    ('rejstart', '0'),
    ('rejthreshtail', '0'),
    ('rejthreshupper', '0'),
    ('rejthreshlower', '0'),
    ('rawGate', '0'),
    ('rawGateChan', '0'),
    ('rawGatePol', '0'),
    ('defaultTime', '0'),
    ('defaultMode', '0'),
    ('trial_comment', '0'),
    ('digout_state', '0'),
    ('stim_phase', '0'),
    ('stim_period', '0'),
    ('bp1lowcut', '0'),
    ('bp1highcut', '0'),
    ('thresh_lookback', '0'),
    ('palette', '0'),
    ('checkUpdates', '0'),
    ('Spike2msMode', '0'),
    ('DIOTimeBase', '0'),
    ('pretrigSamps', '0'),
    ('spikeLockout', '0'),
    ('BPFspikelen', '0'),
    ('BPFspikeLockout', '0'),
    ('tracked_spots', '0'),
    ('colmap_algorithm', '0'),
    ('cluster_delta', '0'),
    ('tracker_pixels_per_metre', '0'),
    ('two_cameras', '0'),
    ('xcoordsrc', '0'),
    ('ycoordsrc', '0'),
    ('zcoordsrc', '0'),
    ('twocammode', '0'),
    ('stim_pwidth', '0'),
    ('stim_pamp', '0'),
    ('stim_pperiod', '0'),
    ('stim_prepeat', '0'),
    ('stim_tnumber', '0'),
    ('stim_tperiod', '0'),
    ('stim_trepeat', '0'),
    ('stim_bnumber', '0'),
    ('stim_bperiod', '0'),
    ('stim_brepeat', '0'),
    ('stim_gnumber', '0'),
    ('single_pulse_width', '0'),
    ('single_pulse_amp', '0'),
    ('stim_patternmask_1', '0'),
    ('stim_patterntimes_1', '0'),
    ('stim_patternnames_1', '0'),
    ('stim_patternmask_2', '0'),
    ('stim_patterntimes_2', '0'),
    ('stim_patternnames_2', '0'),
    ('stim_patternmask_3', '0'),
    ('stim_patterntimes_3', '0'),
    ('stim_patternnames_3', '0'),
    ('stim_patternmask_4', '0'),
    ('stim_patterntimes_4', '0'),
    ('stim_patternnames_4', '0'),
    ('stim_patternmask_5', '0'),
    ('stim_patterntimes_5', '0'),
    ('stim_patternnames_5', '0'),
    ('scopestimtrig', '0'),
    ('stim_start_delay', '0'),
    ('biphasic', '0'),
    ('use_dacstim', '0'),
    ('stimscript', '0'),
    ('stimfile', '0'),
    ('numPatterns', '0'),
    ('stim_patt_1', '0'),
    ('stim_patt_2', '0'),
    ('numProtocols', '0'),
    ('stim_prot_1', '0'),
    ('stim_prot_2', '0'),
    ('stim_during_rec', '0'),
    ('info_subject', '0'),
    ('info_trial', '0'),
    ('waveform_period', '0'),
    ('pretrig_period', '0'),
    ('deadzone_period', '0'),
    ('fieldtrig', '0'),
    ('sa_manauto', '0'),
    ('sl_levlat', '0'),
    ('sp_manauto', '0'),
    ('sa_time', '0'),
    ('sl_levstart', '0'),
    ('sl_levend', '0'),
    ('sl_latstart', '0'),
    ('sl_latend', '0'),
    ('sp_startt', '0'),
    ('sp_endt', '0'),
    ('resp_endt', '0'),
    ('recordcol', '0'),
    ('extin_port', '0'),
    ('extin_bit', '0'),
    ('extin_edge', '0'),
    ('trigholdwait', '0'),
    ('overlap', '0'),
    ('xmin', '0'),
    ('xmax', '0'),
    ('ymin', '0'),
    ('ymax', '0'),
    ('brightness', '0'),
    ('contrast', '0'),
    ('saturation', '0'),
    ('hue', '0'),
    ('gamma', '0'),
    ('nullEEG', '0')
]


def make_set_entries():
    _entries_to_number = dict([
        (e[0] + str(n), e[1]) for n in range(64) for e in entries_to_number])

    _entries_to_number_one_indexed = dict([
        (e[0] + str(n), e[1])
        for e in entries_to_number_one_indexed for n in range(1, 65)])

    _entries_to_number_to_sixteen = dict([
        (e[0] + str(n), e[1])
        for e in entries_to_number_to_sixteen for n in range(1, 17)])

    _entries_to_number_to_three = dict([
        (e[0] + str(n), e[1])
        for e in entries_to_number_to_three for n in range(1, 4)])

    _entries_to_replace_one = dict([
        (e[0].replace('1', str(n)), e[1])
        for e in entries_to_replace_one for n in range(1, 5)])

    _entries_to_number_to_nine = dict([
        (e[0] + str(n), e[1])
        for e in entries_to_number_to_nine for n in range(0, 10)])

    _entries_groups = dict([
        (e[0].replace('X', str(n)).replace('Y', str(m)), e[1])
        for e in entries_groups for n in range(1, 18)
        for m in range(10)])

    _singleton_entries = dict(singleton_entries)

    return {
        **_entries_to_number, **_singleton_entries,
        **_entries_to_number_one_indexed,
        **_entries_to_number_to_nine, **_entries_to_number_to_sixteen,
        **_entries_to_number_to_three, **_entries_to_replace_one,
        **_entries_groups}


set_meta_info = [
    ("sw_version", None),
    ("ADC_fullscale_mv", None),
    ("tracker_version", None),
    ("stim_version", None),
    ("audio_version", None),
]


def make_set_meta():
    return dict(set_meta_info)


@dataclass
class SetHeader(AxonaHeader):
    meta_info: dict = field(default_factory=make_set_meta)
    set_entries: dict = field(default_factory=make_set_entries)
