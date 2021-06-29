import numpy as np
from ephysiopy.dacq2py.dacq2py_util import AxonaTrial
from ephysiopy.common.phasecoding import phasePrecession2D
from ephysiopy.common.phasecoding import phase_precession_config as ppc


def test_phase_precession_2d_setup(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    spike_ts = T.TETRODE.get_spike_samples(1, 1)
    phasePrecession2D(T.EEG.sig, 250., T.xy, spike_ts, T.xyTS, ppc)


def test_perform_regression(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    spike_ts = T.TETRODE.get_spike_samples(1, 1)
    pp2d = phasePrecession2D(T.EEG.sig, 250., T.xy, spike_ts, T.xyTS, ppc)
    pp2d.performRegression()


def test_pos_props_with_events(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    spike_ts = T.TETRODE.get_spike_samples(1, 1)
    pp2d = phasePrecession2D(T.EEG.sig, 250., T.xy, spike_ts, T.xyTS, ppc)
    laser_events = np.array(
        np.ceil(T.STM['on'] / T.STM.timebase * T.pos_sample_rate)).astype(int)
    peaksXY, _, labels, _ = pp2d.partitionFields()
    pp2d.getPosProps(
        labels, peaksXY, laserEvents=laser_events, plot=True)


def test_circ_circ_corr(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    spike_ts = T.TETRODE.get_spike_samples(1, 1)
    pp2d = phasePrecession2D(T.EEG.sig, 250., T.xy, spike_ts, T.xyTS, ppc)
    pp2d._circCircCorrTLinear(
        theta=T.dir[0:10].data, phi=np.random.vonmises(0, 4, 10), k=10, conf=True)

    pp2d._circCircCorrTLinear(
        theta=T.dir[0:3].data, phi=np.random.vonmises(0, 4, 3), k=3, conf=True)