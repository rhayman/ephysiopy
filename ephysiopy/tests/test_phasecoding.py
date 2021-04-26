from ephysiopy.dacq2py.dacq2py_util import AxonaTrial
from ephysiopy.common.phasecoding import phasePrecession2D
from ephysiopy.common.phasecoding import phase_precession_config as ppc


def test_phase_precession_2d_setup(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    spike_ts = T.TETRODE.get_spike_ts(3, 3)
    phasePrecession2D(T.EEG.sig, 250., T.xy, spike_ts, ppc)


def test_perform_regression(path_to_axona_data):
    T = AxonaTrial(path_to_axona_data)
    T.load()
    spike_ts = T.TETRODE.get_spike_ts(3, 3)
    pp2d = phasePrecession2D(T.EEG.sig, 250., T.xy, spike_ts, ppc)
    pp2d.performRegression()
