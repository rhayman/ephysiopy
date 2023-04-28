# import numpy as np
# from ephysiopy.io.recording import AxonaTrial
# from ephysiopy.common.phasecoding import phasePrecession2D
# from ephysiopy.common.phasecoding import phase_precession_config as ppc


# def test_phase_precession_2d_setup(path_to_axona_data):
#     T = AxonaTrial(path_to_axona_data)
#     T.load_pos_data(path_to_axona_data)
#     T.load_lfp(path_to_axona_data, "EEG")
#     spike_ts = T.TETRODE.get_spike_samples(1, 1)
#     pp2d = phasePrecession2D(
#         T.EEGCalcs.sig,
#         250.,
#         T.PosCalcs.xy,
#         spike_ts,
#         T.PosCalcs.xyTS,
#         ppc)
#     # test setting some properties
#     pp2d.ts = range(10)


# def test_perform_regression(path_to_axona_data):
#     T = AxonaTrial(path_to_axona_data)
#     T.load_pos_data(path_to_axona_data)
#     T.load_lfp(path_to_axona_data, "EEG")
#     spike_ts = T.TETRODE.get_spike_samples(1, 1)
#     pp2d = phasePrecession2D(T.EEGCalcs.sig,
#                              250.,
#                              T.PosCalcs.xy,
#                              spike_ts,
#                              T.PosCalcs.xyTS,
#                              ppc)
#     pp2d.performRegression()
#     # do the regression with different options
#     peaksXY, _, labels, _ = pp2d.partitionFields()
#     posD, runD = pp2d.getPosProps(labels, peaksXY)
#     pp2d.getThetaProps()
#     spkD = pp2d.getSpikeProps(posD["runLabel"],
#                               runD["meanDir"],
#                               runD["runDurationInPosBins"])
#     pp2d._ppRegress(spkD, whichSpk="last")
#     # the following test fails..
#     # pp2d._ppRegress(spkD, whichSpk="mean")


# def test_pos_props_with_events(path_to_axona_data):
#     T = AxonaTrial(path_to_axona_data)
#     T.load_pos_data(path_to_axona_data)
#     T.load_lfp(path_to_axona_data, "EEG")
#     T.load_ttl()
#     spike_ts = T.TETRODE.get_spike_samples(1, 1)
#     pp2d = phasePrecession2D(T.EEGCalcs.sig,
#                              250.,
#                              T.PosCalcs.xy,
#                              spike_ts,
#                              T.PosCalcs.xyTS,
#                              ppc)
#     laser_events = np.array(
#         np.ceil(T.ttl_data['on'] /
#                 T.ttl_data.timebase *
#                 T.PosCalcs.sample_rate)).astype(int)
#     peaksXY, _, labels, _ = pp2d.partitionFields()
#     pp2d.getPosProps(
#         labels, peaksXY, laserEvents=laser_events, plot=True)


# def test_circ_circ_corr(path_to_axona_data):
#     T = AxonaTrial(path_to_axona_data)
#     T.load_pos_data(path_to_axona_data)
#     T.load_lfp(path_to_axona_data)
#     spike_ts = T.TETRODE.get_spike_samples(1, 1)
#     pp2d = phasePrecession2D(T.EEGCalcs.sig,
#                              250.,
#                              T.PosCalcs.xy,
#                              spike_ts,
#                              T.PosCalcs.xyTS,
#                              ppc)
#     pp2d._circCircCorrTLinear(
#         theta=T.PosCalcs.dir[0:10].data,
#         phi=np.random.vonmises(0, 4, 10),
#         k=10,
#         conf=True)

#     pp2d._circCircCorrTLinear(
#         theta=T.PosCalcs.dir[0:3].data,
#         phi=np.random.vonmises(0, 4, 3),
#         k=3,
#         conf=True)
