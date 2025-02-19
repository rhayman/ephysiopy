from pathlib import Path
import matplotlib.pylab as plt
from ephysiopy.common.phasecoding import (
    phase_precession_config,
    phasePrecession2D,
)
from ephysiopy.io.recording import AxonaTrial

T = AxonaTrial(
    Path(
        "/home/robin/Documents/Science/SST_data_and_paper/SST_data/raw/M851/M851_140908t2rh.set"
    )
)
T.load_pos_data()
T.load_lfp()

pp_config = phase_precession_config
pp_config["minimum_allowed_run_speed"] = 0.1
pp_config["minimum_allowed_run_duration"] = 1
pp_config["convert_xy_2_cm"] = True
pp_config["ppm"] = 445
pp_config["field_smoothing_kernel_len"] = 31
pp_config["field_smoothing_kernel_sigma"] = 13
pp_config["field_threshold"] = 0.5
pp_config["field_threshold_percent"] = 20

P = phasePrecession2D(
    T.EEGCalcs.sig,
    T.EEGCalcs.fs,
    T.PosCalcs.orig_xy,
    T.get_spike_times(3, 5),
    T.PosCalcs.xyTS,
    pp_config,
)

peaksXY, _, labels, _ = P.partitionFields(plot=True)

posD, runD = P.getPosProps(labels, peaksXY, laserEvents=None, plot=False)

P.getThetaProps()

spkD = P.getSpikeProps(posD["runLabel"], runD["meanDir"], runD["runDurationInPosBins"])

plt.show()
