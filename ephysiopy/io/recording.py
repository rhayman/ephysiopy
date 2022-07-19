import abc
import os
from enum import Enum
from pathlib import Path, PurePath
from typing import NoReturn

from ephysiopy.common.ephys_generic import PosCalcsGeneric
from ephysiopy.dacq2py.axonaIO import IO, Pos


def fileContainsString(pname: str, searchStr: str) -> bool:
    if os.path.exists(pname):
        with open(pname, "r") as f:
            strs = f.read()
        lines = strs.split("\n")
        found = False
        for line in lines:
            if searchStr in line:
                found = True
        return found
    else:
        return False


class RecordingKind(Enum):
    FPGA = 1
    NEUROPIXELS = 2


class TrialInterface(metaclass=abc.ABCMeta):
    def __init__(self, pname: Path, **kwargs) -> None:
        assert pname.exists()
        setattr(self, "pname", pname)
        setattr(self, "settings", None)
        setattr(self, "xy", None)
        setattr(self, "xyTS", None)
        setattr(self, "ppm", None)  # pixels per metre
        setattr(self, "clusterData", None)  # Kilosort or .cut / .clu file
        setattr(self, "recording_start_time", None)
        setattr(self, "ttl_data", None)
        setattr(self, "accelerometer_data", None)

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "load_neural_data")
            and callable(subclass.load_neural_data)
            and hasattr(subclass, "load_pos")
            and callable(subclass.load_pos)
            or NotImplemented
        )

    @abc.abstractmethod
    def load_neural_data(self, pname: Path) -> NoReturn:
        """Load the neural data"""
        raise NotImplementedError

    @abc.abstractmethod
    def load_pos_data(self, pname: Path) -> NoReturn:
        """Load the position data"""
        raise NotImplementedError


class AxonaTrial(TrialInterface):
    def __init__(self, pname: Path, **kwargs) -> None:
        super().__init__(pname, **kwargs)
        self.__settings = None

    @property
    def settings(self) -> None:
        if self.__settings is None:
            try:
                settings_io = IO()
                self.__settings = settings_io.getHeader(self.pname)
            except IOError:
                print(".set file not loaded")
                self.__settings = None
        return self.__settings

    @settings.setter
    def settings(self, value) -> None:
        self.__settings = value

    def load_neural_data(self, pname: Path) -> None:
        pass

    def load_pos_data(self, pname: Path) -> None:
        if getattr(self, "xy") is None:
            try:
                AxonaPos = Pos(self.pname)
                P = PosCalcsGeneric(
                    AxonaPos.led_pos[0, :],
                    AxonaPos.led_pos[1, :],
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


class OpenEphysBase(TrialInterface):
    def __init__(self, pname: Path, **kwargs) -> None:
        super().__init__(pname, **kwargs)
        setattr(self, "sync_messsage_file", None)

    def load_neural_data(self, pname: Path) -> None:
        pass

    def load_pos_data(self, pname: Path) -> None:
        pass

    def find_files(
        self,
        pname_root: str,
        experiment_name: str = "experiment1",
        recording_name: str = "recording1",
        recording_kind: RecordingKind = RecordingKind.NEUROPIXELS,
    ):
        exp_name = Path(experiment_name)
        PosTracker_match = (
            exp_name / recording_name / "events" / "*Pos_Tracker*/BINARY_group*"
        )
        TrackingPlugin_match = (
            exp_name / recording_name / "events" / "*Tracking_Port*/BINARY_group*"
        )
        sync_file_match = exp_name / recording_name
        rec_kind = ""
        if recording_kind == RecordingKind.NEUROPIXELS:
            rec_kind = "Neuropix-PXI-[0-9][0-9][0-9]."
        elif recording_kind == RecordingKind.FPGA:
            rec_kind = "Rhythm_FPGA-[0-9][0-9][0-9]."
        APdata_match = exp_name / recording_name / "continuous" / (rec_kind + "0")
        LFPdata_match = exp_name / recording_name / "continuous" / (rec_kind + "1")

        if pname_root is None:
            pname_root = self.pname_root

        for d, c, f in os.walk(pname_root):
            for ff in f:
                if "." not in c:  # ignore hidden directories
                    if "data_array.npy" in ff:
                        if PurePath(d).match(str(PosTracker_match)):
                            if self.path2PosData is None:
                                self.path2PosData = os.path.join(d)
                                setattr(self, "pos_data_type", "PosTracker")
                                print(f"Found pos data at: {self.path2PosData}")
                            self.path2PosOEBin = Path(d).parents[1]
                        if PurePath(d).match("*pos_data*"):
                            if self.path2PosData is None:
                                self.path2PosData = os.path.join(d)
                                print(f"Found pos data at: {self.path2PosData}")
                        if PurePath(d).match(str(TrackingPlugin_match)):
                            if self.path2PosData is None:
                                self.path2PosData = os.path.join(d)
                                setattr(self, "pos_data_type", "TrackingPlugin")
                                print(f"Found pos data at: {self.path2PosData}")
                    if "continuous.dat" in ff:
                        if PurePath(d).match(str(APdata_match)):
                            self.path2APdata = os.path.join(d)
                            print(f"Found continuous data at: {self.path2APdata}")
                            self.path2APOEBin = Path(d).parents[1]
                        if PurePath(d).match(str(LFPdata_match)):
                            self.path2LFPdata = os.path.join(d)
                            print(f"Found continuous data at: {self.path2LFPdata}")
                    if "sync_messages.txt" in ff:
                        if PurePath(d).match(str(sync_file_match)):
                            sync_file = os.path.join(d, "sync_messages.txt")
                            if fileContainsString(sync_file, "Processor"):
                                self.sync_message_file = sync_file
                                print(f"Found sync_messages file at: {sync_file}")


class OpenEphysNPX(OpenEphysBase):
    def __init__(self, pname: Path, **kwargs) -> None:
        super().__init__(pname, **kwargs)

    def load_neural_data(self, pname: Path) -> None:
        pass

    def load_pos_data(self, pname: Path) -> None:
        super().load_pos_data(pname)


class OpenEphysBinary(OpenEphysBase):
    def __init__(self, pname: Path, **kwargs) -> None:
        super().__init__(pname, **kwargs)

    def load_neural_data(self, pname: Path) -> None:
        pass

    def load_pos_data(self, pname: Path) -> None:
        super().load_pos_data(pname)


class OpenEphysNWB(OpenEphysBase):
    def __init__(self, pname: Path, **kwargs) -> None:
        super().__init__(pname, **kwargs)

    def load_neural_data(self, pname: Path) -> None:
        pass

    def load_pos_data(self, pname: Path) -> None:
        super().load_pos_data(pname)
