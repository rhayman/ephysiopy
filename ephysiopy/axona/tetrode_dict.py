from ephysiopy.axona import axonaIO
import numpy as np


class TetrodeDict(dict):
    """
    A dictionary-like object that returns a Tetrode object when
    a key is requested. The Tetrode object is created on the fly
    if it does not already exist. The Tetrode object is created
    using the axonaIO.Tetrode class. The Tetrode object is stored
    in the dictionary for future use.
    """

    def __init__(self, filename_root, *args, **kwargs):
        self.filename_root = filename_root
        self.valid_keys = range(1, 33)
        self.update(*args, **kwargs)
        self.use_volts = kwargs.get("volts", True)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def __getitem__(self, key):
        if isinstance(key, int):
            try:
                val = dict.__getitem__(self, key)
                return val
            except KeyError:
                if key in self.valid_keys:
                    try:
                        val = axonaIO.Tetrode(
                            self.filename_root, key, volts=self.use_volts
                        )
                        self[key] = val
                        return val
                    except Exception:
                        raise KeyError(f"Tetrode {key} not available")
                else:
                    raise KeyError(f"Tetrode {key} not available")

    def get_spike_samples(self, tetrode: int, cluster: int) -> np.ndarray:
        """
        Returns spike times in seconds for given cluster from given
        tetrode

        Parameters
        ----------
        tetrode : int
            The tetrode number
        cluster : int
            The cluster number

        Returns
        -------
        np.ndarray
            The spike times in seconds

        """
        try:
            this_tet = self[tetrode]
            return this_tet.getClustTS(cluster) / this_tet.timebase
        except Exception:
            raise Exception(f"Could not get timestamps for cluster: {cluster}")
