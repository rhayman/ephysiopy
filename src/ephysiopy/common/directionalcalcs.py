import numpy as np
from pycircstat2 import Circular, circ_plot
from pycircstat2.descriptive import (
    circ_mean,
    circ_std,
    circ_dispersion,
    circ_kurtosis,
)
from pycircstat2.hypothesis import rayleigh_test, omnibus_test


class HeadDirectionCalcs:
    def __init__(self, head_directions: np.ma.MaskedArray):
        """
        Initialize the HeadDirectionCalcs class with head direction data.

        Parameters
        ----------
        head_directions (np.ma.MaskedArray): A masked array containing head
        direction data in degrees (0-360).
        Masked values represent invalid or missing data.

        Notes
        -----
        An instance of this class can be created from a Trial object
        with cluster and channel information like so:

        >>> from ephysiopy.common.utils import repeat_ind
        >>> trial = Trial(...)
        >>> trial.load_pos_data()
        >>> trial.load_neural_data()
        >>> cluster = 1
        >>> channel = 1
        >>> spk_weights = trial.get_spike_weights(cluster, channel)
        >>> idx = np.take(trial.PosCalcs.dir, repeat_ind(spk_weights))
        >>> hd = HeadDirectionCalcs(idx)
        """
        self.head_directions = Circular(
            head_directions, unit="degree", kwargs_median={"method": None}
        )

    # Descriptive statistics:
    def mean(self):
        return np.rad2deg(circ_mean(self.head_directions.alpha))

    def variability(self):
        return np.rad2deg(circ_std(self.head_directions.alpha))

    def consistency(self):
        return 1 - (self.variability() / np.pi)

    def dispersion(self):
        return np.rad2deg(circ_dispersion(self.head_directions.alpha))

    def entropy(self):
        hist, _ = np.histogram(
            self.head_directions.data, bins=36, range=(0, 360), density=True
        )
        hist = hist[hist > 0]  # Remove zero entries to avoid log(0)
        return -np.sum(hist * np.log(hist))

    def kurtosis(self):
        return np.rad2deg(circ_kurtosis(self.head_directions.alpha))

    # Hypothesis testing:
    def rayleigh_test(self):
        return rayleigh_test(self.head_directions.alpha)

    def omnibus_test(self):
        return omnibus_test(self.head_directions.alpha)

    # Plotting
    def plot(self):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, projection="polar")
        circ_plot(self.head_directions, ax=ax, config={
                  "median": False, "mean": False})
        ax.set_xticklabels([])

        return ax
