import numpy as np
from pycircstat2 import Circular
from pycircstat2.descriptive import circ_mean, circ_std, circ_dispersion, circ_kurtosis
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

        """
        self.head_directions = Circular(
            head_directions, unit="degrees", kwargs_median={"method": None}
        )

    # Descriptive statistics:
    def mean(self):
        return circ_mean(self.head_directions.alpha)

    def variability(self):
        return circ_std(self.head_directions.alpha)

    def consistency(self):
        return 1 - (self.variability() / np.pi)

    def dispersion(self):
        return circ_dispersion(self.head_directions.alpha)

    def entropy(self):
        hist, _ = np.histogram(
            self.head_directions.data, bins=36, range=(0, 360), density=True
        )
        hist = hist[hist > 0]  # Remove zero entries to avoid log(0)
        return -np.sum(hist * np.log(hist))

    def kurtosis(self):
        return circ_kurtosis(self.head_directions.alpha)

    # Hypothesis testing:
    def rayleigh_test(self):
        return rayleigh_test(self.head_directions.alpha)

    def omnibus_test(self):
        # Placeholder for Omnibus test implementation
        return omnibus_test(self.head_directions.alpha)
