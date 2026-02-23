# Quickstart guide

From an ipython terminal, you can load some Axona data as follows:

```python title="Load data"
from ephysiopy.io.recording import AxonaTrial
from pathlib import Path

data = Path("/path/to/data/M851_140908t2rh.set")

trial = AxonaTrial(data)
```

!!! note

    The path should point to the .set file for the recording trial files


Similarly, to load data recorded using OpenEphys:

```python title="Load data"
from ephysiopy.io.recording import OpenEphysBase
from pathlib import Path

data = Path("/path/to/data/RHA1-00064_2023-07-07_10-47-31")

trial = OpenEphysBase(data)
```

!!! note

    The path should point to the top level folder containing the OpenEphys data

In both cases, the trial object is the main interface for working with data.

Plot the rate map for cluster 1 on channel 2:

```python title="Plot data"
import matplotlib.pylab as plt

trial.plot_rate_map(1, 2)
plt.show()
```

![rate map](rate_map.png){width='400px'}
