# Welcome to ephysiopy

## Installation

```
pip install ephysiopy
```

## Basic usage

python```
from ephysiopy.io.recording import OpenEphysBase
from pathlib import Path
trial = OpenEphysBase(Path("/path/to/top/folder"))
trial.load_pos_data()
trial.load_neural_data()
trial.rate_map(1, 1)
```

This will load the data recorded with openephys contained
in the folder "/path/to/top/folder", load the position data
and the neural data (KiloSort output) and plot the cluster 1
on channel 1