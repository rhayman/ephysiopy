[![Python package](https://github.com/rhayman/ephysiopy/actions/workflows/python-package.yml/badge.svg)](https://github.com/rhayman/ephysiopy/actions/workflows/python-package.yml)

Synopsis
========

Tools for the analysis of electrophysiological data collected with the Axona or openephys recording systems.

Installation
============

ephysiopy requires python3.7 or greater. The easiest way to install is using pip:

``python3 -m pip install ephysiopy``

or,

``pip3 install ephysiopy``

Or similar.

Code Example
============

Neuropixels / openephys tetrode recordings
------------------------------------------

For openephys-type analysis there are two main entry classes depending on whether you are doing
OpenEphys- or Axona-based analysis. Both classes inherit from the same abstract base
class (TrialInterface) and so share a high degree of overlap in what they can do. Because
of the inheritance structure, the methods you call on each concrete class are the same

```python
from ephysiopy.io.recording import OpenEphysBase
trial = OpenEphysBase("/path/to/top_level")
```

The "/path/to/top_level" bit here means that if your directory hierarchy looks like this:

::

    ├── 2020-03-20_12-40-15
    ├── Record Node 101
    |    └── settings.xml
             experiment1
    |        └── recording1
    |            ├── structure.oebin
    |            ├── sync_messages.txt
    |            ├── continuous
    |            |   └── Neuropix-PXI-107.0
    |            |       └── continuous.dat
    |            └── events
    ├── Record Node 102


Then the "/path/to/top_level" is the folder "2020-03-20_12-40-15"

On insantiation of an OpenEphysBase object the directory structure containing the recording
is traversed and various file locations are noted for later processing of the data in them.

The pos data is loaded by calling the load_pos_data() method:

```python
npx.load_pos_data(ppm=300, jumpmax=100, cm=True)
```

Note
ppm = pixels per metre, used to convert pixel coords to cms.
jumpmax = maximum "jump" in cms for point to be considered "bad" and smoothed over

The same principles apply to the other classes that inherit from TrialInterface (AxonaTrial and OpenEphysNWB)


Plotting data
=============

A mixin class called FigureMaker allows consistent plots, regardless of recording technique. All plotting functions
there begin with "make" e.g "makeRateMap" and return an instance of a matplotlib axis