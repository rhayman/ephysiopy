[![Python package](https://github.com/rhayman/ephysiopy/actions/workflows/python-package.yml/badge.svg)](https://github.com/rhayman/ephysiopy/actions/workflows/python-package.yml)

Synopsis
========

Tools for the analysis of electrophysiological data collected with the Axona or openephys recording systems.

Installation
============

ephysiopy requires python3.6 or greater. The easiest way to install is using pip:

``python3 -m pip install ephysiopy``

or,

``pip3 install ephysiopy``

Or similar.

I haven't yet tried this in a conda like environment but a quick google shows it should be pretty easy.

Code Example
============

Neuropixels / openephys tetrode recordings
------------------------------------------

For openephys-type analysis there are two main entry classes depending on whether you are doing
Neuropixels or tetrode-based analysis. Both classes inherit from the same
parent class (OpenEphysBase) and so share a high degree of overlap in what they can do.

```python
from ephysiopy.openephys2py.OEKiloPhy import OpenEphysNPX
npx = OpenEphysNPX("/path/to/top_level")
```

The "/path/to/top_level" bit here means that if your directory hierarchy looks like this:

::

    ├── settings.xml
    ├── 2020-03-20_12-40-15
    |    └── experiment1
    |        └── recording1
    |            ├── structure.oebin
    |            ├── sync_messages.txt
    |            ├── continuous
    |            |   └── Neuropix-PXI-107.0
    |            |       └── continuous.dat
    |            └── events


Then OpenEphysNPX should be instantiated as follows:

```python
npx = OpenEphysNPX("2020-03-20_12-40-15")
```

When you load the data the directory structure is iterated over to find files such as sync_messages.txt and settings.xml and so on. The data is loaded by calling the load method:

```python
npx.load()
```

The same principles apply to the OpenEphysNWB class - as the name suggests this is for use with data recorded in the `.nwb format <https://www.nwb.org/>`_

When data is recorded using the `PosTracker <https://github.com/rhayman/PosTracker>`_ plugin for openephys, the position data recorded using that plugin is also loaded, which means plots of position can be produced:

```python
npx.plotPos()
```

For ratemaps and so on you call plotMaps() which can take several arguments to make maps of different types. See the documentation for the base class for details (OpenEphysBase)

Axona tetrode recordings
------------------------

The main entry class for Axona related analysis is "Trial" contained in ephysiopy.dacq2py.dacq2py_util i.e.

```python
from ephysiopy.dacq2py.dacq2py_util import Trial
T = Trial("/path/to/dataset/mytrial")
```

The "usual" Axona dataset includes the following files:

* mytrial.set
* mytrial.1
* mytrial.2
* mytrial.3
* mytrial.4
* mytrial.pos
* mytrial.eeg

Note that you shouldn't specify a suffix when constructing the filename in the code example above.

You can now start analysing your data! i.e.

```python
T.plotEEGPower()
T.plotMap(tetrode=1, cluster=4)
```

Motivation
==========

Analysis using Axona's Tint cluster cutting program or phy/ phy2 (openephys) is great but limited. This extends that functionality.

Optional packages include:

* `klustakwik <https://github.com/klusta-team/klustakwik>`_

Download the files and extract to a folder and make sure it's on your Python path
NB this is limited to data recorded using Axona as it has now been superceded by tools such as KiloSort/ KiloSort2 etc.

Contributors
============

Robin Hayman.
