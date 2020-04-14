Usage examples
==============

There should be shared functionality between both the openephys and Axona interfaces as ideally they both use the code in the ephysiopy.common sub-package (I don't think this is 100% true for the Axona interface).

openephys
---------

There are two recording formats that the package deals with, data recorded in the nwb format or data recorded in a binary format.
For more details of the nwb format see `https://www.nwb.org/ <https://www.nwb.org/>`_ and for more details on the binary format take a look `here <https://open-ephys.atlassian.net/wiki/spaces/OEW/pages/166789121/Flat+binary+format>`_. In my hands I've recorded in the nwb format when doing tetrode recording and the flat binary format when doing neuropixels recordings. Therefore there are two classes that load the two different data formats, OpenEphysNWB and OpenEphysNPX for the nwb / tetrode-based data and the neuropixels format data respectively.

To load some neuropixels data:

>>> from ephysiopy.openephys2py.OEKiloPhy import OpenEphysNPX
>>> npx = OpenEphysNPX("/path/to/top_level")

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

>>> npx = OpenEphysNPX("2020-03-20_12-40-15")

When you load the data the directory structure is iterated over to find files such as sync_messages.txt and settings.xml and so on. The data is loaded by calling the load() method:

>>> npx.load()

The format of the directory hierarchy is a bit flatter when recording in the nwb format. It might look something like:

::

    ├── 2020-03-20_12-40-15
    |    └── experiment1.nwb
    |    └── settings.xml

But the same principles apply to the OpenEphysNWB class, i.e.

>>> from ephysiopy.openephys2py.OEKiloPhy import OpenEphysNWB
>>> nwb = OpenEphysNWB("2020-03-20_12-40-15")
>>> nwb.load()

In both cases you can now call various methods to do things like plot ratemaps and so on (assuming the data has been clustered/ spike sorted)

>>> npx.plotMaps(clusters=[1, 24, 87, 129])

The above command will plot out the ratemaps for clusters 1, 24, 87 and 129 in a single figure window. Alternatively,

>>> npx.plotMapsOneAtATime(clusters=[1, 24, 87, 129])

will plot them in separate windows. Check the documentation for both methods to see what else you can do i.e :py:meth:`~ephysiopy.openephys2py.OEKiloPhy.OpenEphysBase.plotMaps`, and :py:meth:`~ephysiopy.openephys2py.OEKiloPhy.OpenEphysBase.plotMapsOneAtATime`.

Probably a good place to get started is to look at the various classes defined in the common package, :py:mod:`ephysiopy.common`. Of 
particular interest are the classes in the sub-module :py:mod:`ephysiopy.common.ephys_generic` . The two main classes for doing spatial 
analysis are :py:class:`~ephysiopy.common.ephys_generic.MapCalcsGeneric` and :py:class:`~ephysiopy.common.ephys_generic.PosCalcsGeneric`.

Axona
-----

The package started life as a way of doing more with Axona data and extended some old code I had that did some of the same things but in Matlab. The main entry class for Axona related analysis is "Trial" contained in ephysiopy.dacq2py.dacq2py_util i.e.

>>> from ephysiopy.dacq2py.dacq2py_util import Trial
>>> T = Trial("/path/to/dataset/mytrial")

The "usual" Axona dataset includes the following files:

* mytrial.set
* mytrial.1
* mytrial.2
* mytrial.3
* mytrial.4
* mytrial.pos
* mytrial.eeg

Note that you don't specify a suffix when constructing the filename in the code example above.

You can now start analysing your data, for example

>>> T.plotEEGPower()
>>> T.plotMap(tetrode=1, cluster=4)
