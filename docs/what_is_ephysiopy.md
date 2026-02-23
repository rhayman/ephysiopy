# What is ephysiopy?

ephysiopy is a Python package for the analysis of electrophysiological data collected with the Axona or openephys recording systems. It provides a unifying set of tools for the analysis of electrophysiological data recorded using tetrode or Neuropixels (LINK).

It can load the data and so some plotting and analysis too.

What it **isn't** is a way to pre-process data (drift correction etc) or to do spike sorting.

There are some excellent tools for that already, and ephysiopy is designed to work with the output of those tools, not to replace them.

There are many spike sorting tools available, [KiloSort](https://github.com/MouseLand/Kilosort) is popular among Neuropixels users, but there are many others (LINKS)

Although there are some methods in ephysiopy for visualising spike waveforms, auto- and cross-correlograms and so on
it is not supposed to be a replacement for tools like [phy](https://github.com/cortex-lab/phy) or [SpikeInterface](https://github.com/spikeinterface)

Iniitially, it was a collection of functions and scripts for loading and analysing data from the Axona system,
but over time grew to encompass data from the OpenEphys system. That led to the development of an interface
class (TrialInterface) to provide a common set of methods for loading, plotting and analysing the data regardless
of the recording system used. It should therefore be possible to extend that interface for other recording systems in the future, if there is demand for that.

