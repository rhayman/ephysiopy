# OpenEphys settings

All of the plugins that sit in the OpenEphys signal chain are parsed out from
the settings.xml file and take a similar format. They are available as the 'processors'
OrderedDict of the Settings class.

```python
trial.settings.processors.keys()
odict_keys(['Acquisition Board 100', 'Bandpass Filter 107', 'LFP Viewer 102', 'TrackMe 103', 'Tracking Visual 105', 'StimControl 109', 'StimControl 110'])
```

The settings for each RecordNode is similarly avaialble as
an OrderedDict in the 'record_nodes' attribute:

```python
trial.settings.record_nodes.keys()
odict_keys(['Record Node 101', 'Record Node 104'])
```

## Record nodes

The settings for the Record Nodes are saved as a dataclass:

::: ephysiopy.openephys2py.OESettings.RecordNode

The settings for OpenEphys data are saved to a settings.xml file. This is loaded automatically
when an OpenEphys trial object is created. Each OpenEphys plugin derives from the OEPlugin dataclass:

## OpenEphys plugins

::: ephysiopy.openephys2py.OESettings.OEPlugin

It should be straightforawrd to create new classes that derive from OEPlugin when new plugins become
available.

::: ephysiopy.openephys2py.OESettings.RhythmFPGA

::: ephysiopy.openephys2py.OESettings.NeuropixPXI

::: ephysiopy.openephys2py.OESettings.AcquisitionBoard

::: ephysiopy.openephys2py.OESettings.BandpassFilter

::: ephysiopy.openephys2py.OESettings.TrackingPort

::: ephysiopy.openephys2py.OESettings.PosTracker

::: ephysiopy.openephys2py.OESettings.TrackMe

::: ephysiopy.openephys2py.OESettings.StimControl

::: ephysiopy.openephys2py.OESettings.RippleDetector

There are some helper classes for doing string to int and float conversions:

::: ephysiopy.openephys2py.OESettings.IntConversion

::: ephysiopy.openephys2py.OESettings.FloatConversion


