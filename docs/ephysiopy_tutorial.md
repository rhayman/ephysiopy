# ephysiopy user guide

This guide is an overview and explains basic features; details are found in the [ephysiopy reference](io)

## Getting started

[What is ephysiopy?](what_is_ephysiopy.md)

[Installation](installation)

[ephsyiopy quickstart](quickstart)

## Data organisation

With data recorded using OpenEphys the folder structure looks something like this:

```
  RHA1-00064_2023-07-07_10-47-31
  ├── Record Node 101
  │   ├── experiment1
  │   │   └── recording1
  │   │       ├── continuous
  │   │       │   └── Acquisition_Board-100.Rhythm Data
  │   │       │       ├── amplitudes.npy
  │   │       │       ├── channel_map.npy
  │   │       │       ├── channel_positions.npy
  │   │       │       ├── cluster_Amplitude.tsv
  │   │       │       ├── cluster_ContamPct.tsv
  │   │       │       ├── cluster_group.tsv
  │   │       │       ├── cluster_info.tsv
  │   │       │       ├── cluster_KSLabel.tsv
  │   │       │       ├── continuous.dat
  │   │       │       ├── params.py
  │   │       │       ├── pc_feature_ind.npy
  │   │       │       ├── pc_features.npy
  │   │       │       ├── phy.log
  │   │       │       ├── rez.mat
  │   │       │       ├── similar_templates.npy
  │   │       │       ├── spike_clusters.npy
  │   │       │       ├── spike_templates.npy
  │   │       │       ├── spike_times.npy
  │   │       │       ├── template_feature_ind.npy
  │   │       │       ├── template_features.npy
  │   │       │       ├── templates_ind.npy
  │   │       │       ├── templates.npy
  │   │       │       ├── whitening_mat_inv.npy
  │   │       │       └── whitening_mat.npy
  │   │       ├── events
  │   │       │   ├── Acquisition_Board-100.Rhythm Data
  │   │       │   │   └── TTL
  │   │       │   │       ├── full_words.npy
  │   │       │   │       ├── sample_numbers.npy
  │   │       │   │       ├── states.npy
  │   │       │   │       └── timestamps.npy
  │   │       │   └── MessageCenter
  │   │       │       ├── sample_numbers.npy
  │   │       │       ├── text.npy
  │   │       │       └── timestamps.npy
  │   │       ├── structure.oebin
  │   │       └── sync_messages.txt
  │   └── settings.xml
  └── Record Node 104
      ├── experiment1
      │   └── recording1
      │       ├── continuous
      │       │   └── TrackMe-103.TrackingNode
      │       │       ├── continuous.dat
      │       │       ├── sample_numbers.npy
      │       │       └── timestamps.npy
      │       ├── events
      │       │   ├── MessageCenter
      │       │   │   ├── sample_numbers.npy
      │       │   │   ├── text.npy
      │       │   │   └── timestamps.npy
      │       │   └── TrackMe-103.TrackingNode
      │       │       └── TTL
      │       │           ├── full_words.npy
      │       │           ├── sample_numbers.npy
      │       │           ├── states.npy
      │       │           └── timestamps.npy
      │       ├── structure.oebin
      │       └── sync_messages.txt
      └── settings.xml
```

Despite looking overwhelming this is actually fairly straighforward...

The top level (or parent) folder is called 

```
RHA1-00064_2023-07-07_10-47-31
```

and contains everything else.

In OpenEphys language there are two Record Nodes, "Record Node 101" and "Record Node 104"

Record Node 101 is the one that contains the neural recording data as well as some other stuff.

The main path through to the actual recording data is:

```
RHA1-00064_2023-07-07_10-47-31/Record Node 101/experiment1/recording1/continuous/Acquisition_Board-100.Rhythm Data
```

In the Acquisition_Board-100.Rhythm Data folder are a bunch of files but most of these are the results
of running KiloSort on the acquired data so we'll deal with those later.

The main recording file is continuous.dat which is a binary file usually fairly large in size (many 10's of Gbs).

This raw binary file is something we can filter in different frequency bands to examine different parts
of the activity of the brain. For high frequency spiking activity we can filter the data between about 300-500Hz and 
several kHz. For lower frequency local field potential (LFP) activity we can filter the data between about 1-300Hz.

The files in the rest of the folders for this Record Node are less relevant for now so we'll ignore them.

Record Node 104 contains the tracking data for the animal's position in the arena. The main path to the tracking data is:

```
RHA1-00064_2023-07-07_10-47-31/Record Node 104/experiment1/recording1/continuous/TrackMe-103.TrackingNode
```

The main data for the tracking is again in the continuous.dat file but there is also importamnt information in the files here:

```
RHA1-00064_2023-07-07_10-47-31/Record Node 104/experiment1/recording1/events/TrackMe-103.TrackingNode
```

This folder contains the TTL events for the tracking data which are important for aligning the tracking data with the neural recording data.

## Loading the data

### Getting started

- To load the data using Python start an ipython session in the terminal:

```bash
ipython
```

- Then import the relevant bits of the ephysiopy package and a way to tell ephysiopy where the data is:

```python title="Load data"
from ephysiopy.io.recording import OpenEphysBase
from pathlib import Path

data = Path("/path/to/data/RHA1-00064_2023-07-07_10-47-31")

trial = OpenEphysBase(data)
```

If things are working ok you should see a message saying the settings data has been loaded. This refers to the data
in the settings.xml file in the Acquisition_Board-100.Rhythm Data folder. This file contains important information about the recording such as the sampling rate, number of channels etc.

Try to load the position data as well:

```python
trial.load_pos_data()
```

You should see a message saying th TrackMe data has been loaded:

```
Loading TrackMe data...
Loaded pos data
```

There are lots of potential plugins / ways for the position data to be acquired and I've attempted to make it fariyl
modular so that it should be possible to load data from different sources. If you have a different setup and the data doesn't load then let me know and I can add support for it.

At it's heart, position data for the type of electrophysiology we do is simply a set of x and y coordinates for the position of the animal in the arena at different time points
The TrackMe plugin is one way to acquire this data but there are other ways to do it as well. The important thing is that we have a way to align the position data with the neural recording data.

This is a critical step and if the data is misaligned you won't be able to come to any conclusions about how they brain is encoding space! Fortunately, ephysiopy takes care of this alignment for you!

The next thing to do is load the neural data. This is actually a bit of a misnomer as, in order to be fast, the results of the KiloSort session are loaded rather than the raw data.

If we want to plot waveforms i.e. actually index into the raw data file, then we have to load the raw data. For most purposes all we need are the timestamps that a given cluster emitted spikes.

Similarly if we want to look at the LFP data then we need to load the raw data and filter it in the relevant frequency band(s). You can do that like so:

```python
trial.load_lfp()
```

This will take a while as we need to load the data and filter it. For now, all you need to know is that you can call this load_lfp() methods with different frequency bands. We'll come back to this...

## Plotting data

This is where things get a bit more interesting as we can start to see the results of our hard work with some nice pictures.

The most straighforward thing to plot is the path of the animal and overlay on top of that the locations at which spikes were emitted.

```python
trial.plot_spike_path()
```

This will plot the path of the animal over the course of the recording and can be a good check that we tracked the position of the animal well. Bad tracking will appear as distinct and abrupt jumps 
in the tracked position, most likely looking like straight lines that go to a common source/ sink. This is often the casued by reflections of the LED(s) on the animals head against some surface.

Calling this function with some arguments is more interesting! We need to know valid arguments to call it with though; that is we need to know which channels on our probe (or recording device) had clusters assigned
to them by KiloSort. You can find that information by calling get_available_clusters_channels():

```python
channels_clusters = trial.get_available_clusters_channels()
```

This will return a dictionary where the keys are channels and the values for each key is a list of the clusters assigned to that channel.

```python
{7: [8],
 8: [4, 13, 14, 122],
 9: [3, 11],
 10: [9, 10, 142, 143, 180, 186],
 11: [0, 1, 2, 5, 7, 12, 19, 23, 140, 141, 147, 200],
 ...
```

I've truncated the output here but you can see that channel 7 has one cluster (8) while channel 8 has 4 (4, 13, 14, 122).

Under the hood the phy library used to pull all this information together actually assigns a cluster to 12 potential channels I just pick out the "best" one for succinctness.

You can access the underlying functions of the phy library by looking at the functions available to the TemplateModel instance that gets added to our **trial** object. In an ipython
terminal type the following (including the period at the end) and hit Tab to see the list of available functions:

```python
trial.template_model.
```

You also can see how to call these functions here:

https://phy.readthedocs.io/en/latest/api/#phyappstemplatetemplatemodel

Now that we know the channels and clusters we have available to us we can plot them out:

```python
import matplotlib.pylab as plt

trial.plot_spike_path(8, 4)
plt.show()
```

It's important to note that the function signature for all of the plot_* and get_* functions is (cluster, channel) which is kind of the other way they are shown to use when we called
get_available_clusters_channels().

In the plot above, depending on your screen resolution etc, the markers denoting where the cell fired may have appeared too small. To fix that add the markersize argument to the 
plotting call:

```python
ax = trial.plot_spike_path(8, 4, ms=6)
plt.show()
```

What's happening here is that ms (markersize) is getting passed to the matplotlib plot command (or something similar) and the markers are plotted at a more appropriate size.

You can also change the color of the squares with the color argument as well as lots of other stuff. Look at the matplotlib documentation for more details. In most cases
the plotting functions return an Axes object that you can manipulate - note sometimes a plotting function may return a handle to the Figure _not_ the Axes - this is
especially true when there are multiple sub-plots in the Figure window.

We can also plot a more processed version of the data by plotting the firing rate map of the cluster:

```python
trial.plot_rate_map(8, 4)
plt.show()
```

This shows where cluster emitted spikes as a heatmap where 'hot' colours are high firing rates and 'cold' colours are low rates (the colormap used is called 'jet' and is awful 
for lots of reasons but it's a de facto standard). White areas in the firing rate map are areas the animal didn't / couldn't go to and so have no valid values.

Before moving on to the other types of plots we can make it's worthwhile to understand the underlying data of the rate map as this will help you understand how the data
is processed and the kinds of arguments you can add/ change in calling functions like plot_rate_map() and the impact they have.

### [BinnedData](./utils.md#ephysiopy.common.utils.BinnedData)

All (op nearly all) of the plot_* (i.e. plot_rate_map(), plot_hd_map() etc) functions have a get_* counterpart (i.e. get_rate_map() etc). The plot functions take what is
returned from the get_* functions and simply plot that data (with some bells and whistles added). The object returned from the get_* functions is almost always (CHECK THIS) a
BinnedData object.

This is a fairly simple [dataclass](https://docs.python.org/3/library/dataclasses.html) that encapsulates some variable(s) that has been binned up according to some algorithm. To give a concrete example given the data
we've been working with above:

```python
data = trial.get_rate_map(8, 4)
```

data here is an instance of the BinnedData class. It has several important properties:

* cluster_id 
* bin_edges
* binned_data
* map_type
* variable

- cluster_id is a list containing a [namedtuple](https://docs.python.org/3/library/collections.html#collections.namedtuple) with fields Cluster and Channel
- bin_edges is a list of numpy ndarray(s) that contains the bin edges of the binned up data
- binned_data is a list of numpy [masked arrays](https://numpy.org/doc/stable/reference/maskedarray.generic.html) but of the actual binned up data 
- map_type is an [Enum](https://docs.python.org/3/library/enum.html) called [MapType]((./utils.md#ephysiopy.common.utils.MapType)) telling us the type of map we have (i.e. RATE, SPK, POS etc)
- variable is an Enum called [VariableToBin](./utils.md#ephysiopy.common.utils.VariableToBin) which tells us the variable that has been binned (X, Y, XY, TIME etc)

In the above example we called the get_rate_map() function with only a single cluster (8) and channel (4). It's possible to call the same function but with lists of clusters
and channels like so:

```python
data = trial.get_rate_map([79, 82], [34, 34])
```

Now 'data' contains data for both clusters (79 & 82). Note the map_type, variable and bin_edges values will all be the same (MapType.RATE, VariableToBin.XY and a list of length 
2 containing the x and y bin edges). You can't have a situation with a BinnedData instance that has different map_type or variable properties - you need to re-calculate and
re-bin the data to do that.

There are several methods available to the BinnedData instance that hopefully make data processing a bit easier.

When analysing data like this programmatically it's often faster and easier to do all the binning at once and access the processed data after that step. To access an individual cluster
from the example above you can call the get_cluster() function:

```python
cluster_data = data.get_cluster(ClusterID(79, 34))
```

#### Correlating maps

One of the commonly performed analysis steps is to correlate rate maps together to try and get a sense of how similar two maps are to each other. To do that you can call
correlate():

```python
data.correlate()
array([0.1179206 , 0.14373477, 0.04730432])
```

This is showing the Pearson Product Moment Correlation Coefficent between the pairs of rate maps in data i.e. map1 correlated with map2 and then map3 and map2 correlated with 
map3 (so three correlation values in total). You can also get the correlation matrix which is mirror symmetric across the diagonal:

```python
data.correlate(as_matrix=True)
array([[1.        , 0.1179206 , 0.14373477],
       [0.1179206 , 1.        , 0.04730432],
       [0.14373477, 0.04730432, 1.        ]])

```

You can also correlate one BinnedData instance with another BinnedData instance:

```python
data1 = trial.get_rate_map([109,110],[36,36])
data.correlate(data1, as_matrix=True)
array([[ 0.21015777,  0.16216187],
       [ 0.0269544 , -0.00144629],
       [ 0.13845497, -0.01680002]])
```

The output is hopefully fairly self-explanatory given its shape...

You can also add together two BinnedData instances to concatenate them:

```python
data2 = data + data1
```

data2 will now contain the binned data for clusters 79 & 82 on channel 34 **and** clusters 109 & 110 on channel 36

Now that we know what the BinnedData contains (you don't need to know the underlying implementation bu design) you can understand the different arguments you can supply
to the get_* and plot_* functions and the effect they have. If we have linear track data we can bin up data according to only one of the X or Y variables (you can also
bin up a PHI variable which is the cartesian distance down the track) and this will give us a 1D rate map:

```python
ratemap = trial.get_rate_map(79, 34, var_type=VariableToBin.X)
```

Now ratemap.bin_edges will be a list of length 1 containing the x bin edges.

#### Iterating over BinnedData

BinnedData can also be iterated over with each iteration yielding a BinnedData instance:

```python
channels_clusters = trial.get_available_clusters_channels()
all_maps = trial.get_all_maps(channels_clusters)
```

all_maps is an instance of BinnedData, the VariableToBin is XY with MapType of RATE (the defaults) and we can iterate over it:

```python
for i_map in all_maps:
  print(f"{i_map.cluster_id}")
ClusterID(Cluster=8, Channel=7)
ClusterID(Cluster=4, Channel=8)
...
```


