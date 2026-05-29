# pyright: reportOverlappingOverload=false
# pyright:  reportInconsistentOverload=false
# pyright:  reportNoOverloadImplementation=false
# pyright:  reportOverlappingOverload=false

import os
import warnings
import numpy as np
import pandas as pd
from collections import OrderedDict
from pathlib import Path
from phylib.utils import Bunch
from phylib.io.model import get_closest_channels


def fileExists(pname, fname) -> bool:
    return os.path.exists(os.path.join(pname, fname))


class KiloSortSession(object):
    """
    Loads and processes data from a Kilosort session.

    A kilosort session results in a load of .npy files, a .csv or .tsv file.
    The .npy files contain things like spike times, cluster indices and so on.
    Importantly the .csv (or .tsv) file contains the cluster identities of
    the SAVED part of the phy template-gui (ie when you click "Save" from the
    Clustering menu): this file consists of a header ('cluster_id' and 'group')
    where 'cluster_id' is obvious (relates to identity in spk_clusters.npy),
    the 'group' is a string that contains things like 'noise' or 'unsorted' or
    whatever as the phy user can define their own labels.
    """

    # this mirrors TemplateModel
    amplitude_threshold = 0
    n_closest_channels = 12

    def __init__(self, src_dir):
        """
        Initialize the KiloSortSession.

        Walk through the path to find the location of the files in case this
        has been called in another way i.e., binary format a la Neuropixels.

        Parameters
        ----------
        src_dir : str
            The top-level directory containing the Kilosort session files.
        """
        self.src_dir = src_dir

        for d, c, f in os.walk(src_dir):
            for ff in f:
                if "." not in c:  # ignore hidden directories
                    if "spike_times.npy" in ff:
                        self.src_dir = d
        self.cluster_id = None
        self.spk_clusters = None
        self.spike_times = None
        self.amplitudes = None
        self.contamPct = None
        self.good_clusters = []
        self.mua_clusters = []
        self.wm = None
        self.wmi = None
        self.channel_positions = None
        self.templates = None
        self.templates_ind = None
        self.sparse_templates = None
        self.cluster_info = None

    def load(self):
        """
        Load all the relevant files

        There is a distinction between clusters assigned during the automatic
        spike sorting process (here KiloSort2) and the manually curated
        distillation of the automatic process conducted by the user with
        a program such as phy.

        * The file cluster_KSLabel.tsv is output from KiloSort.
            All this information is also contained in the cluster_info.tsv
            file! Not sure about the .csv version (from original KiloSort?)
        * The files cluster_group.tsv or cluster_groups.csv contain
            "group labels" from phy ('good', 'MUA', 'noise' etc).
            One of these (cluster_groups.csv or cluster_group.tsv)
            is from kilosort and the other from kilosort2
            I think these are only amended to once a phy session has been
            run / saved...
        """
        import os

        # read in params.py and get the sample rate from there
        params_path = self.src_dir / Path("params.py")
        if params_path.exists():
            with open(params_path, "r") as f:
                for line in f:
                    if line.startswith("sample_rate"):
                        self.sample_rate = float(line.split("=")[1].strip())
                    elif line.startswith("n_channels_dat"):
                        self.n_channels_dat = int(line.split("=")[1].strip())
                    elif line.startswith("dtype"):
                        self.dtype = line.split(
                            "=")[1].strip().strip('"').strip("'")

        dtype = [("cluster_id", "i4"), ("group", "U20")]
        # One of these (cluster_groups.csv or cluster_group.tsv) is from
        # kilosort and the other from kilosort2
        # and is updated by the user when doing cluster assignment in phy
        # See comments above this class definition for a bit more info

        def __load_KS_tsv__(s):
            self.cluster_id, self.group = np.genfromtxt(
                self.src_dir / Path(s),
                unpack=True,
                skip_header=1,
                delimiter="\t",
                dtype=dtype,
                encoding="utf-8",
            )

        if fileExists(self.src_dir, "cluster_groups.csv"):
            __load_KS_tsv__("cluster_groups.tsv")

        elif fileExists(self.src_dir, "cluster_group.tsv"):
            __load_KS_tsv__("cluster_group.tsv")

        else:
            warnings.warn(
                f"\nNo cluster_groups.csv or cluster_group.tsv file found.\n\
                Falling back to KiloSort labels\nCheck directory: {
                    self.src_dir
                } for cluster_groups.csv or cluster_group.tsv files."
            )
            # 'Raw' labels from a kilosort session
            if fileExists(self.src_dir, "cluster_KSLabel.tsv"):
                __load_KS_tsv__("cluster_KSLabel.tsv")
            else:
                warnings.warn(
                    "No cluster_KSLabel.tsv file was found.\
                    Run KiloSort and make sure this file is saved."
                )
        """
        Output some information to the user if self.cluster_id is still None
        it implies that data has not been sorted / curated
        """

        # HWPD 20200527
        # load cluster_info file and add X co-ordinate to it
        if fileExists(self.src_dir, "cluster_info.tsv"):
            self.cluster_info = pd.read_csv(
                os.path.join(self.src_dir, "cluster_info.tsv"), sep="\t"
            )

        if fileExists(self.src_dir, "cluster_ContamPct.tsv"):
            self.contamPct = pd.read_csv(
                os.path.join(self.src_dir, "cluster_ContamPct.tsv"), sep="\t"
            )

        files_to_load = [
            "spike_clusters.npy",
            "amplitudes.npy",
            "spike_times.npy",
            "templates.npy",
            "templates_ind.npy",
            "spike_templates.npy",
            "channel_positions.npy",
        ]

        def __get_KS_npy__(s):
            a = np.ma.MaskedArray(np.load(self.src_dir / Path(s)))
            return np.squeeze(a)

        [
            setattr(
                self,
                s.split(".")[0],
                __get_KS_npy__(s) if fileExists(self.src_dir, s) else None,
            )
            for s in files_to_load
        ]
        # iterate over self.cluster_id and self.group and make sure
        # self.cluster_id has spikes
        # not sure why but some are of 0 length when loading
        # the KSLabel based stuff...
        has_spikes = [len(self.get_cluster_spikes(c))
                      > 0 for c in self.cluster_id]
        self.cluster_id = self.cluster_id[has_spikes]
        self.group = self.group[has_spikes]

        return True

    def removeNoiseClusters(self):
        """
        Removes clusters with labels 'noise' and 'mua' in self.group
        """
        if self.cluster_id is not None:
            self.good_clusters = []
            for id_group in zip(self.cluster_id, self.group):
                if "noise" not in id_group[1] and "mua" not in id_group[1]:
                    self.good_clusters.append(id_group[0])

    def get_cluster_spike_times(self, cluster: int) -> np.ndarray:
        """
        Returns the spike times for a given cluster in samples.

        Parameters
        ----------
        cluster : int
            The cluster ID.

        Returns
        -------
        np.ndarray
            The spike times for the specified cluster.
        """
        if cluster in self.cluster_id:
            return self.spike_times[self.spk_clusters == cluster]

    def get_cluster_channels(self, cluster: int) -> np.ndarray:
        """
        Returns the most relevant channels for a given cluster.

        Parameters
        ----------
        cluster : int
            The cluster ID.

        Returns
        -------
        np.ndarray
            The channels for the specified cluster.
        """
        spike_ids = self.get_cluster_spikes(cluster)
        return self._get_template_from_spikes(spike_ids).channel_ids.astype(int)

    def _get_template_from_spikes(self, spike_ids):
        """
        Returns the template for a given set of spike ids.

        Parameters
        ----------
        spike_ids : np.ndarray
            The spike indices.

        Returns
        -------
        np.ndarray
            The template for the specified spikes.
        """
        st = self.spike_templates[spike_ids]
        template_ids, counts = np.unique(st, return_counts=True)
        ind = np.argmax(counts)
        template_id = template_ids[ind]
        template = self.get_template(template_id)
        return template

    def get_cluster_spikes(self, cluster: int) -> np.ndarray:
        """
        Returns the spike ids that belong to a given template.

        Parameters
        ----------
        cluster : int
            The cluster ID.

        Returns
        -------
        np.ndarray
            The spike indices for the specified cluster.
        """
        if not np.any(self.spike_clusters):
            self._load_spike_clusters()

        return self._spikes_in_clusters(self.spike_clusters, [cluster])

    def _spikes_in_clusters(self, clusters, cluster_ids):
        """
        Returns the spike ids that belong to a given cluster.

        Parameters
        ----------
        clusters : np.ndarray
            The cluster IDs for each spike.
        cluster_ids : list
            The cluster IDs to filter.

        Returns
        -------
        np.ndarray
            The spike indices for the specified cluster.
        """
        if len(clusters) == 0 or len(cluster_ids) == 0:
            return np.array([], dtype=int)
        return np.nonzero(np.isin(clusters, cluster_ids))[0]

    def apply_mask(self, mask: np.ma.MaskedArray | bool, **kwargs):
        """
        Apply a mask to the data.

        Parameters
        ----------
        mask :bool or array of mask values

        """
        # this is slightly annoying as we need to bin up spikes in time
        # so that the length of that vector is the same as the position mask
        # then select the spikes that fall within the position mask and mask those
        # the xy times might not be equally sampled (indeed likely arent)
        # mask might be a single bool (False) to remove the masking
        # or an array of mask values to apply to the spike times and clusters
        if isinstance(mask, np.ma.MaskedArray):
            xy_ts = kwargs.get("xy_ts", None)  # in seconds to 2 decimal places
            # extract the masked times
            xy_ts = xy_ts.data[~xy_ts.mask]
            sample_rate = kwargs.get("sample_rate", None)
            spike_times = np.round(
                self.spike_times / self.sample_rate, 2
            )  # in seconds to 2 decimal places
            indices = np.digitize(spike_times, xy_ts)
            indices_seconds = np.round(indices / sample_rate, 2)

            mask = np.ma.isin(indices_seconds, xy_ts, assume_unique=True)

        if isinstance(self.spike_times, np.ma.MaskedArray):
            self.spike_times.mask = mask
        else:
            self.spike_times = np.ma.MaskedArray(self.spike_times, mask)
        if isinstance(self.spike_clusters, np.ma.MaskedArray):
            self.spike_clusters.mask = mask
        else:
            self.spike_clusters = np.ma.MaskedArray(self.spike_clusters, mask)
        if isinstance(self.amplitudes, np.ma.MaskedArray):
            self.amplitudes.mask = mask
        else:
            self.amplitudes = np.ma.MaskedArray(self.amplitudes, mask)

    def _load_channel_positions(self):
        """
        Load the channel positions
        """
        self.channel_positions = np.load(
            self.src_dir / Path("channel_positions.npy"))

    def _load_spike_clusters(self):
        """
        Load the spike clusters
        """
        self.spike_clusters = np.load(
            self.src_dir / Path("spike_clusters.npy"))

    def _load_wmi(self):
        """
        Load the inverse whitening matrix
        """
        self.wmi = np.load(self.src_dir / Path("whitening_mat_inv.npy"))

    def _load_wm(self):
        """
        Load the whitening matrix
        """
        self.wm = np.load(self.src_dir / Path("whitening_mat.npy"))

    def _load_templates(self) -> tuple | None:
        """
        Read the templates.npy file

        Parameters
        ----------
        src_dir - Path
            The location of all the KiloSort files

        Returns
        -------
        np.ndarray - the result

        """
        fpath = self.src_dir / Path("templates.npy")
        assert fpath.exists(), f"{fpath} does not exist"

        try:
            data = np.load(fpath, mmap_mode="r+")
            data = np.atleast_3d(data)

            empty_templates = np.all(np.all(np.isnan(data), axis=1), axis=1)
            data[empty_templates, ...] = 0
            self.n_templates, _, self.n_channels_loc = data.shape

        except IOError:
            return

        try:
            cols = np.load(
                self.src_dir / Path("template_ind.npy"), mmap_mode="r+")
            cols = np.atleast_2d(cols)

            assert cols.shape == (
                self.n_templates,
                self.n_channels_loc,
            ), f"Expected shape {(self.n_templates, self.n_channels_loc)}, got {
                cols.shape
            }"

        except IOError:
            cols = None

        B = Bunch(data=data, cols=cols)

        self.sparse_templates = B

        return data, cols

    def _get_dense_templates(
        self,
        template_id,
        channel_ids=None,
        amplitude_threshold=None,
        unwhiten=True,
    ):
        """
        Get the dense template for a given template ID.

        Parameters
        ----------
        template_id : int
            The ID of the template to retrieve.
        channel_ids : list of int, optional
            The channel IDs to include in the template. If None, all channels are included.

        amplitude_threshold : float, optional
            The minimum amplitude threshold for including a channel in the template. If None, no threshold is
             applied.
        unwhiten : bool, optional
            Whether to unwhiten the template using the whitening matrix. Default is True.

        """
        if not self.sparse_templates:
            self._load_templates()

        template_w = self.sparse_templates.data[template_id, ...]
        template = (
            self._unwhiten(template_w.astype(np.float32)
                           ) if unwhiten else template_w
        )
        channel_ids, amplitude, best_channel = self._find_best_channels(
            template, amplitude_threshold
        )

        assert template.ndim == 2

        return Bunch(
            template=template,
            amplitude=amplitude,
            channel_ids=channel_ids,
            best_channel=best_channel,
        )

    def get_template(
        self,
        template_id,
        channel_ids=None,
        amplitude_threshold=None,
        unwhiten=True,
    ):
        """
        Get the template for a given template ID.

        Parameters
        ----------
        template_id : int
            The ID of the template to retrieve.
        channel_ids : list of int, optional
            The channel IDs to include in the template. If None, all channels are included.
        amplitude_threshold : float, optional
            The minimum amplitude threshold for including a channel in the template. If None, no threshold is applied.
        unwhiten : bool, optional
            Whether to unwhiten the template using the whitening matrix. Default is True.

        Returns
        -------
        np.ndarray
            The template for the given template ID.
        """
        _, cols = self._load_templates()

        if cols is None:
            return self._get_dense_templates(
                template_id, channel_ids, amplitude_threshold, unwhiten
            )

    def _load_sparse_templates(self):
        """
        This is a misnomer for the recordings I've done as the templates
        aren't sparse but dense and so in the _load_templates function above
        the cols part of the returned parameters is None. Anyway a few shape/
        dimension parameters are set Anyway
        """
        if not self.templates:
            self._load_templates()
        n_templates, n_samples_waveforms, n_channels_loc = self.templates.shape

    def _unwhiten(self, data: np.ndarray, channel_ids=None) -> np.ndarray:
        """
        Unwhiten some data

        Parameters
        ----------
        data : np.ndarray
            The data to unwhiten.
        channel_ids : list of int, optional
            The channel IDs to include in the unwhitening process. If None, all channels are included.

        Returns
        -------
        np.ndarray
            The unwhitened data.

        """
        if not np.any(self.wmi):
            self._load_wmi()

        wmi = self.wmi

        if channel_ids is not None:
            wmi = wmi[np.ix_(channel_ids, channel_ids)]
            assert wmi.shape == (len(channel_ids),) * 2

        assert data.shape[1] == wmi.shape[0]

        out = np.dot(data, wmi) * getattr(self, "template_scaling", 1.0)

        return out

    def _find_best_channels(self, template, amplitude_threshold=0):
        """
        Find the best channels for a given template

        Parameters
        ----------
        template
        amplitude_threshold - this is set at the class level in TemplateModel
            and is 0 by default

        Notes
        -----
        n_closest_channels set at the class level
        """

        amplitude = np.ptp(template, axis=0)
        assert not np.all(np.isnan(amplitude)), "Template is all NaNs!"

        best_channel = np.argmax(amplitude)
        max_amp = amplitude[best_channel]

        amplitude_threshold = getattr(
            self, "amplitude_threshold", amplitude_threshold)

        peak_channels = np.nonzero(
            amplitude >= amplitude_threshold * max_amp)[0]

        if not np.any(self.channel_positions):
            self._load_channel_positions()

        close_channels = get_closest_channels(
            self.channel_positions, best_channel, self.n_closest_channels
        )

        assert best_channel in close_channels

        channel_ids = np.intersect1d(peak_channels, close_channels).astype(int)
        # for some fucking annoying reason the channel_ids are floats
        # convert to int here - doing this using astype(int) doesn't work
        # for some reason
        channel_ids = np.array([int(i) for i in channel_ids])
        order = np.argsort(amplitude[channel_ids])[::-1]
        channel_ids = channel_ids[order]
        amplitude = amplitude[order]

        return channel_ids, amplitude, best_channel

    def get_all_channels_clusters(self) -> dict:
        """
        Get the best channel(s) for all the clusters

        Returns
        -------
        dict
            A dictionary where the keys are the best channel(s) and the values
            are the clusters for the best channel

        Notes
        -----
        To maintain consistency with the way the Axona cut data is handled
        only a single channel is returned for each cluster, even if multiple channels are above the amplitude threshold. This is done by taking the best channel (the one with the highest amplitude) and then finding the closest channels to it. The intersection of these two sets of channels is then taken as the final set of channels for that cluster.
        If there are multiple channels in this intersection, only the best channel
        is returned.

        To retrieve the n best channels for each cluster, the get_cluster_channels function can be used directly for each cluster ID.

        """
        all_channels = {}

        for cluster in self.cluster_id:
            channel = int(self.get_cluster_channels(cluster)[0])

            if channel in all_channels:
                all_channels[channel].append(int(cluster))
            else:
                all_channels[channel] = [int(cluster)]

        # remove any empty channels
        {ky: va for ky, va in all_channels.items() if va}

        # sort by channel number
        all_channels = OrderedDict(sorted(all_channels.items()))

        return all_channels
