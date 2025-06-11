import os
import warnings
import numpy as np


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

    def __init__(self, fname_root):
        """
        Initialize the KiloSortSession.

        Walk through the path to find the location of the files in case this
        has been called in another way i.e., binary format a la Neuropixels.

        Parameters
        ----------
        fname_root : str
            The top-level directory containing the Kilosort session files.
        """
        self.fname_root = fname_root

        for d, c, f in os.walk(fname_root):
            for ff in f:
                if "." not in c:  # ignore hidden directories
                    if "spike_times.npy" in ff:
                        self.fname_root = d
        self.cluster_id = None
        self.spk_clusters = None
        self.spike_times = None
        self.amplitudes = None
        self.contamPct = None
        self.good_clusters = []
        self.mua_clusters = []

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

        import pandas as pd

        dtype = {"names": ("cluster_id", "group"), "formats": ("i4", "<U10")}
        # One of these (cluster_groups.csv or cluster_group.tsv) is from
        # kilosort and the other from kilosort2
        # and is updated by the user when doing cluster assignment in phy
        # See comments above this class definition for a bit more info
        if fileExists(self.fname_root, "cluster_groups.csv"):
            self.cluster_id, self.group = np.loadtxt(
                os.path.join(self.fname_root, "cluster_groups.csv"),
                unpack=True,
                skiprows=1,
                dtype=dtype,
            )
        if fileExists(self.fname_root, "cluster_group.tsv"):
            self.cluster_id, self.group = np.loadtxt(
                os.path.join(self.fname_root, "cluster_group.tsv"),
                unpack=True,
                skiprows=1,
                dtype=dtype,
            )

        """
        Output some information to the user if self.cluster_id is still None
        it implies that data has not been sorted / curated
        """
        # if self.cluster_id is None:
        #     print(f"Searching {os.path.join(self.fname_root)} and...")
        #     warnings.warn("No cluster_groups.tsv or cluster_group.csv file
        # was found.\
        #         Have you manually curated the data (e.g with phy?")

        # HWPD 20200527
        # load cluster_info file and add X co-ordinate to it
        if fileExists(self.fname_root, "cluster_info.tsv"):
            self.cluster_info = pd.read_csv(
                os.path.join(self.fname_root, "cluster_info.tsv"), sep="\t"
            )
            if fileExists(self.fname_root, "channel_positions.npy") and fileExists(
                self.fname_root, "channel_map.npy"
            ):
                chXZ = np.load(os.path.join(self.fname_root, "channel_positions.npy"))
                chMap = np.load(os.path.join(self.fname_root, "channel_map.npy"))
                chID = np.asarray(
                    [np.argmax(chMap == x) for x in self.cluster_info.ch.values]
                )
                self.cluster_info["chanX"] = chXZ[chID, 0]
                self.cluster_info["chanY"] = chXZ[chID, 1]

        dtype = {"names": ("cluster_id", "KSLabel"), "formats": ("i4", "<U10")}
        # 'Raw' labels from a kilosort session
        if fileExists(self.fname_root, "cluster_KSLabel.tsv"):
            self.ks_cluster_id, self.ks_group = np.loadtxt(
                os.path.join(self.fname_root, "cluster_KSLabel.tsv"),
                unpack=True,
                skiprows=1,
                dtype=dtype,
            )
        if fileExists(self.fname_root, "cluster_ContamPct.tsv"):
            _, self.contamPct = np.loadtxt(
                os.path.join(self.fname_root, "cluster_ContamPct.tsv"),
                unpack=True,
                skiprows=1,
                dtype=dtype,
            )
        if fileExists(self.fname_root, "spike_clusters.npy"):
            self.spk_clusters = np.ma.MaskedArray(
                np.squeeze(np.load(os.path.join(self.fname_root, "spike_clusters.npy")))
            )
        if fileExists(self.fname_root, "amplitudes.npy"):
            self.amplitudes = np.ma.MaskedArray(
                np.squeeze(np.load(os.path.join(self.fname_root, "amplitudes.npy")))
            )
        if fileExists(self.fname_root, "spike_times.npy"):
            self.spike_times = np.ma.MaskedArray(
                np.squeeze(np.load(os.path.join(self.fname_root, "spike_times.npy")))
            )
        if fileExists(self.fname_root, "templates.npy"):
            self.templates = np.load(os.path.join(self.fname_root, "templates.npy"))
        if fileExists(self.fname_root, "templates_ind.npy"):
            self.templates_ind = np.load(
                os.path.join(self.fname_root, "templates_ind.npy")
            )

            return True
        warnings.warn(
            "No spike times or clusters were found \
            (spike_times.npy or spike_clusters.npy).\
                You should run KiloSort"
        )
        return False

    def removeNoiseClusters(self):
        """
        Removes clusters with labels 'noise' and 'mua' in self.group
        """
        if self.cluster_id is not None:
            self.good_clusters = []
            for id_group in zip(self.cluster_id, self.group):
                if "noise" not in id_group[1] and "mua" not in id_group[1]:
                    self.good_clusters.append(id_group[0])

    def removeKSNoiseClusters(self):
        """
        Removes "noise" and "mua" clusters from the kilosort labelled stuff
        """
        for cluster_id, kslabel in zip(self.ks_cluster_id, self.ks_group):
            if "good" in kslabel:
                self.good_clusters.append(cluster_id)
            if "mua" in kslabel:
                self.mua_clusters.append(cluster_id)

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
        if cluster in self.ks_cluster_id:
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
        return self._get_template_from_spikes(spike_ids).channel_ids

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

    def get_template(self, template_id: int):
        """
        Returns the template for a given template ID.

        Parameters
        ----------
        template_id : int
            The template ID.

        Returns
        -------
        np.ndarray
            The template for the specified ID.
        """
        data, cols = self.templates, self.templates_ind
        template_w, channel_ids = data[template_id], cols[template_id]
        # template_w is (n_samples, n_channels):

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

    def apply_mask(self, mask, **kwargs):
        """
        Apply a mask to the data.

        Parameters
        ----------
        mask : tuple
            A tuple (start, end) in seconds specifying the mask range.

        Notes
        -----
        The times inside the bounds are masked, i.e., the mask is set to True.
        The mask can be a list of tuples, in which case the mask is applied
        for each tuple in the list. The mask can be an empty tuple, in which
        case the mask is removed.

        """
        # get spike and pos times into position sample units
        xy_ts = kwargs.get("xy_ts", None)
        sample_rate = kwargs.get("sample_rate", 50)
        spike_pos_samples = np.ma.MaskedArray(
            self.spike_times / 30000 * sample_rate, dtype=int
        )
        pos_times_in_samples = np.ma.MaskedArray(xy_ts * sample_rate, dtype=int)
        mask = np.isin(spike_pos_samples, pos_times_in_samples)
        if isinstance(self.spike_times, np.ma.MaskedArray):
            self.spike_times.mask = mask.data
        else:
            self.spike_times = np.ma.MaskedArray(self.spike_times, mask)
        if isinstance(self.spk_clusters, np.ma.MaskedArray):
            self.spk_clusters.mask = mask.data
        else:
            self.spk_clusters = np.ma.MaskedArray(self.spk_clusters, mask)
        if isinstance(self.amplitudes, np.ma.MaskedArray):
            self.amplitudes.mask = mask.data
        else:
            self.amplitudes = np.ma.MaskedArray(self.amplitudes, mask)
