import numpy as np
import os
from subprocess import Popen, PIPE


class Kluster:
    """
    Runs KlustaKwik (KK) against data recorded on the Axona dacqUSB recording
    system

    Inherits from axonaIO.Tetrode to allow for mask construction (for the
    newer version of KK) and easy access to some relevant information (e.g
    number of spikes recorded)
    """

    def __init__(self, filename, tet_num, feature_array):
        """
        Inherits from dacq2py.IO so as to be able to load .set file for fmask
        construction (ie remove grounded channels, eeg channels etc)

        Parameters
        ---------------
        filename: fully qualified, absolute root filename (i.e. without the
        .fet.n)

        tet_num: the tetrode number

        feature_array: array containing the features to be put into the fet file

        """
        self.filename = filename
        self.tet_num = tet_num
        self.feature_array = feature_array
        self.n_features = feature_array.shape[1] / 4
        self.distribution = 1
        self.feature_mask = None

    def make_fet(self):
        """
        Creates and writes a .fet.n file for reading in  to KlustaKwik given
        the array input a
        """
        fet_filename = self.filename + ".fet." + str(self.tet_num)
        with open(fet_filename, "w") as f:
            f.write(str(self.feature_array.shape[1]))
            f.write("\n")
            np.savetxt(f, self.feature_array, fmt="%1.5f")

    def get_mask(self):
        """
        Returns a feature mask based on unused channels, eeg recordings etc
        Loads the set file associated with the trial and creates two dicts
        containing the mode on each channel and the channels which contain the
        eeg recordings
        keys and values for both dicts are in the form of ints NB mode is
        numbered from 0 upwards but eeg_filter is from 1 upwards
        The mode key/ value pair of the Axona .set file correspond to the
        following values:
        2 - eeg
        5 - ref-sig
        6 - grounded
        The collectMask key in the Axona .set file corresponds to whether or
        not a tetrode was kept in the recording - use this also to construct
        the feature mask
        """
        #  use the feature array a to calculate which channels to include etc
        sums = np.sum(self.feature_array, 0)
        feature_mask = np.repeat(np.ones(4, dtype=int), self.n_features)
        #  if there are "missing" channels use the older version of KK
        zero_sums = sums == 0
        if np.any(zero_sums):
            self.distribution = 1
            feature_mask[zero_sums] = 0
        self.feature_mask = feature_mask
        return feature_mask

    def make_fmask(self, feature_mask):
        """
        Create a .fmask.n file for use in the new (01/09/14) KlustaKwik program
        where n denotes tetrode id
        From the github site:
        "The .fmask file is a text file, every line of which is a vector of
        length the number of features, in which 1 denotes unmasked and 0
        denotes masked, and values between 0 and 1 indicate partial masking"
        Inputs:
                filename: fully qualified, absolute root filename (i.e. without the
                .fmask.n)
                a: array containing the features to be put into the fet file
                n: the tetrode number
                feature_mask: array of numbers between 0 and 1 (see above
                description from github site)
        """
        fmask_filename = self.filename + ".fmask." + str(self.tet_num)
        mask = np.tile(feature_mask, (self.feature_array.shape[0], 1))
        with open(fmask_filename, "w") as f:
            f.write(str(self.feature_array.shape[1]))
            f.write("\n")
            np.savetxt(f, mask, fmt="%1d")

    def kluster(self):
        """
        Using a .fet.n file this makes a system call to KlustaKwik (KK) which
        clusters data and saves istributional ' + str(self.distribution) +
                ' -MinClusters 5'
                ' -MaxPossibleClusters 31'
                ' -MaskStarts 30'
                ' -FullStepEvery 1'
                ' -SplitEvery 40'
                ' -UseMaskedInitialConditions 1'
                ' -AssignToFirstClosestMask 1'
                ' -DropLastNFeatures 1'
                ' -RandomSeed 123'
                ' -PriorPoint 1'
                ' -MaxIter 10000'
                ' -PenaltyK 1'
                ' -PenaltyKLogN 0'
                ' -Log 0'
                ' -DistThthe result in a cut file that can be read into
        Axona's Tint cluster cutting app
        Inputs:
                fname - the root name of the file (i.e. without the .fet.n)
        Outputs:
                None but saves a Tint-friendly cut file in the same directory as
                the spike data
        """
        # specify path to KlustaKwik exe
        kk_path = r"/media/robin/data/Dropbox/Programming/klustakwik/KlustaKwik"
        if not os.path.exists(kk_path):
            print(kk_path)
            raise IOError()
        kk_proc = Popen(
            kk_path
            + " "
            + self.filename
            + " "
            + str(self.tet_num)
            + " -UseDistributional "
            + str(self.distribution)
            + " -MinClusters 5"
            " -MaxPossibleClusters 31"
            " -MaskStarts 30"
            " -FullStepEvery 1"
            " -SplitEvery 40"
            " -UseMaskedInitialConditions 1"
            " -AssignToFirstClosestMask 1"
            " -DropLastNFeatures 1"
            " -RandomSeed 123"
            " -PriorPoint 1"
            " -MaxIter 10000"
            " -PenaltyK 1"
            " -PenaltyKLogN 0"
            " -Log 0"
            " -DistThresh 9.6"
            " -UseFeatures " + "".join(map(str, self.feature_mask)),
            shell=True,
            stdout=PIPE,
        )
        # Print the output of the KlustaKwik algo
        for line in kk_proc.stdout:
            print(line.replace("\n", ""))

        """
		now read in the .clu.n file that has been created as a result of this
		process and create the Tint-friendly cut file
		"""
        clu_filename = self.filename + ".clu." + str(self.tet_num)
        clu_data = np.loadtxt(clu_filename)
        n_clusters = clu_data[0]
        clu_data = clu_data[1:] - 1  # -1 so cluster 0 is junk
        n_chan = 4
        n_spikes = int(clu_data.shape[0])
        cut_filename = self.filename.split(".")[0] + "_" + str(self.tet_num) + ".cut"
        with open(cut_filename, "w") as f:
            f.write(
                "n_clusters: {nClusters}\n".format(nClusters=n_clusters.astype(int))
            )
            f.write("n_channels: {nChan}\n".format(nChan=n_chan))
            f.write("n_params: {nParam}\n".format(nParam=2))
            f.write("times_used_in_Vt:    {Vt}    {Vt}    {Vt}    {Vt}\n".format(Vt=0))
            for i in range(0, n_clusters.astype(int)):
                f.write(
                    " cluster: {i} center:{zeros}\n".format(
                        i=i, zeros="    0    0    0    0    0    0    0    0"
                    )
                )
                f.write(
                    "                min:{zeros}\n".format(
                        i=i, zeros="    0    0    0    0    0    0    0    0"
                    )
                )
                f.write(
                    "                max:{zeros}\n".format(
                        i=i, zeros="    0    0    0    0    0    0    0    0"
                    )
                )
            f.write(
                "Exact_cut_for: {fname} spikes: {nSpikes}\n".format(
                    fname=os.path.basename(self.filename), nSpikes=str(n_spikes)
                )
            )
            for spk in clu_data:
                f.write("{spk}  ".format(spk=spk.astype(int)))
