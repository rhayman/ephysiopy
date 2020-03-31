#!/usr/bin/python
import os
import argparse
import numpy as np


def printNwbFiles(mydir):
	list_of_dirs = []
	for root, dirs, files in os.walk(mydir):
		for file in files:
			if 'nwb' in file:
				list_of_dirs.append(root)
	for nwbfile in list_of_dirs:
		print(nwbfile)
	return list_of_dirs

parser = argparse.ArgumentParser(description='Process some data.')
parser.add_argument('-d', '--directory', type=str, default='.', help='the directory to look in')
parser.add_argument("-p", "--pos", action='store_true', help="Export position data")
parser.add_argument("-t", "--tetrodes", action='store_true', help="Export tetrode data")
parser.add_argument("--ntetrodes", nargs="+", default=['1', '2', '3', '4'], type=str, help="The tetrodes to use")
parser.add_argument("-l", "--lfp", type=str, default="egf", help="The type of LFP to save.")
parser.add_argument("--ppm", type=int, default=462, help="Pixels per metre")
parser.add_argument("-c", "--channel", type=int, default=0, help="The LFP channel to export")
parser.add_argument("-g", "--gain", type=int, default=5000, help="The gain applied to the LFP channel")
parser.add_argument("-s", "--set", action='store_true', help="Whether to export the set file")
parser.add_argument("-A", "--all", action='store_true', help="Export the everything to give a 'full' Axona data set.")
args = parser.parse_args()

dirs2use = printNwbFiles(args.directory)

from ephysiopy.format_converters import OE_Axona


for idx in dirs2use:
	file_name = os.path.join(idx, 'experiment_1.nwb')
	oe = OE_Axona.OE2Axona(file_name)
	oe.tetrodes = args.ntetrodes

	oe.getOEData(file_name)

	ppm = args.ppm

	if args.all:
		oe.exportPos(ppm=ppm)
		oe.exportSpikes()
		oe.exportLFP(args.channel, args.lfp, args.gain)
		oe.makeSetData(args.channel) # can take **kwargs

	if args.pos:
		oe.exportPos(ppm=ppm)
	if args.tetrodes:
		oe.exportSpikes()
	if args.lfp:
		oe.exportLFP(args.channel, args.lfp, args.gain)
	if args.set:
		oe.makeSetData(args.channel)





