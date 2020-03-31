#!/usr/bin/env python3
import argparse
parser = argparse.ArgumentParser(description="quick script for debugging")
parser.add_argument("-d", "--directory", type=str, default='.', help='The directory to look in for files')
parser.add_argument("-p", "--pos", action='store_true', help="Export position data")
parser.add_argument("--postxt", action='store_true', help="Export pos data to txt file")
parser.add_argument("-t", "--tetrodes", action='store_true', help="Export tetrode data")
parser.add_argument("--eeg", action='store_true', help="Export LFP to an eeg file.")
parser.add_argument("--egf", action='store_true', help="Export LFP to an egf file.")
parser.add_argument("--ppm", type=int, default=300, help="Pixels per metre")
parser.add_argument("-c", "--channel", type=int, default=0, help="The LFP channel to export")
parser.add_argument("-g", "--gain", type=int, default=6000, help="The gain applied to the LFP channel")
parser.add_argument("-A", "--all", action='store_true', help="Export the everything to give a 'full' Axona data set.")
parser.add_argument("--ntetrodes", nargs="+", default=['1', '2', '3', '4'], type=str, help="The tetrodes to use")

args = parser.parse_args()

from ephysiopy.format_converters import OE_Axona

oe = OE_Axona.OE2Axona(args.directory)

oe.getOEData(args.directory)
oe.lp_gain = 1200000
oe.ntetrodes = args.ntetrodes
ppm = args.ppm

if args.all:
	oe.exportPos(ppm=ppm)
	oe.exportSpikes()
	oe.exportLFP(args.channel, "egf", args.gain)
	oe.makeSetData(args.channel) # can take **kwargs

if args.postxt:
	oe.exportPos(ppm=ppm, as_text=True)
if args.pos:
	oe.exportPos(ppm=ppm)
if args.tetrodes:
	oe.exportSpikes()
if args.eeg:
	oe.exportLFP(args.channel, "eeg", args.gain)
if args.egf:
	oe.exportLFP(args.channel, "egf", args.gain)