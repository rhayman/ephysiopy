#!/usr/bin/env python3
import argparse
parser = argparse.ArgumentParser(description="Exports LFP and TTL data recorded using openephys as numpy files")
parser.add_argument("-d", "--directory", type=str, default='.', help='The directory to look in for files')
parser.add_argument("-c", "--channels", nargs="+", type=str, help="The LFP channel(s) to export")
parser.add_argument("-f", "--freq", type=int, default=4800, help="The output frequency of the LFP data")
parser.add_argument("-t", "--ttl", action='store_true', default=True, help="Export ttl data")
args = parser.parse_args()

from ephysiopy.format_converters.OE_numpy import OE2Numpy

oe = OE2Numpy(args.directory)

oe.getOEData(args.directory)

oe.exportLFP(args.channels, args.freq)
if args.ttl:
	oe.exportTTL()