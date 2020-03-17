# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 17:55:43 2012

@author: robin
"""

import os, sys, getopt
from ephysiopy import dacq2py

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hb:c:",["base=","comment_key="])
    except getopt.GetoptError:
        print("dacq_comment_finder.py -b <basedir> -c <comment_key>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print("dacq_comment_finder.py -b <basedir> -c <comment_key>")
        elif opt in ("-b", "--base"):
            basedir = arg
        elif opt in ("-c", "--comment_key"):
            comment = arg
    file_list = []
    n = 0
    for r, d, f in os.walk(basedir):
        for ff in f:
            if ff.endswith(('set','SET')):
                a = dacq2py.IO(os.path.join(r,ff[0:-4]))
                h = a.getHeader(os.path.join(r,ff))
                n += 1
                try:
                    if comment in h['comments']:
                        file_list.append(os.path.join(r,ff))
                        print(os.path.join(r,ff))
                except KeyError:
                    if comment in h['trial_comment']:
                        file_list.append(os.path.join(r,ff))
                        print(os.path.join(r,ff))
    print("\n")
    print("Num files searched: ", n)
    print("Num files found: ", len(file_list), "\n")
    return file_list

if __name__ == "__main__":
    main(sys.argv[1:])