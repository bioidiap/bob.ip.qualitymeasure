#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

'''
Compute average specular histogram of an entire picture folder.
'''
from __future__ import print_function

import os
import sys
import argparse
import time

import bob.io.base
import bob.io.image
import bob.io.video
import bob.ip.base
import numpy as np

from bob.ip.qualitymeasure import remove_highlights

def main(command_line_parameters=None):

    argParser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    argParser.add_argument(
        '-p',
        '--path',
        dest='path',
        default=None,
        help='complete path to directory.')

    argParser.add_argument(
        '-o',
        '--output',
        dest='output',
        default=None,
        help='output text file.')

    args = argParser.parse_args(command_line_parameters)

    num_hist = 0.0
    hist     = np.zeros(256, dtype='uint64')

    f = open(args.output, 'w')
    print('# i bin_value', file=f)

    # 1. open input image
    print("Opening dir: %s" % args.path)
    files = os.listdir(args.path)

    # 2. compute
    for file in files:
        print('processing file: %s' % file)

        video = bob.io.video.reader(args.path + file)
        frame = video[0]

        sfi, diff, residue = remove_highlights(frame.astype(np.float32), 0.06)

        residue[np.where(np.isinf(residue))] = 0
        residue[np.where(np.isnan(residue))] = 0
        residue[np.where(residue < 0)] = 0
        residue[np.where(residue > 255)] = 255

        hist = hist + bob.ip.base.histogram(residue[0], (0.0, 255.0), 256)


    # 1. save output image

    for i in range(256):
        print(i, hist[i], file= f)



if __name__ == '__main__':
    main(sys.argv[1:])
