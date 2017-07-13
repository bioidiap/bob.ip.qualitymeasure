#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

'''
Created on 28 Jun 2017

@author: dgeissbuhler
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

from bob.ip.qualitymeasure import remove_highlights_orig
from bob.ip.qualitymeasure import remove_highlights
from bob.ip.qualitymeasure import tsh

def main(command_line_parameters=None):
    """Remove the specular component of the input image and write result to
    a file.
    """

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
        help='output file.')

    args = argParser.parse_args(command_line_parameters)

    num_hist = 0.0
    hist_v0  = np.zeros(256, dtype='uint64')
    hist_v1  = np.zeros(256, dtype='uint64')
    hist_v2  = np.zeros(256, dtype='uint64')

    f = open(args.output, 'w')
    print('# i v0 v1 v2', file=f)

    # 1. open input image
    print("Opening dir: %s" % args.path)
    files = os.listdir(args.path)

    # 2. compute
    for file in files:
        print('processing file: %s' % file)
        video = bob.io.video.reader(args.path + file)
        frame = video[0]

        sfi, diff, residue = tsh.remove_highlights(frame.astype(np.float32), 0.06)
        residue[np.where(np.isinf(residue))] = 0
        residue[np.where(np.isnan(residue))] = 0
        residue[np.where(residue < 0)] = 0
        residue[np.where(residue > 255)] = 255
        hist_v0 = hist_v0 + bob.ip.base.histogram(residue[0], (0.0, 255.0), 256)

        sfi, diff, residue = remove_highlights_orig(frame.astype(np.float32), 0.06)
        residue[np.where(np.isinf(residue))] = 0
        residue[np.where(np.isnan(residue))] = 0
        residue[np.where(residue < 0)] = 0
        residue[np.where(residue > 255)] = 255
        hist_v1 = hist_v1 + bob.ip.base.histogram(residue[0], (0.0, 255.0), 256)

        sfi, diff, residue = remove_highlights(frame.astype(np.float32), 0.06)
        residue[np.where(np.isinf(residue))] = 0
        residue[np.where(np.isnan(residue))] = 0
        residue[np.where(residue < 0)] = 0
        residue[np.where(residue > 255)] = 255
        hist_v2 = hist_v2 + bob.ip.base.histogram(residue[0], (0.0, 255.0), 256)


    # 1. save output image

    for i in range(256):
        print(i, hist_v0[i], hist_v1[i], hist_v2[i], file= f)



if __name__ == '__main__':
    main(sys.argv[1:])
