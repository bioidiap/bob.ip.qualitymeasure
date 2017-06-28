#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

'''
Created on 28 Jun 2017

@author: dgeissbuhler
'''

import sys
import argparse

import bob.io.base
import numpy as np

from bob.ip.qualitymeasure import remove_highlights

def main(command_line_parameters=None):
    """Remove the specular component of the input image and write result to
    a file.
    """

    argParser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    argParser.add_argument(
        '-i',
        '--input',
        dest='inpImg',
        default=None,
        help='filename of image to be processed (including complete path). ')

    argParser.add_argument(
        '-o',
        '--output',
        dest='outImg',
        default=None,
        help='filename of specular free image.')

    args = argParser.parse_args(command_line_parameters)

    if not args.inpImg:
        argParser.error('Specify parameter --input')
    if not args.outImg:
        argParser.error('Specify parameter --output')

    # 1. open input image
    print("Opening file: %s" % args.inpImg)
    img = bob.io.image.load(args.inpImg)

    # 2. compute
    print("Extracting diffuse component...")

    sfi, diff, residue = remove_highlights(img.astype(np.float32), 0.5)

    # 1. save output image
    print("Saving output file: %s" % args.outImg)
    bob.io.base.save(diff.astype('uint8'), args.outImg)
    print('Done')


if __name__ == '__main__':
    main(sys.argv[1:])
