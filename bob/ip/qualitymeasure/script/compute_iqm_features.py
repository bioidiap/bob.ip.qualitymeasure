#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

'''
Created on 13 Oct 2015

@author: sbhatta
'''

import os, sys
import argparse

import bob.io.base
import bob.io.video
import bob.ip.color
import numpy as np
import galbally_iqm_features as iqm
import antispoofing.utils.db as bobdb


#'''
#Matlab-like RGB to gray...
#    @param: rgbImage : numpy array for the form: [3,h,w] where h is the height of the image and w is the width of the image.
#    Returns a y-image in floating-point format (range [(16/255) .. (235/255)])
#'''
#def matlab_rgb2gray(rgbImage):
#    #g1 = 0.299*rgbFrame[0,:,:] + 0.587*rgbFrame[1,:,:] + 0.114*rgbFrame[2,:,:] #standard coeffs CCIR601
#    
#    #this is how it's done in matlab...
#    rgbImage = rgbImage / 255.0
#    C0 = 65.481/255.0
#    C1 = 128.553/255.0
#    C2 = 24.966/255.0
#    scaleMin = 16.0/255.0
#    #scaleMax = 235.0/255.0
#    gray = scaleMin + (C0*rgbImage[0,:,:] + C1*rgbImage[1,:,:] + C2*rgbImage[2,:,:])    
#
#    return gray


# """
# loads a video, and returns a feature-vector for each frame of video
# """
# def computeIQM_1video(vidPath):
#     inputVideo = bob.io.video.reader(vidPath)
#     vin = inputVideo.load()
#     numframes = vin.shape[0]
#     fset = np.zeros([numframes, 21])
#     for f in range(numframes):
#         rgbFrame = vin[f,:,:,:]
#         grayFrame = matlab_rgb2gray(rgbFrame) #compute gray-level image for input color-frame
#         bobQFeats = np.asarray(iqm.computeQualityFeatures(grayFrame)) # computeQualityFeatures() returns a tuple
#         fset[f,:] = bobQFeats
#     
#     return fset
#


'''
computes image-quality features for a set of frames comprising a video.
    @param video4d: a '4d' video (N frames, each frame representing an r-g-b image).
    returns  a set of feature-vectors, 1 vector per frame of video4d
'''
def computeVideoIQM(video4d):
    numframes = video4d.shape[0]
    
    #process first frame separately, to get the no. of iqm features
    f=0
    rgbFrame = video4d[f,:,:,:]
    grayFrame = matlab_rgb2gray(rgbFrame) #compute gray-level image for input color-frame
    iqmSet = iqm.compute_quality_features(grayFrame)
    numIQMs = len(iqmSet)
    #now initialize fset to store iqm features for all frames of input video.
    fset = np.zeros([numframes, numIQMs])
    bobQFeats = np.asarray(iqmSet) # computeQualityFeatures() returns a tuple
    fset[f,:] = bobQFeats
    
    for f in range(1,numframes):
        rgbFrame = video4d[f,:,:,:]
#        grayFrame = matlab_rgb2gray(rgbFrame) #compute gray-level image for input color-frame
#        bobQFeats = np.asarray(iqm.compute_quality_features(grayFrame)) # computeQualityFeatures() returns a tuple
        bobQFeats = np.asarray(iqm.compute_quality_features(rgbFrame)) # computeQualityFeatures() returns a tuple
        fset[f,:] = bobQFeats
    
    return fset


'''
loads a video, and returns a feature-vector for each frame of video
'''
def computeIQM_1video(vidPath):
    inputVideo = bob.io.video.reader(vidPath)
    vin = inputVideo.load()
    return computeVideoIQM(vin)
    


def main(command_line_parameters=None):
    
    #code for parsing command line args.
    argParser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argParser.add_argument('-f', '--print_num_files', action='store_true', dest='printNumFiles',
      default=False, help='Option to print no. of files that will be processed. (Default: %(default)s)')
    
    argParser.add_argument('-i', '--input_videofile', dest='inpFile', default = None,
       help='filename of video to be processed (including complete path). Video expected in .mov format.')
    
    argParser.add_argument('-o', '--output_featurefile', dest='outFile', default = None,
       help='filename where computed features will be stored. Output file will be in hdf5 format.')
    
    args = argParser.parse_args(command_line_parameters)
    #make sure the user specifies a folder where feature-files exist
    if not args.inpFile: argParser.error('Specify parameter --input_videofile')
    if not args.outFile: argParser.error('Specify parameter --output_featurefile')

    #1. load video file
    infile = args.inpFile #k.make_path(videoRoot, '.mov')
    #2. compute features, 1 vector per frame of input video.
    bobIqmFeats = computeIQM_1video(infile)
    #3. save features in file 
    outfile = args.outFile #k.make_path(featRoot, '.h5')
    ohf = bob.io.base.HDF5File(outfile, 'w')
    ohf.set('bobiqm', bobIqmFeats)
    del ohf
    

if __name__ == '__main__':
    main(sys.argv[1:])
