'''
Created on 13 Oct 2015

@author: sbhatta
'''

import os, sys
import argparse

import bob.io.base
import bob.io.image #under the hood: loads Bob plugin for image file

import bob.io.video
import bob.ip.color
import numpy as np
import msu_iqa_features as iqa
#import MSU_MaskedIQAFeats as iqa
import antispoofing.utils.db as bobdb


'''
computes image-quality features for a set of frames comprising a video.
    @param video4d: a '4d' video (N frames, each frame representing an r-g-b image).
    returns  a set of feature-vectors, 1 vector per frame of video4d
'''
def computeVideoIQA(video4d, validFrames):
    numframes = video4d.shape[0]
    
    #process first frame separately, to get the no. of iqm features
    
    numValidFrames = np.sum(validFrames)
    k=0
    while validFrames[k] == 0: k+=1
    print 'first valid frame: ', k
    rgbFrame = video4d[k,:,:,:]
    
    iqmSet = iqa.computeMsuIQAFeatures(rgbFrame)

    numIQMs = len(iqmSet)
    #now initialize fset to store iqm features for all frames of input video.
    fset = np.zeros([numValidFrames, numIQMs])
    msuQFeats = np.asarray(iqmSet) # computeQualityFeatures() returns a tuple
    fset[0,:] = msuQFeats
    print 'fset shape:', fset.shape
    j=1
    for f in range(k+1,numframes):
        if validFrames[f]==1:
            rgbFrame = video4d[f,:,:,:]
            #grayFrame = matlab_rgb2gray(rgbFrame) #compute gray-level image for input color-frame
            msuQFeats = np.asarray(iqa.computeMsuIQAFeatures(rgbFrame)) # computeQualityFeatures() returns a tuple
            fset[j,:] = msuQFeats
            #print j, f
            j += 1
            
  
    return fset


'''
loads a video, and returns a feature-vector for each frame of video
'''
def computeIQA_1video(videoFile, frameQualFile):
    inputVideo = bob.io.video.reader(videoFile)
    
    #load input video
    vin = inputVideo.load()
    numFrames = vin.shape[0]
    
    
    if frameQualFile is not None:
        f = bob.io.base.HDF5File(frameQualFile) #read only
        validFrames = (f.read('/frameQuality')).flatten() #reads list of frame-quality indicators
        validFrames[np.where(validFrames <> 1)]=0
    else:
        validFrames = np.ones(numFrames)
    #print validFrames
#     print type(validFrames)
    numValidFrames = np.sum(validFrames)
    
    print 'valid frames:', numValidFrames, 'of', numFrames
    
    #bob.io.base.save(vin[0,:,:,:].astype('uint8'), '/idiap/temp/sbhatta/msudb_colorImg.png')
    
    import time
    startTime = time.time()
    fset = computeVideoIQA(vin, validFrames)
    print("Time for one video --- %s seconds ---" % (time.time() - startTime))
    
    return fset
    

'''
'''
def parse_arguments(arguments):
        #code for parsing command line args.
    argParser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

#     # verbose
    argParser.add_argument('-v', '--verbose', dest='verbose', metavar='INT', type=int, choices=[0, 1, 2], default=1,
      help='Prints (hopefully helpful) messages  (Default: %(default)s)')
    
    argParser.add_argument('-db', '--dbase_path', dest='dbPath', default = None, #'/idiap/user/sbhatta/work/Antispoofing/ImageQualityMeasures',
       help='path where database videos exist.')

    argParser.add_argument('-op', '--output_path', dest='outPath', default = None,
       help='path where face-files will be stored.')
    
    argParser.add_argument('-nf', '--numFiles', action='store_true', dest='numFiles',
      default=False, help='Option to print no. of files that will be processed. (Default: %(default)s)')
    
    argParser.add_argument('-f', '--singleFile', dest='singleFile', default=None, 
      help='filename (including complete path) of video to be used to test this code: %(default)s)')
    
    argParser.add_argument('-ve', '--video_ext', dest='vidExtn', default=None, choices = ['avi', 'mov', 'mp4'],
      help='filename (including complete path) of video to be used to test this code: %(default)s)')
    
    
    bobdb.Database.create_parser(argParser, implements_any_of='video')
    args = argParser.parse_args(arguments)
    
    database = args.cls(args)
   
    if args.singleFile is None:
        #make sure the user specifies a folder where feature-files exist
        if not args.dbPath: argParser.error('Specify parameter --dbase_path')
    else:
        folder = os.path.dirname(args.singleFile)
        filename = os.path.basename(args.singleFile)
        args.dbPath = folder
        args.singleFile = filename
            
    if not args.outPath: argParser.error('Specify parameter --output_path')
        
    return (args, database)


'''
'''
def main(arguments):
    
    args, database = parse_arguments(arguments)
    
    inpDir = args.dbPath
    outDir = args.outPath
    assert os.path.exists(inpDir), "Input database folder %s does not exist" %inpDir
    if args.verbose>0: print 'Loading data from',inpDir
    
    if args.singleFile is None:
            
        tr_realFiles, tr_attackFiles = database.get_train_data()
        dv_realFiles, dv_attackFiles = database.get_devel_data()
        ts_realFiles, ts_attackFiles = database.get_test_data()
        allFiles = tr_realFiles + dv_realFiles + ts_realFiles + tr_attackFiles + dv_attackFiles + ts_attackFiles
        del tr_realFiles, tr_attackFiles, dv_realFiles, dv_attackFiles, ts_realFiles, ts_attackFiles
        
        numFiles = len(allFiles)
        if args.numFiles:
            print 'Number of files to be processed:',numFiles
            print 'exiting'
            return
        
    #     print numFiles
    #     numFiles = 1        #test
        
        # if we are on a grid environment, just find what I have to process.
        fileSet = allFiles[0:numFiles]
        if os.environ.has_key('SGE_TASK_ID'):
            pos = int(os.environ['SGE_TASK_ID']) - 1
            
            if pos >= numFiles:
                raise RuntimeError, "Grid request for job %d on a setup with %d jobs" % (pos, numFiles)
            fileSet = [allFiles[pos]] # objects = [objects[pos]]

        print 'processing', len(fileSet), ' files'
        k1=0
        for k in fileSet:
            #1. load video file
            print 'filenum:', k1
    #         infile = k.make_path(videoRoot, '.avi')
    #         outfile = k.make_path(featRoot, '.h5')
            print k
            if args.vidExtn is None:
                inVidFile = k.videofile(inpDir)  #k.make_path(inpDir, '.avi')
            else:
                inVidFile = k.make_path(inpDir, ('.' + args.vidExtn))
            inFrameQFile = None #k.make_path(inpDir, '_frameQ.h5')
            outFeatFile = k.make_path(outDir, '.h5')
            head, tail = os.path.split(outFeatFile)
            if not os.path.exists(head): os.makedirs(head)      #create output folder, if it doesn't exist
            
            print inFrameQFile
            print outFeatFile
            
    #         if True: #not os.path.isfile(outFeatFile):
            msuIQAFeats = computeIQA_1video(inVidFile, inFrameQFile)
    
            #4. save features in file 
            ohf = bob.io.base.HDF5File(outFeatFile, 'w')
            ohf.set('msuiqa', msuIQAFeats)
            del ohf
            
    #         assert 0>1, 'stop'
            k1 += 1    
    else:
        # test feature-computation with a single file specified as input
        filePart = os.path.splitext(args.singleFile)[0]
        inVidFile = os.path.join(args.dbPath, filePart)+ '.avi'
        inFrameQFile = os.path.join(args.dbPath, filePart)+ '_frameQ.h5'
        
        outFeatFile = os.path.join(outDir, filePart)+ '.h5'
        head, tail = os.path.split(outFeatFile)
        if not os.path.exists(head): os.makedirs(head)      #create output folder, if it doesn't exist
            
        print 'single file:', inVidFile
        print inFrameQFile
        print outFeatFile
        
        msuIQAFeats = computeIQA_1video(inVidFile, inFrameQFile)
        #4. save features in file 
        ohf = bob.io.base.HDF5File(outFeatFile, 'w')
        ohf.set('msuiqa', msuIQAFeats)
        del ohf

# special fn to extract first frame from video-file and store it as hdf5
def extractTestFrame():
    videoFile = '/idiap/home/sbhatta/work/git/refactoring/bob.db.msu_mfsd_mod/bob/db/msu_mfsd_mod/test_images/real/real_client022_android_SD_scene01.mp4'
    inputVideo = bob.io.video.reader(videoFile)
    
    #load input video
    vin = inputVideo.load()
    numFrames = vin.shape[0]
    outframe = vin[0]
    outfile = '/idiap/home/sbhatta/work/git/refactoring/bob.db.msu_mfsd_mod/bob/db/msu_mfsd_mod/test_images/real_client022_android_SD_scene01_frame0_correct.hdf5'
    ohf = bob.io.base.HDF5File(outfile, 'w')
    ohf.set('color_frame', outframe)
    del ohf

if __name__ == '__main__':
#    extractTestFrame()
    main(sys.argv[1:])
