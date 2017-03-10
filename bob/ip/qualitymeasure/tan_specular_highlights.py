'''
Created on May 9, 2016

@author: sbhatta

#------------------------------#
   reference: 
   "separating reflection components of textured surfaces using a single image"
   by Robby T. Tan, Katsushi Ikeuchi,
   IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI),
   27(2), pp.179-193, February, 2005
   
   This Python implementation is inspired by the C++ code provided by Prof. Robby Tan:
       http://tanrobby.github.io/code.html#
       http://tanrobby.github.io/code/highlight.zip
    
   The main difference between this implementation and the original C++ implementation is that
   here the init_labels() function also ignores pixels marked G_DIFFUSE. This leads to a smaller
   number of iterations per epsilon value, while producing the same result as the C++ code.
#------------------------------#
'''


import os, sys
import argparse
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import math

from scipy import ndimage
from scipy import misc

import bob.io.image
import bob.io.base

#special-pixel-markers. The values don't matter, but they should all remain different from each other.
G_SPECULARX     = 10
G_SPECULARY     = 11
G_DIFFUSE       = 12
G_BOUNDARY      = 13
G_NOISE         = 14
G_CAMERA_DARK   = 15


'''
main function that processes the input srcImage and returns the separate components.
'''
def remove_highlights(srcImage, startEps=0.5, verboseFlag=True):
    """Returns the separate reflection components (highlights and diffuse components) comprising the input color RGB image.
    
    Input:
        srcImage: numpy array of shape (3, maxY, maxX), containing a RGB image. (maxY, maxX) give the image-size (num. rows, num. cols).
        startEps: floating point (small value), determines the number of iterations of algorithm. epsilon is initialized to this value.
                  At each iteration epsilon is reduced by 0.01, and if it is not less than 0.0, another iteration is performed.
                  Thus, a value of 0.5 leads to 50 iterations. Specify a smaller value if less iterations are desired.
        verboseFlag: flag to indicate whether to print some intermediate values or not. Specify 'False' if you want no message printed.

    Outputs:
        sfi: numpy array of shape (3, maxY, maxX), containing speckle-free image
        diffuse: numpy array of shape (3, maxY, maxX), containing the diffuse component
        speckleResidue: numpy array of shape (3, maxY, maxX), containing specular component.
    """
    
    assert (len(srcImage.shape) == 3 and srcImage.shape[0]==3), "remove_highlights():: input srcImage should be a numpy array of shape (3, maxY, maxX) representing a RGB image"
    srcImage = srcImage.astype(float)
    #initialize resulting diffuse-image
    diffuse = np.copy(srcImage)     #this copy will be updated in zIteration()
    
    srcPixelStatus, sfi, srcMaxChroma, srcMax, srcTotal = specular_free_image(diffuse)
    # sfi is the specular-free image.

    epsilon=startEps
    step=0.01
    
    while(epsilon>=0.0):
        if verboseFlag: print('*')
        diffuse, srcPixelStatus = iteration(diffuse, srcPixelStatus, sfi, epsilon, verboseFlag)
        epsilon -= step
        if verboseFlag: print('ep: %f' % epsilon)

    speckleResidue = srcImage - diffuse
    
    return sfi, diffuse, speckleResidue #speckleResidue is 3D but for every pixel all channels have the same value.

'''
returns the specular-free image corresponding to the input color image 'src'.
Called in zRemoveHighlights()
'''
def specular_free_image(src, srcPixelStatus=None):
    """Generates specular-free version of input RGB color image src.
    Inputs:
        src: numpy array of shape (3, maxY, maxX) containing a RGB image.
        srcPixelStatus: numpy array of shape (maxX, maxY) containing a marker per pixel, indicating the type of pixel.
            This array is updated inside this function, so if the input param. is None, it is first allocated and initialized to 0-values.
    
    Outputs:
        srcPixelStatus: numpy array of shape (maxX, maxY) containing a marker per pixel, indicating the type of pixel.
        sfi: numpy array of shape (3, maxY, maxX) containing the specular-free image corresponding to src.
        srcMaxChroma: numpy array of shape (maxX, maxY) containing the max. chroma for each pixel. 
        srcMax: numpy array of shape (maxX, maxY) containing the max. among the 3 color-intensities, for each pixel in src.
        srcTotal: numpy array of shape (maxX, maxY) containing pixel-wise sum of color-intensities in src.
    """
    
    # allocate output image
    cLambda = 0.6       # arbitrary constant. See paper referenced above.
    camDark = 10.0      # threshold to determine if a pixel is too dark
    lambdaConst = 3.0*cLambda - 1.0
    
#     assert(len(src.shape) == 3 and src.shape[0] == 3), "zSpecularFreeImage():: input src should be a 3D numpy array representing a RGB image (first dim should be 3)"
    
    if srcPixelStatus is None:
        srcPixelStatus = np.zeros((src.shape[1], src.shape[2])) 
    else:
        assert(src.shape[1:-1] == srcPixelStatus.shape), "specular_free_image():: srcPixelStatus should be None, or should match the pixel-layout of src."
    
    for y in range(src.shape[1]):
        for x in range(src.shape[2]):
            if np.all(src[:,y,x]<camDark):
                srcPixelStatus[y,x] = G_CAMERA_DARK

    srcMaxChroma, srcMax, srcTotal = max_chroma(src)
    
    numer = srcMax*(3.0*srcMaxChroma - 1.0)
    denom = srcMaxChroma*lambdaConst
    dI = np.divide(numer, denom)
    sI = (srcTotal - dI)/3.0
    
    drgb = src.astype(float) - sI   #src: 3d array (rgb image); sI: 2D array matching pixel-layout of src
    
    drgb[np.where(np.isinf(drgb))]=0
    drgb[np.where(np.isnan(drgb))]=0
    drgb[np.where(drgb<0)]=0
    drgb[np.where(drgb>255)]=255
    sfi=drgb

    return srcPixelStatus, sfi, srcMaxChroma, srcMax, srcTotal


'''
implements one step of the iterative process to convert specular pixels in src to diffuse.
Called in zRemoveHighlights()
'''
def iteration(src, srcPixelStatus, sfi, epsilon, verboseFlag=True):
    """Iteratively converts each specular-pixel to diffuse.
    Inputs:
        src: numpy array of shape (3, maxY, maxX) containing a RGB image.
        srcPixelStatus: numpy array of shape (maxX, maxY) containing a marker per pixel, indicating the type of pixel.
        sfi: numpy array of shape (3, maxY, maxX) containing the speckle-free image corresponding to src
        epsilon: floating-point (small) value. 
        verboseFlag: indicator to print something in every loop.
    
    Outputs:
        src: numpy array of shape (3, maxY, maxX) containing updated input image
        srcPixelStatus: numpy array of shape (maxX, maxY) containing updated pixel-markers
    """
    
    thR=0.1
    thG=0.1
    pcount=0
#     assert len(src.shape)==3, "zIteration():: input src should be a 3D numpy array representing a RGB image"

    count, srcPixelStatus = init_labels(src, sfi, epsilon, srcPixelStatus)

    while(1):
        srcMaxChroma, srcMax, srcTotal = max_chroma(src)
        srcChroma, srcTotal = rgb_chroma(src, srcTotal)
        maxY = src.shape[1]
        maxX = src.shape[2]
        
        red_chroma_diff_x = np.diff(srcChroma[0,:,:], axis=1)
        red_chroma_diff_y = np.diff(srcChroma[0,:,:], axis=0)
        
        grn_chroma_diff_x = np.diff(srcChroma[1,:,:], axis=1)
        grn_chroma_diff_y = np.diff(srcChroma[1,:,:], axis=0)
    
        if verboseFlag: print('.')
        for y in range(maxY-1):
            for x in range(maxX-1):
                if(srcPixelStatus[y,x] <> G_CAMERA_DARK):
                    
                    drx = red_chroma_diff_x[y,x]
                    dgx = grn_chroma_diff_x[y,x]
                    dry = red_chroma_diff_y[y,x]
                    dgy = grn_chroma_diff_y[y,x]
                    
                    if(srcPixelStatus[y,x] == G_SPECULARX):
                        if((abs(drx) > thR) and (abs(dgx) > thG)):     # if it is  a boundary in the x direction 
                            #pixel right    
                            srcPixelStatus[y,x] = G_BOUNDARY
                            continue
                        elif (abs( srcMaxChroma[y,x]- srcMaxChroma[y,x+1] ) <0.01):      # if it is a noise
                            srcPixelStatus[y,x] = G_NOISE
                            continue
                        else: # reduce the specularity at x direction
                            if (srcMaxChroma[y,x] < srcMaxChroma[y,x+1]):
                                iro=src[:,y,x]
                                pStat = srcPixelStatus[y,x]
                                refMaxChroma = srcMaxChroma[y,x+1]
                                iroTotal = srcTotal[y,x]
                                iroMax = srcMax[y,x]
                                iroMaxChroma = srcMaxChroma[y,x]
                                pStat, iroRes = specular_to_diffuse(iro, iroMax, iroMaxChroma, iroTotal, refMaxChroma) 
                                #zSpecular2Diffuse(src(y,x),src(y,x+1).zMaxChroma())
                                
                                if pStat == G_NOISE:        #update pixelStatus only if zSpecular2Diffuse() returned pStat as G_NOISE.
                                    srcPixelStatus[y,x]=pStat
                                else:
                                    src[:,y,x]=iroRes

                            else:
                                iro=src[:,y,x+1]
                                pStat = srcPixelStatus[y,x+1]
                                refMaxChroma = srcMaxChroma[y,x]
                                iroTotal = srcTotal[y,x+1]
                                iroMax = srcMax[y,x+1]
                                iroMaxChroma = srcMaxChroma[y,x+1]
                                pStat, iroRes = specular_to_diffuse(iro, iroMax, iroMaxChroma, iroTotal, refMaxChroma)     
                                #zSpecular2Diffuse(src(y,x+1),src(y,x).zMaxChroma())
                                
                                if pStat == G_NOISE:        #update pixelStatus only if zSpecular2Diffuse() returned pStat as G_NOISE.
                                    srcPixelStatus[y,x+1]=pStat
                                else:
                                    src[:,y,x+1]=iroRes
                                
                            srcPixelStatus[y,x]   = G_DIFFUSE
                            srcPixelStatus[y,x+1] = G_DIFFUSE
                    
                    if(srcPixelStatus[y,x] == G_SPECULARY):
                        if((abs(dry) > thR) and (abs(dgy) > thG)): # if it is a boundary in the y direction 
                            # pixel right
                            srcPixelStatus[y,x] = G_BOUNDARY
                            continue
                        elif (abs(srcMaxChroma[y,x]-srcMaxChroma[y+1,x])<0.01): # if it is a noise
                            srcPixelStatus[y,x] = G_NOISE
                            continue
                        else:     # reduce the specularity in y direction
                            if(srcMaxChroma[y,x] < srcMaxChroma[y+1,x]):
                                iro=src[:,y,x]
                                pStat = srcPixelStatus[y,x]
                                refMaxChroma = srcMaxChroma[y+1,x]
                                iroTotal = srcTotal[y,x]
                                iroMax = srcMax[y,x]
                                iroMaxChroma = srcMaxChroma[y,x]
                                pStat, iroRes = specular_to_diffuse(iro, iroMax, iroMaxChroma, iroTotal, refMaxChroma)    
                                #C call: zSpecular2Diffuse(src(y,x),src(y+1,x).zMaxChroma())
                                
                                if pStat == G_NOISE:        #update pixelStatus only if zSpecular2Diffuse() returned pStat as G_NOISE.
                                    srcPixelStatus[y,x]=pStat
                                else:
                                    src[:,y,x]=iroRes

                            else:
                                iro=src[:,y+1,x]
                                pStat = srcPixelStatus[y+1,x]
                                pMaxChroma = srcMaxChroma[y+1,x]
                                iroTotal = srcTotal[y+1,x]
                                iroMax = srcMax[y+1,x]
                                iroMaxChroma = srcMaxChroma[y+1,x]
                                pStat, iroRes = specular_to_diffuse(iro, iroMax, iroMaxChroma, iroTotal, pMaxChroma)
                                #zSpecular2Diffuse(src(y+1,x),src(y,x).max_chroma())
                                
                                if pStat == G_NOISE:        #update pixelStatus only if zSpecular2Diffuse() returned pStat as G_NOISE.
                                    srcPixelStatus[y+1,x]=pStat
                                else:
                                    src[:,y+1,x]=iroRes
                                    
                            srcPixelStatus[y,x] = G_DIFFUSE
                            srcPixelStatus[y+1,x] = G_DIFFUSE
                            
    
        pcount=count
        count, srcPixelStatus = init_labels(src, sfi, epsilon, srcPixelStatus)

        if(count==0 or pcount<=count): #Robby Tan's original C++ code checks if count<0, but that doesn't make sense as count cannot be negative.
            break       # break out of the while-loop
    
    
    srcPixelStatus = reset_labels(srcPixelStatus)
    
    return src, srcPixelStatus


'''
initializes the labels at the beginning and end of each iteration to compute diffuse pixels.
Called in zIteration()
'''
def init_labels(src, sfi, epsilon, srcPixelStatus):
    """Generates initial labels for all pixels in src (input RGB image).
    Inputs:
        src: numpy array of shape (3, maxY, maxX) containing a RGB image.
        sfi: numpy array of shape (3, maxY, maxX) containing a speckle-free image corresponding to src.
        epsilon: positive floating point (small) value
        srcPixelStatus: numpy array of shape (maxX, maxY) containing pixel-markers corresponding to src.
        
    Returns:
         count: number of pixels marked as specular in src.
         srcPixelStatus: numpy array of shape (maxX, maxY) containing updated pixel-markers corresponding to src. 
    """
    
    # to have initial labels  
    count=0
    zTotalSrc = np.sum(src, axis=0)
    diff_x_src = np.diff(zTotalSrc, axis=1)
    diff_y_src = np.diff(zTotalSrc, axis=0)

    zTotalSfi = np.sum(sfi, axis=0)
    diff_x_sfi = np.diff(zTotalSfi, axis=1)
    diff_y_sfi = np.diff(zTotalSfi, axis=0)

    dlog_src_x = np.log(abs(diff_x_src))
    dlog_src_y = np.log(abs(diff_y_src))
    
    dlog_sfi_x = np.log(abs(diff_x_sfi))
    dlog_sfi_y = np.log(abs(diff_y_sfi))
    
    dlogx = abs(dlog_src_x - dlog_sfi_x)
    dlogy = abs(dlog_src_y - dlog_sfi_y)

    maxY = src.shape[1]
    maxX = src.shape[2]

    for y in range(1, maxY-1):
        for x in range(1, maxX-1):
            pStat = srcPixelStatus[y,x]
#             if pStat <> G_BOUNDARY and pStat <> G_NOISE and pStat <> G_CAMERA_DARK and pStat <> G_DIFFUSE:
            if pStat not in (G_BOUNDARY, G_NOISE, G_CAMERA_DARK, G_DIFFUSE):
                #Robby Tan's original C++ code doesn't check for pStat<>G_DIFFUSE, but this speeds up the processing a lot, and doesn't seem to affect the results.
                if   dlogx[y,x] > epsilon:
                    pStat = G_SPECULARX
                    count+=1
                elif dlogy[y,x] > epsilon:
                    pStat = G_SPECULARY
                    count+=1
                else:
                    pStat = G_DIFFUSE
                    
            srcPixelStatus[y,x]=pStat

    return count, srcPixelStatus    # count is the number of specular pixels


'''
Converts a single specular pixel to diffuse.
Called in zIteration()
'''
def specular_to_diffuse(iro, iroMax, iroMaxChroma, iroTotal, refMaxChroma):
    """Converts a color-pixel from speckled to diffuse, by subtracting a certain amount from its intensity value in each color channel.
    Inputs:
        iro: 3-element column vector containing the rgb-color values of the pixel to be processed.
        iroMax: max value among the elements of iro
        iroMaxChroma: max-chroma for this pixel
        iroTotal: sum of the 3 elements of iro.
        refMaxChroma: chroma-value of a neighboring pixel for comparison
    Returns:
        pixelStatus: a value (marker) indicating whether the pixel is considered as noise or diffuse, after processing.
        iro: updated pixel-color-values.
    """
    c=iroMaxChroma
    pixelStatus = 0 #arbitrary initialization
    numr = (iroMax*(3.0*c - 1.0))
    denm = (c*(3.0*refMaxChroma - 1.0))
    if abs(denm)> 0.000000001:   #to avoid div. by zero.
        dI = numr / denm
        sI = (iroTotal - dI)/3.0
        nrgb = iro - sI
            
        if np.any(nrgb<0):
            pixelStatus = G_NOISE
        else:
            iro = nrgb
    else:
        pixelStatus = G_NOISE

    return pixelStatus, iro

    
'''
reset all pixelStatus labels, except for DARK pixels
src is numpy-array of shape (3,maxY, maxX)
pixelLabels is a numpy-array of shape (maxY, maxX)
Called in zIteration().
'''
def reset_labels(pixelLabels):
    """Resets all pixel-markers (except those indicating "dark" pixels, to 0.
    Input:
        pixelLabels: numpy array of shape (maxX, maxY) containing pixel-markers.
    
    Returns:
        pixelLabels: numpy array of shape (maxX, maxY) containing updated pixel-markers..
    """
    # to reset the label of the pixels
    pixelLabels[np.where(pixelLabels <> G_CAMERA_DARK)] = 0

    return pixelLabels


#####
#functions from globals
#####

'''
returns the max of r,g,b values for a given pixel
'''
def max_color(rgbImg):
    """For input RGB image, this function computes max intensity among all color-channels, for each pixel position.
    
    Input:
        rgbImg: numpy array of shape (3, maxY, maxX) containing a RGB image.
        
    Returns:
        rgbMax: (2D numpy array of shape (maxY, maxY)) contains pixel-wise max. among the three color values.
    """
    
    return np.amax(rgbImg, axis=0)

# '''
# returns min. of r,g,b values for a pixel
# '''
# def zMin(rgbImg):
#     return np.amin(rgbImg, axis=0)

'''
returns total of r,g,b values for pixel
'''
def total_color(rgbImg):
    """For input RGB image, this function computes sum of intensities in each color-channel for each pixel position.
    Input:
        rgbImg: numpy array of shape (3, maxY, maxX) containing a RGB image.
        
    Returns:
        rgbTotal: (2D numpy array of shape (maxY, maxY)) contains pixel-wise sum of the 3 color-values.
    """
    
    return np.sum(rgbImg.astype(float), axis=0)

'''
returns max_chroma of input image rgbImg
'''
def max_chroma(rgbImg, rgbMax=None, rgbTotal=None):
    """Given an input RGB image (rgbImg, with shape (3, maxY, maxX)), this function computes the maximum chroma-value for each pixel position.
    
    Inputs:
        rgbImg: numpy array of shape (3, maxY, maxX) containing a RGB image.
        rgbMax: numpy array of shape (maxY, maxX) containing pixel-wise max. color intensity. If None, it is computed.
        rgbTotal: numpy array of shape (maxY, maxX) containing pixel-wise sum of color-intensities. If None, it is computed.
    
    Returns:
        rgbChroma: (3D numpy array of same shape as rgbImg) contains the chroma-values of the individual channels in rgbChroma, and 
        rgbMax: (2D numpy array of shape (maxY, maxY)) contains pixel-wise max. among the three color values.
        rgbTotal: (2D numpy array of shape (maxY, maxY)) contains pixel-wise sum of the 3 color-values.
    """
    
    if rgbMax is None:
        rgbMax = max_color(rgbImg)
    if rgbTotal is None:
        rgbTotal = total_color(rgbImg)
    
    #srcChroma = np.zeros((src.shape[1], src.shape[2]))
    maxChroma = np.divide(rgbMax.astype(float), rgbTotal.astype(float))
    maxChroma[np.where(rgbTotal==0)]=0
    
    return maxChroma, rgbMax, rgbTotal #max_chroma(rgbImg, rgbMax=None, rgbTotal=None)

'''
computes the 3 chroma values for each pixel in input image.
'''
def rgb_chroma(rgbImg, rgbTotal=None):
    """Given an input RGB color image, compute the chroma-values in the 3 channels.
    
    Inputs:
        rgbImg: numpy array of shape (3, maxY, maxX) containing a RGB image.
        rgbTotal: numpy array of shape (maxY, maxX) containing pixel-wise sum of color-intensities. If None, it is computed.
        
    Returns:
        rgbChroma: (3D numpy array of same shape as rgbImg) contains the chroma-values of the individual channels in rgbChroma, and 
        rgbTotal: (2D numpy array of shape (maxY, maxY)) contains pixel-wise sum of the 3 color-values.
    """
    
    if rgbTotal is None:
        rgbTotal = total_color(rgbImg)
    
    rgbChroma = np.divide(rgbImg.astype(float), rgbTotal.astype(float))
    for p in range(rgbChroma.shape[0]):
        rgbChroma[p, (rgbTotal==0)]=0

    return rgbChroma, rgbTotal


'''
utility function to display image on screen (for debugging).
'''
def imshow(image):
    import matplotlib
    from matplotlib import pyplot as plt
    if len(image.shape)==3:
        #imshow() expects color image in a slightly different format, so first rearrange the 3d data for imshow...
        outImg = image.tolist()
#         print(len(outImg))
        result = np.dstack((outImg[0], outImg[1]))
        outImg = np.dstack((result, outImg[2]))
        plt.imshow((outImg*255.0).astype(np.uint8)) #[:,:,1], cmap=mpl.cm.gray)
         
    else:
        if(len(image.shape)==2):
            #display gray image.
            plt.imshow(image.astype(np.uint8), cmap=matplotlib.cm.gray)
        else:
            print("inshow():: image should be either 2d or 3d.")
             
    plt.show()
    

'''
'''
def computeIQSpecularityFeatures(rgbImage, startEps=0.05):
    
    speckleFreeImg, diffuseImg, speckleImg = remove_highlights(rgbImage, startEps=0.05)
    
    if len(speckleImg.shape)==3:
        speckleImg = speckleImg[0]
    
    speckleImg = speckleImg.clip(min=0)

    speckleMean = np.mean(speckleImg)
    lowSpeckleThresh = speckleMean*1.5
    hiSpeckleThresh = speckleMean*4.0
    print(speckleMean, lowSpeckleThresh, hiSpeckleThresh)
    specklePixels = speckleImg[np.where(np.logical_and(speckleImg > lowSpeckleThresh, speckleImg<hiSpeckleThresh))] #(speckleImg > lowSpeckleThresh and speckleImg<hiSpeckleThresh) #[np.where(lowSpeckleThresh < speckleImg and speckleImg<hiSpeckleThresh)]
    
    r = float(specklePixels.shape[0])/(speckleImg.shape[0]*speckleImg.shape[1])
    m = np.mean(specklePixels)
    s =  np.std(specklePixels)
    
    return (r,m/150.0,s/150.0)

'''
utility function to test the implementation. Relies on bob to load and save images.
'''
def test_highlight_detection():
    inputImageFile = '/idiap/home/sbhatta/work/Antispoofing/ImageQualityMeasures/specular_highlights/highlight_C_Code/images/head.ppm'
    outRoot = '/idiap/home/sbhatta/work/Antispoofing/ImageQualityMeasures/specular_highlights/'
#     inputImageFile = 'C:/IDIAP/AntiSpoofing/ImageQualityMeasures/highlight/images/head.ppm'
#     outRoot = 'C:/IDIAP/AntiSpoofing/ImageQualityMeasures/'
    
    #load color image
    inpImage = bob.io.base.load(inputImageFile)
#     inpImage = misc.imread(inputImageFile)
#     inpImage =  np.rollaxis(inpImage,2)

    speckleFreeImg, diffuseImg, speckleImg = remove_highlights(inpImage, startEps=0.05)
     
    print('saving output images')
    bob.io.base.save(speckleFreeImg.astype('uint8'), outRoot+'speckleFreeHeadImage.ppm')
#     speckleFreeImg = np.rollaxis(np.rollaxis(speckleFreeImg, 0,-1),2,1)
#     misc.imsave(outRoot+'speckleFreeHeadImage.ppm', speckleFreeImg.astype('uint8'))
     
    bob.io.base.save(diffuseImg.astype('uint8'), outRoot+'diffuseImgHeadImage.ppm')
#     diffuseImg = np.rollaxis(np.rollaxis(diffuseImg, 0,-1),2,1)
#     misc.imsave(outRoot+'diffuseImgHeadImage.ppm', diffuseImg.astype('uint8'))
 
    bob.io.base.save(speckleImg.astype('uint8'), outRoot+'speckleImgHeadImage.ppm')
#     speckleImg = np.rollaxis(np.rollaxis(speckleImg, 0,-1),2,1)
#     misc.imsave(outRoot+'speckleImgHeadImage.ppm', speckleImg.astype('uint8'))
#     
    
#     r,m,s=computeIQSpecularityFeatures(inpImage, startEps=0.05)
#     print(r, m, s)

#    
#    imshow(inpImage)
#    imshow(speckleFreeImg)
#    imshow(diffuseImg)
    imshow(speckleImg)

'''
main entry point.
'''
if __name__ == '__main__':
    test_highlight_detection()
