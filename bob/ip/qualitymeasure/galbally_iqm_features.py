#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

'''
Created on 25 Sep 2015

@author: sbhatta
'''


#import re
#import os
import math

import numpy as np
import scipy as sp
import scipy.signal as ssg
import scipy.ndimage.filters as snf

import bob.ip.base



"""
Main function to be called, to extract a set of image quality-features computed for the input image
:param image: 2d numpy array. Should contain input image of size [M,N] (i.e. M rows x N cols).
:return featSet: a tuple of float-scalars, each representing one image-quality measure.
"""
def compute_quality_features(image):
    """Extract a set of image quality-features computed for the input image.
    :param image: 2d or 3d numpy array. Should represent input image of shape [M,N] (M rows x N cols).
    If 2D, image should contain a gray-image of shape [M,N].
    If 3d, image should have a shape [3,M,N], and should contain an RGB-image.

    :return featSet: a tuple of float-scalars, each representing one image-quality measure.
    This function returns a subset of the image-quality features (for face anti-spoofing) that have been 
    described by Galbally et al. in their paper:
    "Image Quality Assessment for Fake Biometric Detection: Application to Iris, Fingerprint, and Face Recognition",
     IEEE Trans. on Image Processing Vol 23(2), 2014.
    """
    gray_image = None
    #print("shape of input image:")
    #print(image.shape)
    if len(image.shape) == 3:
        if(image.shape[0]==3): 
            gray_image = matlab_rgb2gray(image) #compute gray-level image for input color-frame
            print(gray_image.shape)
        else:
            print('error. Wrong kind of input image')
    else:
        if len(image.shape) == 2:
            gray_image = image
            print(gray_image.shape)
        else:
            print('error -- wrong kind of input')

    if gray_image is not None: 

        gwin = gauss_2d((3,3), 0.5) # set up the smoothing-filter
        print("computing degraded version of image")
        smoothed = ssg.convolve2d(gray_image, gwin, boundary='symm', mode='same') 
    
        """
        Some of the image-quality measures computed here require a reference image.
        For these measures, we use the input-image itself as a reference-image, and we compute
        the quality-measure of a smoothed version of the input-image. The assumption in this
        approach is that smoothing degrades a spoof-image more than it does a genuine image
        (see Galbally's paper referenced above).
        """
        print("computing galbally quality features")
        featSet = image_quality_measures(gray_image, smoothed)
    
        return featSet

    else:
        return None



"""
actually computes various measures of similarity between the two input images, but also returns some descriptors of the reference-image that are independent of any other image.
Returns a tuple of 18 values, each of which is a float-scalar. 
The quality measures computed in this function correspond to the Image-quality features discussed in Galbally et al., 2014.
"""
def image_quality_measures(refImage, testImage):
    """Compute image-quality measures for testImage and return a tuple of quality-measures.
       Some of the quality-measures require a reference-image, but others are 'no-reference' measures.
       :input refImage: 2d numpy array. Should represent input 8-bit gray-level image of size [M,N].
       :input testImage: 2d numpy array. Should represent input 8-bit gray-level image of size [M,N]..
       :return a tuple of 18 values, each of which is a float-scalar. 
       The quality measures computed in this function correspond to the Image-quality features discussed in Galbally et al., 2014.
    """
    assert len(refImage.shape)==2, "refImage should be a 2D array"
    assert len(testImage.shape)==2, "testImage should be a 2D array"
    assert (refImage.shape[0] == testImage.shape[0]), "The two images should have the same width"
    assert (refImage.shape[1] == testImage.shape[1]), "The two images should have the same height"
    
    diffImg = refImage.astype(np.float) - testImage.astype(np.float) 
    diffSq = np.square(diffImg)
    sumDiffSq = np.sum(diffSq)
    absDiffImg = np.absolute(diffImg)
    
    refSq = np.square(refImage.astype(np.float))
    sumRefSq = np.sum(refSq)
    
    numPx = refImage.shape[0]*refImage.shape[1] #number of pixels in each image
    maxPxVal = 255.0;
    
    #1 MSE
    mse00 = float(sumDiffSq)/float(numPx)
    
    #2 PSNR
    psnr01 = np.inf
    if mse00>0:
        psnr01 = 10.0*np.log10(maxPxVal*maxPxVal/mse00)
    
    #3 AD: Average difference
    ad02 = float(np.sum(diffImg))/float(numPx)
    
    #4 SC: structural content
    testSq = np.square(testImage.astype(np.float))
    sumTestSq = np.sum(testSq)
    sc03=np.inf
    if sumTestSq>0:
        sc03 = float(sumRefSq)/float(sumTestSq)
    
    #5 NK: normalized cross-correlation
    imgProd = refImage * testImage # element-wise product
    nk04 = float(np.sum(imgProd))/float(sumRefSq)
    
    #6 MD: Maximum difference
    md05 = float(np.amax(absDiffImg))
    
    #7 LMSE: Laplacian MSE
    #scipy implementation of laplacian is different from Matlab's version, especially at the image-borders
    # To significant differences between scipy...laplace and Matlab's del2() are:
    #    a. Matlab del2() divides the convolution result by 4, so the ratio (scipy.laplace() result)/(del2()-result) is 4
    #    b. Matlab does a different kind of processing at the boundaries, so the results at the boundaries are different in the 2 calls.
    #In Galbally's Matlab code, there is a factor of 4, which I have dropped (no difference in result), 
    #because this is implicit in scipy.ndimage.filters.laplace()
    op = snf.laplace(refImage, mode='reflect') #mode can be 'wrap', 'reflect', 'nearest', 'mirror', or ['constant' with a specified value]
    opSq = np.square(op)
    sum_opSq = np.sum(opSq)
    tmp1 = (op - (snf.laplace(testImage, mode='reflect')))
    num_op = np.square(tmp1)
    lmse06 = float(np.sum(num_op))/float(sum_opSq)
    
    #8 NAE: normalized abs. error
    sumRef = np.sum(np.absolute(refImage))
    nae07 = float(np.sum(absDiffImg))/float(sumRef)
    
    #9 SNRv: SNR in db
    snrv08 = 10.0*np.log10(float(sumRefSq)/float(sumDiffSq))
    
    #10 RAMDv: R-averaged max diff (r=10)
    #implementation below is same as what Galbally does in Matlab
    r=10
    sorted = np.sort(diffImg.flatten())[::-1] #the [::-1] flips the sorted vector, so that it is in descending order
    topsum = np.sum(sorted[0:r])
    ramdv09 = np.sqrt(float(topsum)/float(r))
    
    #11,12: MAS: Mean Angle Similarity, MAMS: Mean Angle-Magnitude Similarity
    mas10, mams11 = angle_similarity(refImage, testImage, diffImg) 
    
    fftRef = np.fft.fft2(refImage)
#     fftTest = np.fft.fft2(testImage)
      
    #13, 14: SME: spectral magnitude error; SPE: spectral phase error
    sme12, spe13 = spectral_similarity(refImage, testImage) #spectralSimilarity(fftRef, fftTest, numPx)
    
    #15 TED: Total edge difference
    ted14 = edge_similarity(refImage, testImage)

    #16 TCD: Total corner difference
    tcd15 = corner_similarity(refImage, testImage)
    
    #17, 18: GME: gradient-magnitude error; GPE: gradient phase error  
    gme16, gpe17 = gradient_similarity(refImage, testImage)
    
    #19 SSIM
    ssim18, _ = ssim(refImage, testImage)
    
    #20 VIF
    vif19 = vif(refImage, testImage)
    
    #21,22,23,24,25: RRED, BIQI, JQI, NIQE: these parameters are not computed here.
    
    #26 HLFI: high-low frequency index (implemented as done by Galbally in Matlab).
    hlfi25=high_low_freq_index(fftRef, refImage.shape[1])
    
    return np.asarray((mse00, psnr01, ad02, sc03, nk04, md05, lmse06, nae07, snrv08, ramdv09, mas10, mams11, sme12, gme16, gpe17, ssim18, vif19, hlfi25), dtype=np.float32)


"""
Matlab-like RGB to gray...
"""
def matlab_rgb2gray(rgbImage): 
    '''converts color rgbImage to gray to produce exactly the same result as Matlab would.
    Inputs:
    rgbimage: numpy array of shape [3, height, width]
    Return:
    numpy array of shape [height, width] containing a gray-image with floating-point pixel values, in the range[(16.0/255) .. (235.0/255)]
    '''
    #g1 = 0.299*rgbFrame[0,:,:] + 0.587*rgbFrame[1,:,:] + 0.114*rgbFrame[2,:,:] #standard coeffs CCIR601
    #this is how it's done in matlab...
    rgbImage = rgbImage / 255.0
    C0 = 65.481/255.0
    C1 = 128.553/255.0
    C2 = 24.966/255.0
    scaleMin = 16.0/255.0
    #scaleMax = 235.0/255.0
    gray = scaleMin + (C0*rgbImage[0,:,:] + C1*rgbImage[1,:,:] + C2*rgbImage[2,:,:])
    return gray


"""
SSIM: Structural Similarity index between two gray-level images. The dynamic range is assumed to be 0..255.
Ref:Z. Wang, A.C. Bovik, H.R. Sheikh and E.P. Simoncelli: 
    "Image Quality Assessment: From error measurement to Structural Similarity"
    IEEE Trans. on Image Processing, 13(1), 01/2004
    @param refImage: 2D numpy array (reference image)
    @param testImage: 2D numpy array (test image)
    Both input images should have the same dimensions. This is assumed, and not verified in this function
    @return ssim: float-scalar. The mean structural similarity between the 2 input images.
    @return ssim_map: the SSIM index map of the test image (this map is smaller than the test image).
"""
def ssim(refImage, testImage):
    """Compute and return SSIM between two images.
    @param refImage: 2D numpy array (reference image)
    @param testImage: 2D numpy array (test image)
    Returns ssim and ssim_map
     @return ssim: float-scalar. The mean structural similarity between the 2 input images.
    @return ssim_map: the SSIM index map of the test image (this map is smaller than the test image).
    """
    M=refImage.shape[0]
    N=refImage.shape[1]
        
    winSz=11 #window size for gaussian filter
    winSgm = 1.5 # sigma for gaussian filter
    
    #input image should be at least 11x11 in size.
    if(M<winSz) or (N<winSz):
        ssim_index = -np.inf
        ssim_map = -np.inf
        
        return ssim_index, ssim_map
    
    #construct the gaussian filter
    gwin = gauss_2d((winSz, winSz), winSgm)
    
    K1 = 0.01 # constants taken from the initial matlab implementation provided by Bovik's lab.
    K2 = 0.03
    L = 255 #dynamic range.
    
    C1 = (K1*L)*(K1*L)
    C2 = (K2*L)*(K2*L)
    #refImage=refImage.astype(np.float)
    #testImage=testImage.astype(np.float)
    
    #ssg is scipy.signal
    mu1 = ssg.convolve2d(refImage, gwin, mode='valid')
    mu2 = ssg.convolve2d(testImage, gwin, mode='valid')
    
    mu1Sq = mu1*mu1
    mu2Sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    
    sigma1_sq = ssg.convolve2d((refImage*refImage), gwin, mode='valid') - mu1Sq
    sigma2_sq = ssg.convolve2d((testImage*testImage), gwin, mode='valid') - mu1Sq
    sigma12 = ssg.convolve2d((refImage*testImage), gwin, mode='valid') - mu1_mu2
    
    assert (C1>0 and C2 > 0), "Conditions for computing ssim with this code are not met. Set the Ks and L to values > 0."
    num1 = (2.0*mu1_mu2 + C1) *(2.0*sigma12 + C2) 
    den1 = (mu1Sq + mu2Sq+C1)*(sigma1_sq + sigma2_sq +C2)
    ssim_map = num1/den1

    ssim = np.average(ssim_map)
    
    return ssim, ssim_map


"""
VIF: Visual Information Fidelity measure.
Ref: H.R. Sheikh and A.C. Bovik: "Image Information and Visual Quality", IEEE Trans. Image Processing.
Adapted from Galbally's matlab code, which was provided by Bovik et al's LIVE lab.
    @param refImage: 2D numpy array (reference image)
    @param testImage: 2D numpy array (test image)
    Both input images should have the same dimensions. This is assumed, and not verified in this function
    @return vifp: float-scalar. Measure of visual information fidelity between the 2 input images
"""
def vif(refImage, testImage):
    sigma_nsq = 2.0
    num=0
    den=0
    
    #sc is scale, taking values (1,2,3,4)
    for sc in range(1,5):
        N=(2**(4-sc+1))+1
        #print N, sc
        win = gauss_2d((N,N), (float(N)/5.0))
        
        #ssg is scipy.signal
        
        if sc > 1 :
            refImage = ssg.convolve2d(refImage, win, mode='valid')
            testImage = ssg.convolve2d(testImage, win, mode='valid')
            refImage = refImage[::2, ::2]           #downsample by factor 2 in each direction
            testImage = testImage[::2, ::2]
        
        mu1 = ssg.convolve2d(refImage, win, mode='valid')
        mu2 = ssg.convolve2d(testImage, win, mode='valid')
        mu1Sq = mu1*mu1
        mu2Sq = mu2*mu2
        mu1_mu2 = mu1*mu2
        
        sigma1_sq = ssg.convolve2d((refImage*refImage), win, mode='valid') - mu1Sq
        sigma2_sq = ssg.convolve2d((testImage*testImage), win, mode='valid') - mu2Sq
        sigma12 = ssg.convolve2d((refImage*testImage), win, mode='valid') - mu1_mu2
        
        sigma1_sq[sigma1_sq<0]=0        #set negative filter responses to 0.
        sigma2_sq[sigma2_sq<0]=0
        
        g = sigma12 / (sigma1_sq + 1e-10)
        sv_sq = sigma2_sq - g*sigma12;
        
        g[(sigma1_sq < 1e-10)]=0
        sv_sq[sigma1_sq < 1e-10] = sigma2_sq[sigma1_sq < 1e-10]
        sigma1_sq[sigma1_sq<1e-10]=0
        
        g[(sigma2_sq < 1e-10)]=0
        sv_sq[sigma2_sq < 1e-10] =0
        
        sv_sq[g<0]=sigma2_sq[g<0]
        g[g<0]=0
        sv_sq[sv_sq <= 1e-10] = 1e-10     #sic. As implemented in the original matlab version...
        
        m1 = g*g*sigma1_sq
        m2 = sv_sq+sigma_nsq
        m3 = np.log10(1 + m1/m2)
        
        m4 = np.log10(1 + (sigma1_sq/sigma_nsq))
        
        num += np.sum(m3)
        den += np.sum(m4)
    
    vifp = num/den
    return vifp


"""
HLFI: relative difference between high- and low-frequency energy in image.
Ref: I. Avcibas, N. Memon, B. Sankur: "Steganalysis using image quality metrics", IEEE Trans. Image Processing, 12, 2003.
@param imgFFT: 2D numpy array of complex numbers, representing Fourier transform of test image.
@param ncols: int. Number of columns in image.
@return float-scalar.
"""
def high_low_freq_index(imgFFT, ncols):

    N= ncols
    colHalf = int(round(N/2)) # (N/2) + (N % 2) #round it up
    freqSel = 0.15
    
    freqCol = round(freqSel*N)
    lowFreqColHalf = int(round(freqCol/2.0)) 
    
    fftRes = imgFFT #np.fft.fft2(image)
    fftMag = np.abs(fftRes)
    totalEnergy = np.sum(fftMag)
    #print totalEnergy
    
    lowIdx = colHalf-lowFreqColHalf
    hiIdx = colHalf + lowFreqColHalf
    #print lowIdx, hiIdx
    LowFreqMag = fftMag[:, lowIdx:hiIdx]
    lowFreqMagTotal = np.sum(LowFreqMag)
    
    fftMag[:, lowIdx:hiIdx]=0
    highFreqMagTotal = np.sum(fftMag)
    #print 'partial freq. sums:', lowFreqMagTotal, highFreqMagTotal
    
    highLowFreqIQ = np.abs(lowFreqMagTotal - highFreqMagTotal)/float(totalEnergy)
    
    return highLowFreqIQ
    
        
     

"""
Image similarity based on gradient. Computes the mean phase and magnitude difference of gradient between input reference and test images.
Ref: I. Avcibas, N. Memon, B. Sankur: "Steganalysis using image quality metrics", IEEE Trans. Image Processing, 12, 2003.
    @param refImage: 2D numpy array (reference image)
    @param testImage: 2D numpy array (test image)
    Both input images should have the same dimensions. This is assumed, and not verified in this function.
    @return difGradMag: float-scalar. Mean difference in gradient-magnitude.
    @return difGradPhase: float-scalar. Mean difference in gradient-phase.
"""
def gradient_similarity(refImage, testImage):
    
    numPx = refImage.shape[0]*refImage.shape[1] # we assume that testImage is of the same shape as refImage
    
    #compute gradient (a la matlab) for reference image
    refGrad=np.gradient(refImage,5,5) #5: spacing of 5 pixels between 2 sites of grad. evaluation.
    
    refReal = refGrad[0]
    refImag = refGrad[1]
    refGradComplex = refReal + 1j*refImag
    
    refMag=np.abs(refGradComplex)
    refPhase = np.arctan2(refImag, refReal)

    #compute gradient for test image
    testGrad=np.gradient(testImage,5) #5: spacing of 5 pixels between 2 sites of grad. evaluation. It applies to both dims.
    testReal = testGrad[0]
    testImag = testGrad[1]
    testGradComplex = testReal + 1j*testImag
    
    testMag=np.abs(testGradComplex)
    testPhase = np.arctan2(testImag, testReal)    
    
    absPhaseDiff = np.abs(refPhase - testPhase) 
    difGradPhase = (np.sum(absPhaseDiff))/float(numPx)
    
    absMagDiff = np.abs(refMag - testMag) 
    difGradMag = float(np.sum(absMagDiff))/float(numPx)
    
    return difGradMag, difGradPhase

'''
'''
def testRegionalMax():
    A = 10*np.ones([10,10])
    A[1:4, 1:4] = 22
    A[5:8, 5:8] = 33
    A[1,6]=44
    A[2,7]=45
    A[3,8]=44
    rm = regionalmax(A)
    print A
    print rm

"""
find local maxima using 3x3 mask.
Used in corner_similarity()
Should produce results very similar to matlabs imregionalmax()
@param img: 2d numpy array. Image-like, containing a 'cornerness'-index for every pixel.
@return regmax: 2d numpy array. Binary image showing corners (which are regions of local maxima in input image).
"""
def regionalmax(img):
    h = img.shape[0]
    w = img.shape[1]
    
    #extend input image borders by repeating border-values
    b = np.zeros((img.shape[0]+2,img.shape[1]+2))
    b[1:-1,1:-1] = img
    b[0,:]=b[1,:]
    b[:,0]=b[:,1]
    b[-1,:]=b[-2,:]
    b[:,-1]=b[:,-2]
    
    
    regmax = np.zeros((h,w), dtype='uint8') #will contain the output bitmap showing local maxima.
    
    for i in range(1, h+1):
        for j in range(1, w+1):
            subim = b[i-1:i+2, j-1:j+2]
            lmax = np.amax(subim)
            lmin = np.amin(subim)
            if b[i,j] == lmax and b[i,j]>lmin : 
                regmax[i-1,j-1] = 1
                        
    for i in range(1,h):
        for j in range(w):
            if regmax[i,j]==1:
                imin=i-1
                if imin<0: imin=0
                imax=i+2
                if imax>h: imax=h
                for k in range(imin, imax):
                    jmin=j-1
                    if jmin<0: jmin=0
                    jmax = j+2
                    if jmax>w: jmax = w
                    for l in range(jmin, jmax):
                        if(img[k,l]==img[i,j]):
                            regmax[k,l]=1
    
    return regmax



"""
returns a 'cornerness' image, where each pixel-value specifies the 'degree of cornerness' of the corresponding pixel in input image
The 'cornerness' image is of size (N-2, M-2) for an input image of size (N,M) (no cornerness computed for the border pixel of input.)
@param image: 2d numpy array. Input image for which cornerness needs to be computed.
@return cornerness: 2d numpy array giving a 'cornerness'-value for the input image.
"""
def cornerMetric(image):
    image = image.astype(np.float)
   
    sensitivity_factor = 0.4
    gwin = gauss_2d((5,5), 1.5)
    
    vfilt = np.array([-1,0,1], ndmin=2)
    hfilt = vfilt.T
    A = ssg.convolve2d(image, vfilt, boundary='symm', mode='same') 
    B = ssg.convolve2d(image, hfilt, boundary='symm', mode='same')
    #crop out the valid portions of the filter-response (same size for both A and B)
    A = A[1:-2, 1:-2]
    B = B[1:-2, 1:-2]
    
    #compute products of A, B, C
    C = A*B
    A = A*A
    B = B*B
    
    #filter A, B, and C
    A = ssg.convolve2d(A, gwin, boundary='symm', mode='valid')
    B = ssg.convolve2d(B, gwin, boundary='symm', mode='valid')
    C = ssg.convolve2d(C, gwin, boundary='symm', mode='valid')

    ABsum = A + B
    cornerness = (A*B) - (C*C) - sensitivity_factor *(ABsum*ABsum)
    
    return cornerness
    

# 
# def imshow(image):
#     import matplotlib
#     from matplotlib import pyplot as plt
#     if len(image.shape)==3:
#         #imshow() expects color image in a slightly different format, so first rearrange the 3d data for imshow...
#         outImg = image.tolist()
#         print len(outImg)
#         result = np.dstack((outImg[0], outImg[1]))
#         outImg = np.dstack((result, outImg[2]))
#         plt.imshow((outImg*255.0).astype(np.uint8)) #[:,:,1], cmap=mpl.cm.gray)
#         
#     else:
#         if(len(image.shape)==2):
#             #display gray image.
#             plt.imshow(image.astype(np.uint8), cmap=matplotlib.cm.gray)
#             
#     plt.show()

'''
compute the corner-based similarity between 2 images (how close are the numbers of corners found in the two images?).
returns an index between 0 and 1. The smaller the better.
@param refImage: 2D numpy array (reference image)
@param testImage: 2D numpy array (test image)
@return float-scalar.
'''
def corner_similarity(refImage, testImage):
    
    C = cornerMetric(refImage)
    C_peaks = regionalmax(C)
    
    #imshow(C_peaks)
    
    CG = cornerMetric(testImage)
    CG_peaks = regionalmax(CG)
    
    nCornersRef = np.sum(C_peaks)
    nCornersTest = np.sum(CG_peaks)
    #print 'CornerSim::', nCornersRef, nCornersTest
    
    maxCorners = max(nCornersRef, nCornersTest)
    
    qCornersDiff = np.fabs(nCornersRef - nCornersTest)/float(maxCorners)
    
    
    return qCornersDiff

"""
Similarity between the edge-maps of the two input images.
Ref: I. Avcibas, N. Memon, B. Sankur: "Steganalysis using image quality metrics", IEEE Trans. Image Processing, 12, 2003.
@param refImage: 2D numpy array (reference image)
@param testImage: 2D numpy array (test image)
@return float-scalar
"""
def edge_similarity(refImage, testImage):
    
    #bob..sobel returns filter-responses which need to be thresholded to get the edge-map
    thinning=1
    refImage=refImage.astype(np.float)
    
    #compute edge map for reference image
    refSobel_sep = bob.ip.base.sobel(refImage) #returns 3D image. 1st dim is the edge-direction. 1st component is vertical; 2nd component is hor. responses
    refSobelX = refSobel_sep[0,:,:]
    refSobelY = refSobel_sep[1,:,:]
    refEdge = edge_thinning(refSobelX[:,:], refSobelY[:,:], thinning)
    
    
    #compute edge map for test image
    testSobel_sep = bob.ip.base.sobel(testImage)
    testSobelX = testSobel_sep[0,:,:]
    testSobelY = testSobel_sep[1,:,:]
#     testSobelX = ssg.convolve2d(testImage, x_mask, boundary='symm', mode='valid')
#     testSobelY = ssg.convolve2d(testImage, y_mask, boundary='symm', mode='valid')
    testEdge = edge_thinning(testSobelX[:,:], testSobelY[:,:], thinning)

    numPx = refImage.shape[0]*refImage.shape[1]
    numRefEdgePx = np.sum(refEdge)
    numTestEdgePx = np.sum(testEdge)
    qEdgeD = np.abs(numRefEdgePx - numTestEdgePx)/float(numPx)

    return qEdgeD

"""
    function to perform edge-thining in the same way as done in Matlab. Called in edge_similarity()
    Returns a binary edge-map (uint8 image).
    @param  bx: vertical edge-filter responses (for example, response of 1 of the two Sobel filters)
    @param  by: horizontal edge-filter responses 
    @param  thinning: [0,1]. Default:1, implies 'do edge-thinning'. If set to 0, no edge-thinning is done.
    bx and by should be of the same shape
"""
def edge_thinning(bx, by, thinning=1):
    assert(len(bx.shape)==2) and (len(by.shape)==2), "bx and by should be 2D arrays."
    assert(bx.shape[0]==by.shape[0]) and (bx.shape[1]==by.shape[1]), "bx and by should have the same shape."
    m = bx.shape[0]
    n = by.shape[1]
    e = np.zeros([m,n], dtype=np.uint8)     # will contain the resulting edge-map.
    
#     print 'bx', bx.shape
#     print 'by', by.shape
    #compute the edge-strength from the 2 directional filter-responses
    b = np.sqrt(bx*bx + by*by)
    
    #compute the threshold a la Matlab (as described in "Digital Image Processing" book by W.K. Pratt.
    scale=4
    cutoff = scale*np.mean(b)
    #thresh = np.sqrt(cutoff)
    
    myEps = np.spacing(1)*100.0 #np.spacing(1) is the same as eps in matlab.
    #compute the edge-map a la Matlab
    for r in range(m):
        for c in range(n):
            if thinning:
                if r<0 or r>(m-1) or (c-1)<0 :
                    b1=True
                else:
                    b1=(b[r,c-1] < b[r,c])
                
                if(r<0) or  r>(m-1)  or  (c+1)>(n-1) :
                    b2=True
                else:
                    b2 = (b[r,c] > b[r,c+1])
                
                if (c<0) or  c>(n-1)  or  (r-1)<0 :
                    b3=True
                else:
                    b3=(b[r,c] > b[r-1,c])
                
                if(c<1) or  c>(n-1)  or  (r+1)>(m-1) :
                    b4=True
                else:
                    b4=(b[r,c]> b[r+1, c])
                
                c1 = (b[r,c]>cutoff)
                c2 = ((bx[r,c]>= (by[r,c]-myEps)) & b1 & b2)
                c3 = ((by[r,c]>= (bx[r,c]-myEps)) & b3 & b4)

                e[r,c] = c1 & (c2 | c3)
            else:
                e[r,c] = (b[r,c]>cutoff)
    
    return e


"""
@param refImage: 2D numpy array (reference image)
@param testImage: 2D numpy array (test image)
@return sme: float-scalar. Mean difference in magnitudes of spectra of the two images.
@return spe: float-scalar. Mean difference in phases of spectra of the two images.
"""
#def spectralSimilarity(fftRef, fftTest, numPx):
def spectral_similarity(refImage, testImage):
    
    #assume that ref and test images have the same shape
    rows=refImage.shape[0]
    cols=refImage.shape[1]
    numPx = rows*cols
    fftRef = np.fft.rfft2(refImage)
    fftTest = np.fft.rfft2(testImage)
    
    refMag=np.abs(fftRef)
    testMag=np.abs(fftTest)
    absMagDiff = np.abs(refMag - testMag)
    #SME: spectral magnitude error
    sme = np.sum(absMagDiff * absMagDiff)/float(numPx)
    
    #SPE: spectral phase error
    refPhase = np.angle(fftRef)
    testPhase = np.angle(fftTest)
    absPhaseDiff = np.abs(refPhase - testPhase)
    spe = np.sum(absPhaseDiff * absPhaseDiff)/float(numPx)

    return sme, spe


"""
Cosine-Similarity between the the rows of the two input images.
Ref: I. Avcibas, N. Memon, B. Sankur: "Steganalysis using image quality metrics", IEEE Trans. Image Processing, 12, 2003.
@param refImage: 2D numpy array (reference image)
@param testImage: 2D numpy array (test image)
@param diffImage: 2D numpy array. Difference between refImage and testImage. Not strictly necessary as input but precomputed here, to save computation time.
@return mas: float-scalar. Mean angle-similarity.
@return mams: float-scalar. Mean angle-magnitude similarity.
"""
def angle_similarity(refImage, testImage, diffImage):
    mas=None
    mams=None
    
    numPx = refImage.shape[0]*refImage.shape[1]
    
    refNorm = np.linalg.norm(refImage, axis=1)
    testNorm = np.linalg.norm(testImage, axis=1)
    thetaVec = np.zeros([refImage.shape[0], 1])
    diffNorm = np.linalg.norm(diffImage, axis=1)
    magnitVec = diffNorm/255.0 #Galbally divides by sqrt(255**2)
    magnitVec = np.reshape(magnitVec, (refImage.shape[0], 1))
    
    for i in range(refImage.shape[0]):
        refR = refImage[i,:]
        testR = testImage[i,:]
        cosTheta = np.dot(refR, testR)/(refNorm[i]*testNorm[i])
        if(cosTheta < -1.0): cosTheta = -1.0
        if(cosTheta > 1.0): cosTheta = 1.0
        theta = np.arccos(cosTheta)
        thetaVec[i]=theta
    
    #the following (commented out) code should be more efficient than the for-loop above, but seems to be buggy.
#     rowDP= np.diag(np.dot(refImage,testImage.T))
#     normRefrows= np.linalg.norm(refImage,axis=1)
#     normTestrows= np.linalg.norm(testImage,axis=1)
#     
#     cosThetaVec = rowDP/(normRefrows * normTestrows)
#     cosThetaVec = np.nan_to_num(cosThetaVec) #nan occurs when one of the norms is 0, ie. vector is all 0s. In that case set cosTheta to 0
#     
#     thetaVec = np.arccos(cosThetaVec)
#     
    
    tmp2 = thetaVec*2.0/np.pi
    
    #MAS: mean angle similarity
    mas = 1.0 -( sum(tmp2) / float(numPx) )
    
    tmp3 = 1.0 - tmp2 
    tmp4 = 1.0 - magnitVec    
    chi = 1.0 - (tmp3 * tmp4)
    #MAMS: mean angle-magnitude similarity 
    mams = sum(chi)/float(numPx)

    return (float(mas), float(mams))



'''
Returns a 2D gaussian-filter matrix
equivalent of matlab fspecial('gaussian'...)

Works correctly.
@param shape: tuple defining the size of the desired filter in each dimension. Elements of tuple should be 2 positive odd-integers.
@param sigma: float-scalar
@return h: 2D numpy array. Contains weights for 2D gaussian filter.
'''
def gauss_2d(shape=(3, 3), sigma=0.5): 
    """ 
    2D gaussian mask - should give the same result as MATLAB's 
    fspecial('gaussian',[shape],[sigma]) 
    """ 
    m, n = [(ss-1.)/2. for ss in shape] 
    y, x = np.ogrid[-m:m+1, -n:n+1] 
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma)) 
    h[h < np.finfo(h.dtype).eps*h.max()] = 0 
    sumh = h.sum() 
    if sumh != 0: 
        h /= sumh 
    return h 


'''
returns a smoothed version of input gray-level image
smoothing is done using a fixed 3x3 gaussian filter (sigma=0.5)
'''
def gaussianSmooth(image):
    # perform Gaussian filtering
    gaussian = bob.ip.base.Gaussian(sigma = (0.5, 0.5), radius = (1, 1)) #radius=1 creates a 3x3 filter.
    return gaussian(image)



# '''
# '''
# def testQualityMeasures(image, smoothed):
#     frameFeatSet = imageQualityMeasures(image, smoothed)
#     
#     print frameFeatSet
#     
#     
# if __name__ == '__main__':
#     image, smoothed = testGaussianFilter()
#     
#     #testQualityMeasures(image, smoothed)
#     diffImg = image - smoothed
#     #test MAMS
#     mas, mams = angle_similarity2(image, smoothed, diffImg)
#     print 'mas', mas
#     print 'mams', mams


    
    
    
