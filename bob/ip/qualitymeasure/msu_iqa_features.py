'''
Created on 9 Feb 2016

@author: sbhatta
'''



#import re
#import os
import math

import numpy as np
import scipy as sp
import scipy.signal as ssg
import scipy.ndimage.filters as snf
import galbally_iqm_features as iqm
import bob.ip.base
import bob.ip.color

########## Utility functions ###########
'''
Matlab-like RGB to gray...
    @param: rgbImage : numpy array for the form: [3,h,w] where h is the height of the image and w is the width of the image.
    Returns a y-image in floating-point format (range [(16/255) .. (235/255)])
'''
def matlab_rgb2gray(rgbImage):
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

'''
'''
def matlab_rgb2hsv(rgbImage):
    # first normalize the range of values to 0-1
   
    isUint8 = True
    if isUint8: rgbImage = rgbImage.astype(np.float64)/255.0
        
    hsv = np.zeros_like(rgbImage)
    bob.ip.color.rgb_to_hsv(rgbImage, hsv)
    h = hsv[0,:,:]
    s = hsv[1,:,:]
    v = hsv[2,:,:]
#     
    return (h, s, v)


def imshow(image):
    import matplotlib
    from matplotlib import pyplot as plt
    if len(image.shape)==3:
        #imshow() expects color image in a slightly different format, so first rearrange the 3d data for imshow...
        outImg = image.tolist()
        print len(outImg)
        result = np.dstack((outImg[0], outImg[1]))
        outImg = np.dstack((result, outImg[2]))
        plt.imshow((outImg*255.0).astype(np.uint8)) #[:,:,1], cmap=mpl.cm.gray)
         
    else:
        if(len(image.shape)==2):
            #display gray image.
            plt.imshow(image.astype(np.uint8), cmap=matplotlib.cm.gray)
             
    plt.show()
    

########### End of Utilities ##############

########### Auxilliary functions  ##############

"""
"""
def sobelEdgeMap(image, orientation='both'):
    
    #bob..sobel returns filter-responses which need to be thresholded to get the edge-map
    thinning=1
    refImage=image.astype(np.float)
    
    #compute edge map for reference image
    refSobel_sep = bob.ip.base.sobel(refImage) #returns 3D image. 1st dim is the edge-direction. 1st component is vertical; 2nd component is hor. responses
    refSobelX = refSobel_sep[0,:,:]
    refSobelY = refSobel_sep[1,:,:]
    if orientation is 'horizontal':
        refEdge = iqm.edgeThinning(refSobelX[:,:], refSobelX[:,:], thinning)
    else:
        if orientation is 'vertical':
            refEdge = iqm.edgeThinning(refSobelY[:,:], refSobelY[:,:], thinning)
        else:
            refEdge = iqm.edgeThinning(refSobelX[:,:], refSobelY[:,:], thinning)
            

    return refEdge

########### End of Aux. functions ##############

'''
'''
#def computeMsuIQAFeatures(rgbImage, printFV=False):
def computeMsuIQAFeatures(rgbImage):
    assert len(rgbImage.shape)==3, 'computeMsuIQAFeatures():: image should be a 3D array (containing a rgb image)'
#     hsv = np.zeros_like(rgbImage)
#     bob.ip.color.rgb_to_hsv(rgbImage, hsv)
#     h = hsv[0,:,:]
#     s = hsv[1,:,:]
#     v = hsv[2,:,:]
    h,s,v = matlab_rgb2hsv(rgbImage) #defined above. Calls Bob's rgb_to_hsv() after rescaling the input image.
    
    #print "computeMsuIQAFeatures():: check bob.ip.color.rgb_to_hsv conversion"
    grayImage = np.zeros_like(h, dtype='uint8')
    bob.ip.color.rgb_to_gray(rgbImage, grayImage)
    
    blurFeat = blurriness(grayImage)
#     print 'blurriness:', blurFeat
    
    pinaBlur = marzilianoBlur(grayImage)
    pinaBlur /= 30.0
#     print 'pinaBlur:',pinaBlur
    
    colorHist, totNumColors = calColorHist(rgbImage)
    totNumColors /= 2000.0 #as done in Matlab code provided by MSU.
#     print "color hist shape:", colorHist.shape
#     print colorHist[0:11]
#     print 'totNumColors', totNumColors
    
    # calculate mean, deviation and skewness of each channel
    # use histogram shifting for the hue channel
    #print h.shape
    momentFeatsH = calmoment_shift(h)
    #print 'H-moments:', momentFeatsH
        
    momentFeats = momentFeatsH.copy()
    momentFeatsS = calmoment(s)
    #print 'S-moments:', momentFeatsS
    momentFeats = np.hstack((momentFeats, momentFeatsS))
    momentFeatsV = calmoment(v)
    #print 'V-moments:', momentFeatsV
    momentFeats = np.hstack((momentFeats, momentFeatsV))
    
    fv = momentFeats.copy()
    #print 'moment features:', fv
    
    fv = np.hstack((fv, colorHist))
    fv = np.hstack((fv, totNumColors))
    fv = np.hstack((fv, blurFeat))
    fv = np.hstack((fv, pinaBlur))

    return fv


"""
Implements the method proposed by Marziliano et al. for determining the average width of vertical edges, as a measure of blurredness in an image.
This function is a Python version of the Matlab code provided by MSU.

:param image: 2D gray-level (face) image
:param regionMask: (optional) 2D matrix (binary image), where 1s mark the pixels belonging to a region of interest, and 0s indicate pixels outside ROI. 
"""
def marzilianoBlur(image):
    assert len(image.shape)==2, 'marzilianoBlur():: input image should be a 2D array (gray level image)'
       
    edgeMap = sobelEdgeMap(image, 'vertical')        # compute vertical edge-map of image using sobel 
    
    #There will be some difference between the result of this function and the Matlab version, because the
    #edgeMap produced by sobelEdgeMap() is not exactly the same as that produced by Matlab's edge() function.
    
#     Test edge-map generated in Matlab produces the same result as the matlab version of MarzilianoBlur().
#     edgeMap = bob.io.base.load('/idiap/temp/sbhatta/msudb_faceEdgeMap.png')
#     imshow(edgeMap)
    
    blurImg = image
    C = blurImg.shape[1]    #number of cols in image
    (row, col) = edgeMap.nonzero()  # row, col contain the indices of the pixels that comprise edge-map.
    
    blurMetric = 0
    if len(row) > 0:

        #to make the following code work in a similar fashion to the original matlab code, sort the cols in ascending order, and sort the rows according to the cols.
    #     ind = np.lexsort((row,col))
    #     row = row[ind]
    #     col = col[ind]
        #print 'lexsort_col:', 1+col
        #print 'lexsort_row:', 1+row
        #This was only used for debugging (to compare with Matlab code). In fact it is not necessary, so it is commented out.
    
        
        edgeWidths = np.zeros_like(row, dtype=int)
        
        firstRow = row[0]
    #     print 'firstRow:',firstRow
        for i in range(len(row)):
            rEdge = row[i]
            cEdge = col[i]
    #         if rEdge == firstRow: print "edgePoints:", (i, rEdge, cEdge)
            
            cStart = 0          # instead of setting them to 'inf' as in MSU's Matlab version
            cEnd = 0       
            
            #we want to estimate the edge-width, which will be cEnd - cStart.
            
            #search for start of edge in horizontal direction
            if cEdge > 0: #i.e., edge is not on the left-border
                #2.1: search left of current pixel (backwards)
                if blurImg[rEdge, cEdge] > blurImg[rEdge, cEdge-1]: #edge corresponds to a local peak; estimate left-extent of peak
                    j=cEdge-1
                    while j>0 and blurImg[rEdge, j] >= blurImg[rEdge, j-1]: j -= 1
                    cStart = j
                else:   #edge corresponds to a local valley; determine left-extent of valley
                    j=cEdge-1
                    while j>0 and blurImg[rEdge, j] <= blurImg[rEdge, j-1]: j-= 1
                    cStart = j
            
            #search for end of edge in horizontal direction        
            cEnd = C-1 #initialize to right-border of image -- the max. possible position for cEnd
            if cEdge < C-1:
                if blurImg[rEdge, cEdge] > blurImg[rEdge, cEdge+1]: #edge corresponds to a local peak; estimate right-extent of peak
                    j=cEdge+1
                    while j< C-1 and blurImg[rEdge, j] >= blurImg[rEdge, j+1]: j += 1
                    cEnd = j
                else:   #edge corresponds to a local valley; determine right-extent of valley
                    j=cEdge+1
                    while j< C-1 and blurImg[rEdge, j] <= blurImg[rEdge, j+1]: j += 1
                    cEnd = j
            
            edgeWidths[i] = cEnd - cStart
            
            #sanity-check (edgeWidths should not have negative values)
            negSum = np.sum( edgeWidths[ np.where(edgeWidths<0) ] )
            assert negSum==0, 'marzilianoBlur():: edgeWidths[] contains negative values. YOU CANNOT BE SERIOUS!'
            
        # Final metric computation
        blurMetric = np.mean(edgeWidths)
            
    #compute histogram of edgeWidths ...(later)
                                # binnum = 100;
                                # t = ((1:binnum) - .5) .* C ./ binnum;
                                # whist = hist(width_array, t) ./ length(width_array);

    return blurMetric


"""
    returns the first 3 statistical moments (mean, standard-dev., skewness) and 2 other first-order statistical measures of input image 
    :param channel: 2D array containing gray-image-like data
"""
def calmoment( channel, regionMask=None ):
    assert len(channel.shape) == 2, 'calmoment():: channel should be a 2D array (a single color-channel)'    

    t = np.arange(0.05, 1.05, 0.05) + 0.025                 # t = 0.05:0.05:1;
#     t = np.arange(0.05, 1.05, 0.05) + 0.025                 # t = 0.05:0.05:1;
#     np.insert(t, 0, -np.inf)
#     t[-1]= np.inf
#     print type(t)
#     print t

    nPix = np.prod(channel.shape)                   # pixnum = length(channel(:));
    m = np.mean(channel)                            # m = mean(channel(:));
    d = np.std(channel)                             # d = sqrt(sum((channel(:) - m) .^ 2) / pixnum);
    s = np.sum(np.power( ((channel - m)/d), 3))/nPix  # s = sum(((channel(:) - m) ./ d) .^ 3) / pixnum;
    #print 't:', t
    myHH = np.histogram(channel, t)[0]
    #print myHH
    hh = myHH.astype(float)/nPix              # hh = hist(channel(:),t) / pixnum;
    #print 'numPix:', nPix
    #print 'histogram:',hh
    
    #H = np.array([m,d,s, np.sum(hh[0:1]), np.sum(hh[-2:-1])])  # H = [m d s sum(hh(1:2)) sum(hh(end-1:end))];
    H= np.array([m,d,s])
    s0 = np.sum(hh[0:2])
    #print s0
    H = np.hstack((H,s0))
    s1 = np.sum(hh[-2:])
    #print s1
    H = np.hstack((H, s1) )
    #print 'calmoment:',H.shape
    
    return H


'''
'''
def calmoment_shift( channel ):
    assert len(channel.shape) == 2, 'calmoment_shift():: channel should be a 2D array (a single color-channel)'
    
    channel = channel + 0.5;
                                                    # tag = find(channel>1);
    channel[np.where(channel>1.0)] -= 1.0           # channel(tag) = channel(tag) - 1;

#     t = np.arange(0.05, 1.05, 0.05)                 # t = 0.05:0.05:1;
#     nPix = np.prod(channel.shape)                   # pixnum = length(channel(:));
                                                    #     m = mean(channel(:));
                                                    #     d = sqrt(sum((channel(:) - m) .^ 2) / pixnum);
                                                    #     s = sum(((channel(:) - m) ./ d) .^ 3) / pixnum;
                                                    #     hh = hist(channel(:),t) / pixnum;
                                                    #     
                                                    #     H = [m d s sum(hh(1:2)) sum(hh(end-1:end))];
    
    H = calmoment(channel)
    
    return H



"""
function returns the top 'm' most popular colors in the input image
    :param image: RGB color-image represented in a 3D array
    :param m: integer specifying how many 'top' colors to be counted (e.g. for m=10 the function will return the pixel-counts for the top 10 most popular colors in image)
    :return cHist: counts of the top 100 most popular colors in image
    :return numClrs: total number of distinct colors in image 
"""
def calColorHist(image, m=100):
    #1. compute color histogram of image (normalized, if specified)
    numBins = 32
    maxval=255
    #print "calColorHist():: ", image.shape
    cHist = rgbhist(image, maxval, numBins, 1)

    #2. determine top 100 colors of image from histogram
    #cHist.sort() cHist = cHist[::-1]
    y = sorted(cHist, reverse=True) # [Y, I] = sort(H,'descend');
    cHist=y[0:m]                    # H = Y(1:m)';

    c = np.cumsum(y)                # C = cumsum(Y);
#     print 'cumsum shape:', c.shape
#     thresholdedC = np.where(c>0.999)
#    # print thresholdedC.shape
#     print 'thresholdedC:', thresholdedC[0][0] #:200]
    numClrs = np.where(c>0.99)[0][0]    # clrnum = find(C>.99,1,'first') - 1;
    
    cHist = np.array(cHist)
    return cHist, numClrs


'''
computes 3d color histogram of image
'''
def rgbhist(image, maxval, nBins, normType=0):
    assert len(image.shape)==3, 'image should be a 3D (rgb) array of shape (3, m,n) where m is no. of rows, and n is no. if cols in image.c$'
    assert normType >-1 and normType<2, 'rgbhist():: normType should be only 0 or 1'
    H = np.zeros((nBins, nBins, nBins), dtype=np.uint32)  # zeros([nBins nBins nBins]);

#     testImage = image[:,0:3,0:3].copy()
#     print testImage.shape
#     print image.shape
#     print testImage[0,:,:]
#     print ''
#     print testImage[1,:,:]
#     print ''
#     print testImage[2,:,:]
#     print ''
#     print testImage.reshape(3, 9, order='C').T
#     
#     assert(0>1), "Stop!"
    
    decimator = (maxval+1)/nBins
    numPix = image.shape[1]*image.shape[2]
    im = image.reshape(3, numPix).copy() # im = reshape(I,[size(I,1)*size(I,2) 3]);
    im=im.T
    
    for i in range(0, numPix):          # for i=1:size(I,1)*size(I,2)
        p = (im[i,:]).astype(float)     # p = double(im(i,:));
        p = np.floor(p/decimator)       # p = floor(p/(maxval/nBins))+1;
        H[p[0], p[1], p[2]] += 1        # H(p(1),p(2),p(3)) = H(p(1),p(2),p(3)) + 1;
                                        # end
    #totalNBins = np.prod(H.shape)
    #H = H.reshape(totalNBins, 1, order='F') same as H = reshape(H, nBins**3, 1)
    H = H.ravel()        #H = H(:);
#     print 'H type:',type(H[0])
#     print H.shape
    # Un-Normalized histogram
    
    if normType ==1: H =  H.astype(np.float32) / np.sum(H)    # l1 normalization
#     else:
#         if normType==2:
#             H = normc(H);            # l2 normalization
    
    return H


"""
function to estimate blurriness of an image, as computed by Di Wen et al. in their IEEE-TIFS-2015 paper.
    :param image: a gray-level image
"""
def blurriness(image):
    
    assert len(image.shape) == 2, 'Input to blurriness() function should be a 2D (gray) image'
    
    d=4
    fsize = 2*d + 1
    kver = np.ones((1, fsize))/fsize
    khor = kver.T
    
    Bver = ssg.convolve2d(image.astype(np.float32), kver.astype(np.float32), mode='same');
    Bhor = ssg.convolve2d(image.astype(np.float32), khor.astype(np.float32), mode='same');
    
    #DFver = np.abs(np.diff(image.astype('int32'), axis=0)) # abs(img(2:end,:) - img(1:end-1,:));
    #DFhor = np.abs(np.diff(image.astype('int32'), axis=1)) #abs(img(:,2:end) - img(:,1:end-1));
    # implementations of DFver and DFhor below don't look the same as in the Matlab code, but the following implementation produces equivalent results.
    # there might be a bug in Matlab!
    #The 2 commented statements above would correspond to the intent of the Matlab code.
    DFver = np.diff(image.astype('int16'), axis=0)
    DFver[np.where(DFver<0)]=0

    DFhor = np.diff(image.astype('int16'), axis=1)
    DFhor[np.where(DFhor<0)]=0

    DBver = np.abs(np.diff(Bver, axis=0)) # abs(Bver(2:end,:) - Bver(1:end-1,:));
    DBhor = np.abs(np.diff(Bhor, axis=1)) #abs(Bhor(:,2:end) - Bhor(:,1:end-1));

    Vver = DFver.astype(float) - DBver.astype(float)
    Vhor = DFhor.astype(float) - DBhor.astype(float)
    Vver[Vver<0]=0 #Vver(find(Vver<0)) = 0;
    Vhor[Vhor<0]=0 #Vhor(find(Vhor<0)) = 0;

    SFver = np.sum(DFver)
    SFhor = np.sum(DFhor) #sum(DFhor(:));

    SVver = np.sum(Vver) #sum(Vver(:));
    SVhor = np.sum(Vhor) #sum(Vhor(:));

    BFver = (SFver - SVver) / SFver;
    BFhor = (SFhor - SVhor) / SFhor;

    blurF = max(BFver, BFhor) #max([BFver BFhor]);
    
    return blurF


