.. py:currentmodule:: bob.ip.qualitymeasure

.. testsetup:: *

   from __future__ import print_function
   import math
   import os, sys
   import argparse

   import bob.io.image #remove this if possible
   import bob.io.base
   import bob.io.video
   import bob.ip.color
   import numpy as np
   from bob.ip.qualitymeasure import galbally_iqm_features as iqm
   from bob.ip.qualitymeasure import msu_iqa_features as iqa

   import bob.io.base.test_utils #remove this if possible

   import pkg_resources
   video_file = bob.io.base.test_utils.datafile('real_client001_android_SD_scene01.mp4', 'bob.ip.qualitymeasure', 'data')
   video4d = bob.io.video.reader(video_file)

=============
 User Guide
=============

You can used this Bob package to extract image-quality features for face-PAD applications.
Two sets of quality-features are implemented in this package:

1. The image-quality measures proposed by Galbally et al. (IEEE TIP 2014), and

2. The image-quality features proposed by Wen et al. (IEEE TIFS 2015).

The package includes separate modules for implementing the two feature-sets.
The module ``galbally_iqm_features`` implements the features proposed by Gabally et al., and the module ``msu_iqa_features`` implements the features proposed by Wen et al. 
In each module, a single function needs to be called, to retrieve all the features implemented in the module.
The examples below show how to use the functions in the two modules.

Note that both feature-sets are extracted from still-images. However, in face-PAD experiments, we typically process videos.
Therefore, the examples below use a video as input, but show how to extract image-quality features for a single frame.

Note also, that in the examples below, the input to the feature-extraction functions are full-frames. If you wish to extract features only for the face-region, you will have to first construct an image containing only the region of interest, and pass that as the parameter to the feature-extraction functions.


Computing Galbally's image-quality measures
-------------------------------------------
The function ``compute_quality_features()`` (in the module galbally_iqm_features) can be used to compute 18 image-quality measures 
proposed by Galbally et al. Note that Galbally et al. proposed 25 features in their paper. This package implements the following
18 features from their paper, namely: 
[mse, psnr, ad, sc, nk, md, lmse, nae, snrv, ramdv, mas, mams, sme, gme, gpe, ssim, vif, hlfi].
The function ``galbally_iqm_features::compute_quality_features()`` returns a 1D numpy array of length 18, containing the feature-values in the order listed above.

.. doctest::


   >>> from bob.ip.qualitymeasure import galbally_iqm_features as iqm
   >>> video4d = bob.io.video.reader(video_file) # doctest: +SKIP
   >>> rgb_frame = video4d[0]
   >>> print(rgb_frame.shape)
   (3, 480, 720)
   >>> gf_set = iqm.compute_quality_features(rgb_frame)
   >>> print(len(gf_set))
   18

In the example-code above, we have used a color (RGB) image as input to the function ``compute_quality_features()``.
In fact, the features proposed by Galbally et al. are computed over gray-level images.
Therefore, the function ``galbally_iqm_features::compute_quality_features()`` takes as input either a RGB color-image,
or a gray-level image.
(The input image should be a numpy-array. RGB color-images should be in the format expected by Bob_.)
When the input image is 3-dimensional, the first dimension being '3' (as is the case in the example above), the input
is assumed to represent a color RGB image, and is first converted to a gray-level image.
If the input is 2-dimensional (say, a numpy array of shape [480, 720]), then it is assumed to represent a gray-level
image, and the RGB-to-gray conversion step is skipped. 


Computing Wen's (MSU) image-quality measures
--------------------------------------------
The code below shows how to compute the image-quality features proposed by Wen et al. (Here, we refer to these features as
'MSU features'.)
These features are computed from a RGB color-image. The 3 feature-types (specularity, image-blur, color-diversity) all together form
a 121-D feature-vector.
The function ``compute_msu_iqa_features()`` (from the module ``msu_iqa_features``) returns a 1D numpy array of length 121.

.. doctest::

   >>> from bob.ip.qualitymeasure import msu_iqa_features as iqa
   >>> video4d = bob.io.video.reader(video_file) # doctest: +SKIP
   >>> rgb_frame = video4d[0]
   >>> msuf_set = iqa.compute_msu_iqa_features(rgb_frame)
   >>> print(len(msuf_set))
   121


.. _Bob: https://www.idiap.ch/software/bob/ 
.. _documentation: https://menpofit.readthedocs.io/en/stable/
