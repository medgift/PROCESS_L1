"""
    this file has the functions of preprocessing

    otsu_thresholding
    center_of_sliding_level
    connected_component_image

"""
import sys
import numpy as np
#from skimage.filters import threshold_otsu
import cv2,math
from skimage import measure
import itertools
import math
import numpy as np
from scipy import ndimage as ndi
from collections import OrderedDict
from skimage.exposure import histogram
from skimage._shared.utils import assert_nD, warn, deprecated
from skimage.transform import integral_image
from skimage.util import crop, dtype_limits







def threshold_otsu(image, nbins=256):
    """Return threshold value based on Otsu's method.
    Parameters
    ----------
    image : (N, M) ndarray
        Grayscale input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.
    Raises
    ------
    ValueError
         If `image` only contains a single grayscale value.
    References
    ----------
    .. [1] Wikipedia, http://en.wikipedia.org/wiki/Otsu's_Method
    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_otsu(image)
    >>> binary = image <= thresh
    Notes
    -----
    The input image must be grayscale.
    """
    if len(image.shape) > 2 and image.shape[-1] in (3, 4):
        msg = "threshold_otsu is expected to work correctly only for " \
              "grayscale images; image shape {0} looks like an RGB image"
        warn(msg.format(image.shape))

    # Check if the image is multi-colored or not
    if image.min() == image.max():
        raise ValueError("threshold_otsu is expected to work with images "
                         "having more than one color. The input image seems "
                         "to have just one color {0}.".format(image.min()))

    image = np.asarray([x for x in image.ravel() if x != 255])
    image = np.asarray([x for x in image.ravel() if x != 0])
    print(image.ravel())

    hist, bin_centers = histogram(image.ravel(), nbins)
    hist = hist.astype(float)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]

    #threshold = 150
    return threshold






# def threshold_otsu(image, nbins=256):
#     """Return threshold value based on Otsu's method.
#     Parameters
#     ----------
#     image : (N, M) ndarray
#         Grayscale input image.
#     nbins : int, optional
#         Number of bins used to calculate histogram. This value is ignored for
#         integer arrays.
#     Returns
#     -------
#     threshold : float
#         Upper threshold value. All pixels with an intensity higher than
#         this value are assumed to be foreground.
#     Raises
#     ------
#     ValueError
#          If `image` only contains a single grayscale value.
#     References
#     ----------
#     .. [1] Wikipedia, http://en.wikipedia.org/wiki/Otsu's_Method
#     Examples
#     --------
#     >>> from skimage.data import camera
#     >>> image = camera()
#     >>> thresh = threshold_otsu(image)
#     >>> binary = image <= thresh
#     Notes
#     -----
#     The input image must be grayscale.
#     """
#     if len(image.shape) > 2 and image.shape[-1] in (3, 4):
#         msg = "threshold_otsu is expected to work correctly only for " \
#               "grayscale images; image shape {0} looks like an RGB image"
#         warn(msg.format(image.shape))
#
#     # Check if the image is multi-colored or not
#     if image.min() == image.max():
#         raise ValueError("threshold_otsu is expected to work with images "
#                          "having more than one color. The input image seems "
#                          "to have just one color {0}.".format(image.min()))
#
#     image = image.nonzero()
#
#
#     hist, bin_centers = histogram(image.ravel(), nbins)
#     hist = hist.astype(float)
#
#     # class probabilities for all possible thresholds
#     weight1 = np.cumsum(hist)
#     weight2 = np.cumsum(hist[::-1])[::-1]
#     # class means for all possible thresholds
#     mean1 = np.cumsum(hist * bin_centers) / weight1
#     mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
#
#     # Clip ends to align class 1 and class 2 variables:
#     # The last value of `weight1`/`mean1` should pair with zero values in
#     # `weight2`/`mean2`, which do not exist.
#     variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
#
#     idx = np.argmax(variance12)
#     threshold = bin_centers[:-1][idx]
#     return threshold


def otsu_thresholding(im_float):

    """
        apply otsu thresholding on the whole slide

        im_float = input image as float type

        return otsued image with gray, otsu threshold value
    """

    #print("threshold_otsu\n"))
    idx = im_float != 255
    #print(np.shape(im_float))
    new=np.zeros(np.shape(im_float))
    new += 200
    #new=np.zeros(np.shape(rgb_im))
    #new += 254
    #new[:,:, 1] += 223
    #new[:,:, 2] += 228

    new[idx]=im_float[idx]

    threshold_global_Otsu = threshold_otsu(new)
    if threshold_global_Otsu>229:
        threshold_global_Otsu = 196
    # if threshold_global_Otsu<110:
    #     threshold_global_Otsu = 150

    black_mask = im_float == 0
    #thresholding
    im_bool= (im_float > threshold_global_Otsu)
    im_int = im_bool.astype(float) + black_mask #removes also total black parts
    #print(im_int*255)
    return im_int*255, threshold_global_Otsu

def center_of_slide_level(slide,level):

    """
        center x,y point of input level image

        slide =  input slide with Openslide type
        level = disired level

        return  center point x, y
    """

    c_x, c_y = slide.level_dimensions[level]
    c_x/=2
    c_y/=2
    return c_x, c_y

def connected_component_image(otsu_image,component_num):

    """
        apply Connected Component Analysis to otsu_image
        it is because of detect tissue
        choose the label that has largest spces in the image

        otsu_image = input image that applied otsu thresholding
        component_num = threshold number of CCI(connected_component_image)

        max_label = maximum label of components
        cnt_label = the number of pix which in lebel
        argsort_label = sorted index of cnt_label list

        return tissue image
    """

    image_labels = measure.label(otsu_image,background=0)


    max_label = np.max(image_labels)
    cnt_label = []

    print("before change componont num : {0}".format(component_num))
    #component number check ( shold compoenet number < maximum component of image )
    if component_num > max_label :
        component_num = max_label
    print("after change component num : {0}".format(component_num))

    for i in range(1,max_label+1):
        temp = (image_labels == i)
        cnt_nonzero = np.count_nonzero(temp)
        cnt_label.append(cnt_nonzero)

    argsort_label = np.argsort(np.array(cnt_label))

    #tissue image initialize
    tissue_image = np.zeros(image_labels.shape)

    for i in range(component_num):
        temp= (image_labels == argsort_label[i])
        temp= temp.astype(int)
        tissue_image += temp

    tissue_image = tissue_image.astype(float)

    return tissue_image
