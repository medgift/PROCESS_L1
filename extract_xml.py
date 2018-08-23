from xml.etree.ElementTree import parse
from os import listdir
from os.path import join, isfile, exists, splitext
import numpy as np
#from openslide import OpenSlide
import collections

def make_list_of_contour_from_xml(fn_xml,downsample):
    """
        make the list of contour from xml(annotation file)
        input:
        fn_xml = file name of xml file
        downsample = disired resolution

        var:
        li_li_point = list of tumors
        li_point = the coordinates([x,y]) of a tumor

        return  list of list (2D array list)
    """
    li_li_point = []
    tree = parse(fn_xml)
    for parent in tree.getiterator():
        for i_1, child1 in enumerate(parent):
            for i_2, child2 in enumerate(child1):
                for i_3, child3 in enumerate(child2):
                    li_point=[]
                    for i_4, child4 in enumerate(child3):
                        x_0 = float(child4.attrib['X'])
                        y_0 = float(child4.attrib['Y'])
                        x_s = x_0/downsample
                        y_s = y_0/downsample
                        li_point.append([x_s,y_s])
                    if len(li_point):
                        li_li_point.append(li_point)
    return li_li_point

def convert_list_of_contour_2_opencv_contours(li_li_point):
    """
        conver list of contour(2D list array) to opencv contours
        that list of nparray (not 2-d nparray !)

        input:
        li_li_point = list of contours

        var:
        countours = list of contours
        contour = nparray with x,y coordinate

        return opencv contours
    """
    contours=[]
    for li_point in li_li_point:
        li_point_int = [[int(round(point[0])), int(round(point[1]))] for point in li_point]
        contour = np.array(li_point_int,dtype = np.int32)
        contours.append(contour)
    return contours

def get_opencv_contours_from_xml(fn_xml,downsample):
    """"
        get opencv contours( list of nparrays) from xml annotation file

        input:
        fn_xml = xml file name
        downsample = disired downsample

        return list of contours
    """
    li_li_point = make_list_of_contour_from_xml(fn_xml,downsample)
    l_contours = convert_list_of_contour_2_opencv_contours(li_li_point)
    return l_contours

def convert_contour_coordinate_resolution(l_contours,downsample):
    """
        convert  contours coordinate to downsample resolution

        input:
        l_contours : list of contours coordinate(x,y) in level 0 resolution
        downsample : disired resolution

        return coverted contour list
    """
    cvted_l_contours =[]
    for contour in l_contours:
        print ('shape: ', contour.shape)
        downsample_coor = contour / downsample
        downsample_coor = (downsample_coor).astype(int)
        downsample_coor = np.unique(downsample_coor,axis=0)
        cvted_l_contours.append(downsample_coor)
    return cvted_l_contours

def get_mask_from_opencv_contours(l_contours,slide,level) :
    """
        get binary image map in certain level(resolution)

        input:
        l_contour = list of nparray that contain coordinate(x,y)
        slide = to obtain dimension of mask
        level = desired level

        return tumor mask image (binary image 0-1)
    """
    slid_lev_w, slid_lev_h = slide.level_dimensions[level]
    mask_image = np.zeros((slid_lev_h,slid_lev_w))
    print("mask_image dimension : ", mask_image.shape)
    downsample = slide.level_downsamples[level]
    print("downsample: {0}".format(downsample))
    #convert coordinate to the level resolution from level=0
    for i,npary in enumerate(l_contours):
        print("tummor number : {0}".format(i))
        #check 1 in the tummor region
        li_xy = npary.flatten()
        import pdb; pdb.set_trace()
        d_x, d_y = li_xy[::2],li_xy[1::2]
        mask_image[d_x,d_y] = 255.0
        print("put {0} tummor".format(i))
    return mask_image
