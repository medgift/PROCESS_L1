import cv2
import numpy as np
from PIL import Image
from skimage.transform.integral import integral_image, integrate
from random import randint

def is_white_patch(cur_patch,white_percentage):
    ''' Basic is_white check: checks if the extracted patch is white
        and returns True if so

    input:
    cur_patch, patch to check
    white_percentage, white portion threshold

    output:
    True if percentage of white> white portion threshold
    False otherwise
    '''
    #good buttt slowww
    # half black and half white patches are still kept. not a good thing.
    #print('into is_white')
    is_white = True
    total_white = float(cur_patch.shape[0] *cur_patch.shape[1] * cur_patch.shape[2] * 255)
    if (cur_patch.sum()/total_white)>white_percentage:
        return is_white
    else:
        return not is_white

def patch_sampling_using_integral(slide,slide_level,mask,patch_size,patch_num):

    """
    patch sampling on whole slide image

    input:

    slide = OpenSlide Object
    slide_level = level of mask
    mask = mask image ( 0-1 int type nd-array)
    patch_size = size of patch scala integer n
    patch_num = the number of output patches

    output:

    list of patches(RGB Image), list of patch point (starting from left top)

    """
    patch_list = []  # patches
    patch_point = [] # patch locations
    # taking the nonzero points in the mask
    x_l,y_l = mask.nonzero()
    #slide_level=7
    if len(x_l) > patch_size/slide.level_downsamples[slide_level]*2:
        level_patch_size = int(patch_size/slide.level_downsamples[slide_level])
        print('DEBUGGG: ', slide_level)
        # computing the actual level of resolution
        # applying the nonzero mask as a dot product
        x_ws = (np.round(x_l*slide.level_downsamples[slide_level])).astype(int)
        y_ws = (np.round(y_l*slide.level_downsamples[slide_level])).astype(int)
        cnt = 0 # patch counter
        nt_cnt = 1 # not taken counter
        white_threshold = .3
        #white_threshold = 1.0
        while(cnt < patch_num) :
            
            # sampling from random distribution
            p_idx = randint(0,len(x_l)-1)
            # picking the random point in the mask
            level_point_x,level_point_y = x_l[p_idx], y_l[p_idx]
            if (level_point_y < 50) or (level_point_x < 250): ##new add to check
                continue
            # check the boundary to make patch
            check_bound = np.resize(np.array([level_point_x+level_patch_size,level_point_y+level_patch_size]),(2,))
            if check_bound[0] > mask.shape[0] or check_bound[1] > mask.shape[1]:
                continue
            # make patch from mask image
            level_patch_mask = mask[int(level_point_x):int(level_point_x+level_patch_size),int(level_point_y):int(level_point_y+level_patch_size)]

            # apply integral
            ii_map = integral_image(level_patch_mask)
            ii_sum = integrate(ii_map,(0,0),(level_patch_size-1,level_patch_size-1))
            area_percent = float(ii_sum)/(level_patch_size**2)
            # checking if the total area of the patch covers at least 80% of
            # the annotation region
            
            if area_percent<0.6:
                continue

            if cnt > patch_num*10+1000:
                print("There is no more patches to extract in this slide")
                print("mask region is too small")
                print("final number of patches : ",len(patch_list))

                break

            patch=slide.read_region((y_ws[p_idx],x_ws[p_idx]),0,(patch_size,patch_size))
            patch = np.array(patch)
            #print('[integral] np.sum(patch): ', np.sum(patch))
            
            if np.sum(patch)==0:
                print('[integral] AaAaAH its zeroo!!')
                continue
            
            white_mask = patch[:,:,0:3] > 200
            
            if float(np.sum(white_mask))/(patch_size**2*3) <= white_threshold :
                #if True:
                if np.sum(patch)>0:
                    # adding the patch to the patches list
                    patch_list.append(cv2.cvtColor(patch,cv2.COLOR_RGBA2BGR))
                    # adding the patch location to the list
                    patch_point.append((x_l[p_idx],y_l[p_idx]))
                    cnt += 1 # increasing patch counter
                else:
                    print('This is a black patch!')
            else:
                nt_cnt += 1
                #print('white_mask sum: ', np.sum(white_mask))
                #print('white ratio: ', float(np.sum(white_mask))/(patch_size**2*3))
                #print('Rejected location: {0},{1}'.format(x_l[p_idx],y_l[p_idx]))

            if nt_cnt %1000 == 0:
                if white_threshold < .7:
                    white_threshold += .05
                    nt_cnt = 1
                    print('Increasing white_threshold of 0.05: ', white_threshold)
                else:
                    print('No more patches to extract that have more than 30 percent of not white content')
                    break

    def_pl=[]
    def_pp=[]

    for i in range(len(patch_list)):
        if (np.sum(patch_list[i])>0) and (np.mean(patch_list[i])>90):
            def_pl.append(patch_list[i])
            def_pp.append(patch_point[i])

    return def_pl, def_pp


def tumor_patch_sampling_using_centerwin(slide,slide_level,mask,patch_size,patch_num):
    """
    tumor patch sampling using center window
    plz input the only tumor mask
    it will malfunctioned if you input normal mask or tissue mask

    input parameters are same as patch_sampling_using_integral

    """

    patch_list = []
    patch_point = []
    window_size = int(32/ slide.level_downsamples[slide_level])

    x_l,y_l = mask.nonzero()
    if len(x_l) > patch_size*2:
        level_patch_size = int(patch_size/slide.level_downsamples[slide_level])
        x_ws = (np.round(x_l*slide.level_downsamples[slide_level])).astype(int)
        y_ws = (np.round(y_l*slide.level_downsamples[slide_level])).astype(int)
        cnt=0

        while(len(patch_list) < patch_num) :
            # loop cnt
            cnt+=1
            #random Pick point in mask
            p_idx = randint(0,len(x_l)-1)
            #Get the point in mask
            level_point_x,level_point_y = x_l[p_idx], y_l[p_idx]
            #Check boundary to make patch
            check_bound = np.resize(np.array([level_point_x+level_patch_size,level_point_y+level_patch_size]),(2,))
            if check_bound[0] > mask.shape[0] or check_bound[1] > mask.shape[1]:
                continue
            #make patch from mask image
            level_patch_mask = mask[int(level_point_x):int(level_point_x+level_patch_size),int(level_point_y):int(level_point_y+level_patch_size)]

            '''Biggest difference is here'''
            #apply center window (32x32)
            cntr_x= (level_patch_size/2)-1
            cntr_y= (level_patch_size/2)-1

            win_x = cntr_x-window_size/2
            win_y = cntr_y-window_size/2

            t_window = level_patch_mask[win_x:(win_x+window_size),win_y:(win_y+window_size)]
            #print(level_patch_mask.shape)
            #print(win_x)
            #print(win_y)

            #apply integral to window
            ii_map = integral_image(t_window)
            #print(t_window.shape)
            ii_sum = integrate(ii_map,(0,0),(window_size-1,window_size-1))
            area_percent = float(ii_sum)/(window_size**2)
           # print("integral_area: ",area_percent)
           # print("loop count: ",cnt)

            if area_percent <1.0:
                continue

            if cnt > patch_num*10+1000:
                print("There is no moare patches to extract in this slide")
                print("mask region is too small")
                print("final number of patches : ",len(patch_list))

                break
            #patch,point is appended the list
            #print("region percent: ",area_percent)
            patch_point.append((x_l[p_idx],y_l[p_idx]))
            patch=slide.read_region((y_ws[p_idx],x_ws[p_idx]),0,(patch_size,patch_size))
            patch =np.array(patch)

            patch_list.append(cv2.cvtColor(patch,cv2.COLOR_RGBA2BGR))


    return patch_list, patch_point
