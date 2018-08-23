import matplotlib
matplotlib.use('Agg')
#from openslide import OpenSlide
from os import listdir
from os.path import join, isfile, exists, splitext
import sys
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from util import otsu_thresholding, center_of_slide_level,connected_component_image
from skimage import measure
from scipy import ndimage
from functions import *
from models import getModel
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[0]
print 'Using GPU no: ', sys.argv[0]

pwd = ""
xml_path="/mnt/nas2/results/DatasetsVolumeBackup/ToReadme/CAMELYON17/lesion_notations/patient_051_node_2.xml"
filename ="/mnt/nas2/results/DatasetsVolumeBackup/ToReadme/CAMELYON17/centre_0/patient_017_node_2.tif" #listdir(pwd) [4]
slide_path = join(pwd,filename)
print "file name : "+slide_path+"\n"
if isfile(slide_path):
    """is it file? """
    print('ERR. install openslide')
    exit(0)
    slide=OpenSlide(slide_path)
elif exists(slide_path):
    """ dose it exist? """
    print "slide_path :" + slide_path + " is not a readable file \n"
else:
    """ it is not a file and doesn't exist"""
    print "file dosen't exist in this path :"  + slide_path+"\n"
slide_w, slide_h = slide.dimensions
print "Whole Slide dimensions (with, heigth):{0}\n".format(slide.dimensions)
#Slide.level_dimensions
slide_level = 7
s_level_w, s_level_h = slide.level_dimensions[slide_level]
print "slide.level_count-1 dimensions (width,heigth):{0}\n".format(slide.level_dimensions[slide_level])
#center of slide
c_x, c_y = center_of_slide_level(slide,slide_level)
print "center of x : {0} , center of y : {0}".format(c_x,c_y)

def preprocess_test_data(slide_path, slide_level=7, patch_size=224, verbose=1):
    rgb_im, slide = load_slide(slide_path, slide_level=slide_level)
    otsu_im = get_otsu_im(rgb_im, verbose = 0)
    return slide, rgb_im, otsu_im

slide, rgb_im, otsu_im = preprocess_test_data(slide_path, slide_level=7, patch_size=224, verbose=1)

im_gray_ostu = otsu_im
kernel = np.ones((2,2),np.uint8)
kernel_1 = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(im_gray_ostu,cv2.MORPH_OPEN,kernel)
opening_1 = cv2.morphologyEx(im_gray_ostu,cv2.MORPH_OPEN,kernel_1)
closing = cv2.morphologyEx(im_gray_ostu,cv2.MORPH_CLOSE,kernel)
opening_1= np.abs(255-opening_1)

plt.imshow(opening_1)

y_l, x = np.unique(np.where(opening_1>0)[0]), np.unique(np.where(opening_1>0)[1])

CONFIG_FILE='./config.cfg'
settings = parseTrainingOptions(CONFIG_FILE)
print settings
model=getModel(settings)
model.load_weights('./model.h5')

print 'Loading model: ./model.h5'

patch_size=224
patches=[]
flag=False
heatmap=np.zeros((slide.level_dimensions[slide_level][1], slide.level_dimensions[slide_level][0]))
hfp=open('heatmapFile.csv', 'w')

batch_size = settings['batch_size'] #should be 32
counter = 0
batch=[]

i=0
y = y_l[i]
while y < np.shape(opening_1)[0]*128:
    x = 0
    while x < np.shape(opening_1)[1]*128:
        patch=slide.read_region((x,y),0,(224,224))
        if np.mean(patch)>64:
            if counter<batch_size:
                batch.append(np.asarray(patch)[...,:3])
                counter += 1
            else:
                predictions=model.predict(np.reshape(batch,(batch_size,224,224,3)))
                for pred in predictions:
                    #to do: adapt to binary_crossentropy loss
                    if np.argmax(pred)==1:
                        hfp.write(str(x)+' '+str(y)+' '+str(max(pred))+'\n')
                        heatmap[int(y)/128:int(y)/128+2, int(x)/128:int(x)/128+2]=max(pred)
                batch=[]
                counter=1
                batch.append(np.asarray(patch)[...,:3])
        x += 128
    while y_l[i] - y/128 < 2:
        i += 1
        if (i==len(y_l)-1):
            flag=True
            break
    y = y_l[i] * 128
    if flag:
           break

plt.figure()
plt.imshow(rgb_im, alpha=0.9)
plt.imshow(heatmap, alpha=0.5, cmap='jet')
plt.savefig('test_heatmap.png')
