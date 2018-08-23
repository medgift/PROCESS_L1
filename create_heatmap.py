import matplotlib
matplotlib.use('agg')
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

training_slide=True
pwd='/mnt/nas2/results/DatasetsVolumeBackup/ToReadme/CAMELYON17/centre_0'


# In[50]:


val_files=os.listdir(pwd)


# In[41]:


def preprocess_test_data(slide_path, slide_level=7, patch_size=224, verbose=1):

    rgb_im, slide = load_slide(slide_path, slide_level=slide_level)
    otsu_im = get_otsu_im(rgb_im, verbose = 0)
    return slide, rgb_im, otsu_im

def apply_morph(otsu_im):
    im_gray_ostu = otsu_im
    kernel = np.ones((2,2),np.uint8)
    kernel_1 = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(im_gray_ostu,cv2.MORPH_OPEN,kernel)
    opening_1 = cv2.morphologyEx(im_gray_ostu,cv2.MORPH_OPEN,kernel_1)
    closing = cv2.morphologyEx(im_gray_ostu,cv2.MORPH_CLOSE,kernel)
    opening_1= np.abs(255-opening_1)
    return opening_1


# In[51]:


val_files


# In[53]:


os.environ["CUDA_VISIBLE_DEVICES"]="0"

folder='../trained_models/t012v3/'
CONFIG_FILE=folder+'config.cfg'

settings = parseTrainingOptions(CONFIG_FILE)
print settings

model=getModel(settings)
model.load_weights(folder+'tumor_classifier.h5')


# In[ ]:


for wsi in val_files:
    if '.tif' in wsi:
        slide_path = os.path.join(pwd,wsi)
        print "file name : "+slide_path+"\n"
        slide, rgb_im, otsu_im = preprocess_test_data(slide_path, slide_level=7, patch_size=224, verbose=1)
        slide_w, slide_h = slide.dimensions
        print "Whole Slide dimensions (with, heigth):{0}\n".format(slide.dimensions)
        #Slide.level_dimensions
        slide_level = 7
        s_level_w, s_level_h = slide.level_dimensions[slide_level]
        print "slide.level_count-1 dimensions (width,heigth):{0}\n".format(slide.level_dimensions[slide_level])
        opening_1 = apply_morph(otsu_im)
        plt.figure()
        plt.imshow(opening_1)
        plt.savefig('temp/'+wsi+'opening.png')
        from models import standardPreprocess

        y_l, x_l = np.unique(np.where(opening_1>0)[0]), np.unique(np.where(opening_1>0)[1])
        patch_size=224
        patches=[]
        flag=False
        print 'Heatmap dimensions: ', slide.level_dimensions[slide_level][1], slide.level_dimensions[slide_level][0]
        heatmap=np.zeros((slide.level_dimensions[slide_level][1], slide.level_dimensions[slide_level][0]))
        analysed=np.zeros((slide.level_dimensions[slide_level][1], slide.level_dimensions[slide_level][0]))

        hfp=open(os.path.join('temp/', wsi+'heatmapFile.csv'), 'w')
        all_predictions=[]
        batch_size = 128 #should be 32
        counter = 0
        batch=[]
        batch_locations=[]


        i=0
        y = y_l[i]
        while y < np.shape(opening_1)[0]*slide.level_downsamples[slide_level]:
            x = 0
            while x < np.shape(opening_1)[1]*slide.level_downsamples[slide_level]:
                print x, y
                lowResX = int(x/slide.level_downsamples[slide_level])
                lowResY = int(y/slide.level_downsamples[slide_level])
                #if lowResY/np.shape(opening_1)[0] %10 ==0:
                #    print lowResY/np.shape(opening_1)[0],'% '
                if opening_1[lowResY, lowResX]!=0:
                    analysed[lowResY-1:lowResY+1, lowResX-1:lowResX+1]=1
                    patch=np.asarray(slide.read_region((x,y),0,(224,224)), dtype=np.float32)[...,:3]

                    #Image.fromarray(np.uint8(patch)).save(folder+'patches/{}-{}.png'.format(y,x))
                    #if np.mean(patch)>64:
                    if counter<batch_size:
                        batch.append(patch)
                        counter += 1
                    else:
                        batch=np.asarray(batch)
                        batch=np.reshape(batch,(batch_size,224,224,3))
                        batch.setflags(write=1)
                        #print (batch)
                        batch = standardPreprocess(batch)

                        predictions=model.predict(batch)
                        #print np.shape(batch)
                        for p in range(len(predictions)):
                                lrY, lrX = batch_locations[p]
                                probability = max(predictions[p])
                                hfp.write(str(lrX)+' '+str(lrY)+' '
                                          +str(probability)+'\n')
                                heatmap[lrY-1:lrY+1, lrX-1:lrX+1] = probability
                                if probability>.5:
                                    all_predictions.append(1)
                                else:
                                    all_predictions.append(0)
                        batch=[]
                        batch_locations=[]
                        del predictions
                        counter=1
                        batch.append(np.asarray(patch)[...,:3])
                        print np.shape(batch)

                    #plt.imshow(patch)
                    #plt.savefig('patches/{}-{}.png'.format(y,x))
                    #counter += 1
                x += 128
            while y_l[i] - y/128 < 2:
                i += 1
                if (i==len(y_l)-1):
                    flag=True
                    break
            #i=i+1
            y = y_l[i] * 128
            if flag:
                   break
        np.save('temp/heatmaps/'+wsi+'analysed.npy', analysed)
        np.save('temp/heatmaps/'+wsi+'heatmap.npy', heatmap)
	plt.rcParams["figure.figsize"] = (60,60)
	plt.figure()
	plt.imshow(rgb_im)
	plt.imshow(heatmap, cmap='jet', alpha=.7)
	plt.savefig('temp/heatmaps/heatmap'+wsi+'.png')
	#break
