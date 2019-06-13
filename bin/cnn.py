#!/usr/bin/env python2
# -*- coding: utf-8 -*-
################################################################################
# cnn: EnhanceR PROCESS UC1
################################################################################
# For copyright see the `LICENSE` file.
#
# This file is part of PROCESS_UC1.
################################################################################
# Flags to exclude loading of some modules -- for testing purposes only
HAS_SKIMAGE_VIEW = False  # depends on Qt
HAS_TENSORFLOW = False    # depends on CUDA, used in 'train' step
################################################################################
# system deps
import matplotlib
import sys
import os
from os import listdir
from os.path import join, isfile, exists, splitext
from random import shuffle
import cv2
import numpy as np
from PIL import Image
from skimage.transform.integral import integral_image, integrate
if HAS_SKIMAGE_VIEW:
    from skimage.viewer import ImageViewer
# import skimage
from skimage import io
if HAS_TENSORFLOW:
    from keras import backend as K
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
    import tensorflow as tf
    import tflearn
    from tflearn.data_utils import shuffle, to_categorical
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.estimator import regression
    import horovod.keras as hvd
import time
import h5py as hd
import shutil
import math
################################################################################
# local deps -- path kludge, waiting for proper packaging...
sys.path.insert(0, 'lib/python2.7/')
from datasets import Dataset
from extract_xml import *
from functions import *
from integral import patch_sampling_using_integral
from models import *
################################################################################

# defaults
config = {
    'data_dir'          : './data',
    'results_dir'       : './results',
    'settings'          : {
        'GPU'                   : 0,
        'training_centres'      : [0, 1, 2, 3],
        'source_fld'            : '%(data_dir)/centre_',
        'xml_source_fld'        : '%(data_dir)/lesion_annotations',
        'slide_level'           : 5,
        'patch_size'            : 224,
        'n_samples'             : 500,
    },
    'train'             : {
        'model_type'            : 'resnet',
        'loss'                  : 'binary_crossentropy',
        'activation'            : 'sigmoid',
        'lr'                    : 1e-4,
        'decay'                 : 1e-6,
        'momentum'              : 0.9,
        'nesterov'              : True,
        'batch_size'            : 32,
        'epochs'                : 10,
        'verbose'               : 1,
    },
    'load'              : {
        'PWD'                   : '/mnt/nas2/results/IntermediateResults/Camelyon/all500',
        'h5file'                : 'patches.hdf5',
    },
}

################################################################################

matplotlib.use('agg')

'''
cnn.py -- EnahnceR PROCESS_UC1 launch script

Usage
=====

High resolution patch extraction
--------------------------------

    $ python2 cnn.py - - [XML_ANNOTATION_FILE]

Options
-------

    XML_ANNOTATION_FILE
        Path to a single XML annotation file. If given a single WSI slide will
        be processed for any configured 'centre'


'''

single_ann_mode = False
single_ann_file = None
if sys.argv[4]:
    single_ann_mode = True
    single_ann_file = sys.argv[4]
    print('[cnn][config] single annotation mode. XML file: ', single_ann_file)

# ***TO-DO*** arg5 to be exposed to CLI?
np.random.seed(int(sys.argv[5]))
print 'Setting random seed ', sys.argv[5]

if HAS_TENSORFLOW:
    tf.set_random_seed(int(sys.argv[5]))

    print '[parallel][train] Initialising Horovod...'
    hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))


cam16fld='/mnt/nas2/results/DatasetsVolumeBackup/ToCurate/ContextVision/Camelyon16/TrainingData/Train_Tumor/'
cam16xmls = '/mnt/nas2/results/DatasetsVolumeBackup/ToCurate/ContextVision/Camelyon16/TrainingData/Ground_Truth/Mask/'

''' Loading system configurations '''
# TO-DO: have a '--config-file' option
CONFIG_FILE = os.environ["PROCESS-UC1__CONFIG_FILE"] or 'config.cfg'
print '[cnn][config] Loading system configurations from: ', CONFIG_FILE

''' Selecting the GPU device for training '''
#GPU_DEVICE = get_gpu_from_config(CONFIG_FILE)
GPU_DEVICE = sys.argv[3]
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_DEVICE

''' Selection of COLOR or GREYSCALE'''
COLOR = True

print '[cnn][config] Using GPU no. ', GPU_DEVICE

''' Creating a log file for the run
For each run the script creates a saving folder with the following namesystem:

MMDD-HHmm
where:   MM stands for Month
         DD stands for day
         HHmm stands for Hour and Minute
'''
new_folder = getFolderName()
os.mkdir(new_folder)

# creating an INFO.log file to keep track of the model run
llg.basicConfig(
    filename=os.path.join(new_folder, 'INFO.log'),
    filemode='w',
    level=llg.INFO
)
shutil.copy2(src='./config.cfg', dst=os.path.join(new_folder, '.'))

'''Getting the system configurations from CONFIG_FILE
The system configuration can be set in config.cfg >> see README.rd for more information.
settings is a dictionary containing the following keys:

    training_centres
    source_fld_
    xml_source_fld
    slide_level
    patch_size
    n_samples
'''
load_settings = parseLoadOptions(CONFIG_FILE)

''' Selecting the modality:
load: the patches database is loaded from a separated storage folder
train: network training is performed
'''
load_db = False
if sys.argv[1] == 'load':
    load_db = True
    # if load_db then open the folder with the db
    #and use it, together with the settings specified in INFO.log.

training = False
if sys.argv[2] == 'train':
    training = True

if load_db :
    ''' DATABASE LOADING

    Loading already extracted patches from an existing HDF5 database.
    '''

    settings = parseOptionsFromLog('0102-1835', 'INFO.log') # to implement

    print '[cnn][config] Loading data config: ', load_settings

    PWD = load_settings['PWD'] #/home/mara/CAMELYON.exps/dev05/'
    h5file = load_settings['h5file'] #'0109-1415/patches.hdf5'

    cam16 = hd.File('./data/intermediate_datasets/cam16/patches.hdf5', 'r')
    h5db = hd.File(os.path.join(PWD, h5file), 'r')
    ''' Old dev
    all_tumor_patches = h5db['all_tumor_patches']
    all_normal_patches = h5db['all_normal_patches']
    '''

    ''' NEW DATA LOAD MODULE

    Training and validation centres are selected and disjoint
    Manual Shuffle -- to improve
    '''
    '''DATASET INFO'''
    global dblist
    #global dblist_cam16

    dblist=[]
    def list_entries(name, obj):
        global dblist
        if '/patches' in name:
            dblist.append(name)
    #def list_entries_16(name, obj):
    #    global dblist_cam16
    #    if '/patches' in name:
    #        dblist_cam16.append(name)
    h5db.visititems(list_entries)
    cam16.visititems(list_entries)
    print '[debug][cnn] dblist: ', dblist

else:
    ''' PATCH EXTRACTION MODULE

    Here the patches are extracted and saved in a HDF5 database patches.hdf5
    with the following structure:

            Tumor / Level N / Centre C / Patient P / Node No/  patches
            Tumor / Level N / Centre C / Patient P / Node No/ locations

            Normal / Level N / Centre C / Patient P / Node No/ patches
            Normal / Level N / Centre C / Patient P / Node No/ locations

    where N, C, P and No are respectively the slide_level, the current centre,
    the current patient and the current node.

    '''
    settings = parseOptions(CONFIG_FILE)

    start_time = time.time() # time is recorded to track performance
    h5db = createH5Dataset(os.path.join(new_folder, 'patches.hdf5'))

    camelyon17 = Dataset(
        name='camelyon17',
        slide_source_fld='/mnt/nas2/results/DatasetsVolumeBackup/ToReadme/CAMELYON17/',
        xml_source_fld='/mnt/nas2/results/DatasetsVolumeBackup/ToReadme/CAMELYON17/lesion_annotations',
        centres = settings['training_centres'],
        settings=settings
    )

    #camelyon17.extract_patches(h5db, new_folder)

    camelyon16 = Dataset(
        name='camelyon16',
        slide_source_fld=cam16fld,
        xml_source_fld=cam16xmls,
        settings=settings
    )
    camelyon16.settings['slide_level']=7
    camelyon16.settings['training_centres']=0
    camelyon16.settings['xml_source_fld']=cam16xmls
    camelyon16.settings['source_fld']=cam16fld
    camelyon16.settings['n_samples']=200

    camelyon16.extract_patches(h5db, new_folder)

    # Monitoring running time
    patch_extraction_elapsed = time.time()-start_time
    tot_patches = camelyon16.tum_counter + \
                  camelyon16.nor_counter + \
                  camelyon17.tum_counter + \
                  camelyon17.nor_counter
    time_per_patch = patch_extraction_elapsed / tot_patches
    wlog('ElapsedTime for Patch Extraction: ', patch_extraction_elapsed)
    wlog('Time per patch: ', time_per_patch)

    h5db.close()

print '[cnn] [patch_extraction = FINISHED] DB saved'
    #exit(0)

if training:
    ''' CNN model to classify tumor patches from normal patches

        Note: should load pretrained weights /and this is URGENT

        Also, this part might be split from the patch extraction Module
        as one could use pre-extracted patches to train the model
    '''

    if not HAS_TENSORFLOW:
        exit('Tensorflow not available')

    print '[cnn][train] Training Network...'
    net_settings = parseTrainingOptions(CONFIG_FILE)
    # Adjust number of epochs based on number of GPUs
    net_settings['epochs'] = int(math.ceil(float(net_settings['epochs'])/ hvd.size()))

    wlog('Network settings', net_settings)

    patch_size = settings['patch_size']

    # note: shuffling so that we are removing dependencies btween the
    # extracted patches. One of the problems here might be that if you have overlapping
    # patches or really similar patches then they end up being present both in the
    # training split and in the testing splits. This would give high accuracy but
    # wouldn't necessarily mean that you are learning the right thing.
    settings['split']='validate'

    print '[cnn][train] Data split config: ', settings['split']
    wlog('[cnn][train] Data split config: ', settings['split'])

    settings['color']=COLOR

    if settings['color']:
        print '[cnn][train] COLOR patches'
        wlog('[cnn][train] COLOR patches', '')
    else:
        print '[cnn][train] GREYSCALE patches'
        wlog('[cnn][train] GREYSCALE patches', '')

    if settings['split']=='shuffle':
        tum_patch_list = shuffle(all_tumor_patches)[0]
        nor_patch_list = shuffle(all_normal_patches)[0]
        train_ix = int(len(tum_patch_list) * 0.8)

        print  '[cnn][train] Number of training tumor patches: ', train_ix
        wlog('[cnn][train] No. training tumor patches: ', train_ix)

        train_patches = np.zeros((2*train_ix, patch_size, patch_size, 3))
        train_patches[:train_ix]=tum_patch_list[:train_ix]
        train_patches[train_ix:]=nor_patch_list[:train_ix]
        '''Labeling:
        1 for Tumor patch
        0 for Normal patch
        '''
        y_train = np.zeros((2*train_ix))
        y_train[:train_ix]=1

        stop_idx = min(len(tum_patch_list), len(nor_patch_list))
        val_ix=( stop_idx - train_ix)

        val_patches = np.zeros((2*val_ix, patch_size, patch_size, 3))
        val_patches[:val_ix]=tum_patch_list[train_ix:stop_idx]
        val_patches[val_ix:]=nor_patch_list[train_ix:stop_idx]
        y_val = np.zeros((2*val_ix))
        y_val[:val_ix]=1

    elif settings['split']=='sequential':
        print  '[cnn][train] SEQUENTIAL split '
        wlog('[cnn][train] SEQUENTIAL split ', '')

        if not settings['color']:
            tum_patch_list = [cv2.cvtColor( np.uint8(patch_entry), cv2.COLOR_RGB2GRAY) for patch_entry in all_tumor_patches]
            nor_patch_list = [cv2.cvtColor( np.uint8(patch_entry), cv2.COLOR_RGB2GRAY) for patch_entry in all_normal_patches]
        else:
            tum_patch_list = [patch_entry for patch_entry in all_tumor_patches]
            nor_patch_list = [patch_entry for patch_entry in all_normal_patches]

        train_ix = int(len(tum_patch_list) * 0.8)

        print  'Number of training tumor patches: ', train_ix
        wlog('No. training tumor patches: ', train_ix)

        train_patches = np.zeros((2*train_ix, patch_size, patch_size, 3))
        if not settings['color']:
            # replicating the grayscale image on each channel
            train_patches[:train_ix,:,:, 0]= train_patches[:train_ix,:,:, 1]= train_patches[:train_ix,:,:, 2]= shuffle(tum_patch_list[:train_ix])[0]
            train_patches[train_ix:,:,:,0]= train_patches[train_ix:,:,:,1]= train_patches[train_ix:,:,:,2]= shuffle(nor_patch_list[:train_ix])[0]
        else:
            train_patches[:train_ix] = shuffle(tum_patch_list[:train_ix])[0]
            train_patches[train_ix:]= shuffle(nor_patch_list[:train_ix])[0]

        ''' Labeling:
        1 for Tumor patch
        0 for Normal patch
        '''

        y_train = np.zeros((2*train_ix))
        y_train[:train_ix]=1

        stop_idx = min(len(tum_patch_list), len(nor_patch_list))
        val_ix=( stop_idx - train_ix)

        val_patches = np.zeros((2*val_ix, patch_size, patch_size, 3))

        if not settings['color']:
            # replicating the grayscale image on each channel
            val_patches[:val_ix,:,:, 0]= val_patches[:val_ix,:,:, 1]= val_patches[:val_ix,:,:, 2]= shuffle(tum_patch_list[train_ix:stop_idx])[0]
            val_patches[val_ix:,:,:,0]= val_patches[val_ix:,:,:,1]= val_patches[val_ix:,:,:,2]= shuffle(nor_patch_list[train_ix:stop_idx])[0]
        else:
            val_patches[:val_ix] = tum_patch_list[train_ix:stop_idx]
            val_patches[val_ix:] = nor_patch_list[train_ix:stop_idx]

        y_val = np.zeros((2*val_ix))
        y_val[:val_ix]=1


    elif settings['split']=='select':
        '''NEW DATA LOAD MODULE ## to merge'''
        print '[cnn][split = select] Training centres: ', settings['training_centres'][:-1]
        print '[cnn][split = select] Validation centres: ', settings['training_centres'][-1]
        x_train, y_train = get_dataset(settings['training_centres'][:-1], h5db, dblist)
        x_val, y_val = get_dataset(settings['training_centres'][-1], h5db, dblist)
        x_train = x_train[:5]
        y_train = y_train[:5]
        x_val = x_val[:5]
        y_val = y_val[:5]
    elif settings['split']=='validate':
        '''VALIDATION FOR CHALLENGE
           we isolate one random patient from each center to create the validation set
        '''
        print '[cnn][split = validate] Picking N slides for validation from each center (keeping the patients separated): '
        x_train, y_train, x_val, y_val = get_dataset_val_split(settings['training_centres'], h5db, cam16, dblist)




    '''There you go. here you should have both training data and validation data '''
    '''Maybe shuffling. Cleaning. whiteninig etccccccc'''

    '''
    batch_size = 5
    num_classes = 2
    epochs = 50
    '''
    ###Note: Ya need some data preprocessing here.
    # PostComment: Do I?

    #new shufflin
    Xtrain, Ytrain = shuffle_data(x_train, y_train)
    Xval, Yval = shuffle_data(x_val, y_val)
    ## old shuffling
    #Xtrain, Ytrain = shuffle(train_patches, y_train)
    #Xval, Yval = shuffle(val_patches, y_val)

    #  use only for Categorical Crossentropy loss:
    #  encode the Ys as 2 class labels
    if net_settings['loss']=='categorical_crossentropy':
        print 'Encoding the labels to categorical..'
        Ytrain = to_categorical(Ytrain, 2)
        Yval = to_categorical(Yval, 2)

    print 'Trainig dataset: ', Xtrain.shape
    print 'Validation dataset: ',Xval.shape
    print 'Trainig labels: ', Ytrain.shape
    print 'Validation labels: ', Yval.shape
    wlog('Training data: ', Xtrain.shape)
    wlog('Validation data: ', Xval.shape)

    model = getModel(net_settings)

    #fitModel(model, net_settings, Xtrain, Ytrain, Xval, Yval, save_history_path=new_folder)
    history = fitModel(model, net_settings, Xtrain, Ytrain, Xval, Yval, save_history_path=new_folder)
    wlog('[training] accuracy: ', history.history['acc'])
    wlog('[validation] accuracy: ', history.history['val_acc'])
    wlog('[training] loss: ', history.history['loss'])
    wlog('[validation] loss: ', history.history['val_loss'])

    model.save_weights(os.path.join(new_folder, 'tumor_classifier.h5'))
