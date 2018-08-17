import ConfigParser
import io
import os
from os import listdir
from os.path import join
from openslide import OpenSlide
import numpy as np
import cv2
from integral import patch_sampling_using_integral
from skimage.viewer import ImageViewer
from extract_xml import *
from functions import *
from util import otsu_thresholding
import logging as llg
import datetime
import h5py as hd

def parseOptionsFromLog(folder, logfile):
    settings = {}
    settings = parseOptions('config.cfg')
    return settings

def getInfoFromXml(xml_path, centre):
    info = {}
    info['centre'] = centre
    info['patient'] = xml_path.split('patient_')[1].split('_node_')[0]
    info['node'] = xml_path.split('patient_')[1].split('_node_')[1].strip('.xml')
    return info

def createH5Dataset(dataset_file_path):
    return hd.File(dataset_file_path, "w")


def get_data_val(data_class, centres, db, dblist, rand_patient):
    pat_idx = []
    val_idx = []
    train_patches = []
    val_patches = []
    if np.shape(centres)==():
        centres=[centres]

    for centre in centres:
        for d in dblist:
            if data_class in d and ('centre'+str(centre) in d):
                pat_idx.append(d)
    
    print '[debug][get_data_val] Selected patient index for val: ', rand_patient
    val_idx = pat_idx[rand_patient::10]
    pat_idx = [x for x in pat_idx if x not in val_idx]
    for n in pat_idx:
        if db[n].shape[0]>0:
            train_patches.append(db[n])
    train_patches = np.concatenate(train_patches)
    for n in val_idx:
        if db[n].shape[0]>0:
            val_patches.append(db[n])
    val_patches = np.concatenate(val_patches)

    return pat_idx, train_patches, val_idx,val_patches

def get_data(data_class, centres, db, dblist):
    pat_idx = []
    patches = []
    if np.shape(centres)==():
        centres=[centres]

    for centre in centres:
        for d in dblist:
            if data_class in d and ('centre'+str(centre) in d):
                pat_idx.append(d)
    for n in pat_idx:
        if db[n].shape[0]>0:
            patches.append(db[n])
    patches = np.concatenate(patches)
    return pat_idx, patches

def get_dataset_val_split(centres, db, dblist):
    '''
    get_dataset:
       centres: selected centres to build the dataset
       db: h5py database of Patches
       dblist: list of records per patient case with info about tissue type
                               (normal, tumor), resolution level, node, etc
    '''
    rand_patient=np.random.randint(0,9)
    nor_pat_idx, nor_patches, val_nor_idx, val_nor_patches = get_data_val('normal', centres, db, dblist, rand_patient)
    tum_pat_idx, tum_patches, val_tum_idx, val_tum_patches = get_data_val('tumor', centres, db, dblist, rand_patient)
    x_tr = np.concatenate([nor_patches, tum_patches])
    y_tr = np.concatenate([np.zeros((len(nor_patches),1)),np.ones((len(tum_patches),1))]).T[0]
    x_val = np.concatenate([val_nor_patches, val_tum_patches])
    y_val = np.concatenate([np.zeros((len(val_nor_patches),1)),np.ones((len(val_tum_patches),1))]).T[0]
    print '[debug][functions][get_dataset_val_split] training data shape: ', np.shape(x_tr), np.shape(y_tr)
    print '[debug][functions][get_dataset_val_split] validation data shape: ', np.shape(x_val), np.shape(y_val)
    print '[debug][functions][get_dataset_val_split] trining class balance: ', len(nor_patches), len(tum_patches)
    print '[debug][functions][get_dataset_val_split] val class balance: ', len(val_nor_patches), len(val_tum_patches)
    return x_tr, y_tr, x_val, y_val

def get_dataset(centres, db, dblist):
    '''
    get_dataset:
       centres: selected centres to build the dataset
       db: h5py database of Patches
       dblist: list of records per patient case with info about tissue type
                               (normal, tumor), resolution level, node, etc
    '''
    nor_pat_idx, nor_patches = get_data('normal', centres, db, dblist)
    tum_pat_idx, tum_patches = get_data('tumor', centres, db, dblist)
    x = np.concatenate([nor_patches, tum_patches])
    y = np.concatenate([np.zeros((len(nor_patches),1)),np.ones((len(tum_patches),1))]).T[0]
    print '[debug][functions][get_dataset] loaded data shape: ', np.shape(x), np.shape(y)
    return x, y



def shuffle_data(x, y):
    '''
    Shuffle data:
     Takes dataset x and labels y and shuffles
     '''
    indexes = np.arange(0, int(len(x)))
    np.random.shuffle(indexes)
    y_shuffled =  np.asarray([y[i] for i in indexes])
    x_shuffled = np.zeros(np.shape(x))
    counter = 0
    for i in indexes:
        x_shuffled[counter] = x[i]
        counter += 1
    print 'Checking the shuffle: '
    print np.sum(x_shuffled[0]-x[indexes[0]])
    print y_shuffled[0] - y[indexes[0]]
    return x_shuffled, y_shuffled

def setDBHierarchy(h5db, settings, info):
    '''Hierarchy is a tree:

            Tumor / Level N / Centre C / Patient P / Node No/  patches
            Tumor / Level N / Centre C / Patient P / Node No/ locations

            Normal / Level N / Centre C / Patient P / Node No/ patches
            Normal / Level N / Centre C / Patient P / Node No/ locations
    '''

    if 'tumor' not in h5db:
        h5db.create_group('tumor')
    if 'normal' not in h5db:
        h5db.create_group('normal')

    if 'level{}'.format(settings['slide_level']) not in h5db.get('tumor'):
        h5db.get('tumor').create_group('level{}'.format(settings['slide_level']))
        wlog('DB Adding Level Group to Tumoral patches', 'level{}'.format(settings['slide_level']))
    if 'level{}'.format(settings['slide_level']) not in h5db.get('normal'):
        h5db.get('normal').create_group('level{}'.format(settings['slide_level']))
        wlog('DB Adding Level Group to Normal patches', 'level{}'.format(settings['slide_level']))

    if 'centre{}'.format(info['centre']) not in h5db.get('tumor/level{}'.format(settings['slide_level'])) :
        h5db.get('tumor/level{}'.format(settings['slide_level'])).create_group('centre{}'.format(info['centre']))
        wlog('DB Adding Centre Group to Tumoral Patches of Level {}'.format(settings['slide_level']), 'centre{}'.format(info['centre']))
    if 'centre{}'.format(info['centre']) not in h5db.get('normal/level{}'.format(settings['slide_level'])):
        h5db.get('normal/level{}'.format(settings['slide_level'])).create_group('centre{}'.format(info['centre']))
        wlog('DB Adding Centre Group to Normal Patches of Level {}'.format(settings['slide_level']), 'centre{}'.format(info['centre']))

    if 'patient{}'.format(info['patient']) not in h5db.get('tumor/level{}/centre{}'.format(settings['slide_level'], info['centre'])):
        h5db.get('tumor/level{}/centre{}'.format(settings['slide_level'], info['centre'])).create_group('patient{}'.format(info['patient']))
        wlog('DB Adding Patient Group to Tumor Patches', 'patient{}'.format(info['patient']))
    if 'patient{}'.format(info['patient']) not in h5db.get('normal/level{}/centre{}'.format(settings['slide_level'], info['centre'])):
        h5db.get('normal/level{}/centre{}'.format(settings['slide_level'], info['centre'])).create_group('patient{}'.format(info['patient']))
        wlog('DB Adding Patient Group to Normal Patches', 'patient{}'.format(info['patient']))

    if 'node{}'.format(info['node']) not in h5db.get('tumor/level{}/centre{}/patient{}'.format(settings['slide_level'], info['centre'], info['patient'])):
        h5db.get('tumor/level{}/centre{}/patient{}'.format(settings['slide_level'], info['centre'], info['patient'])).create_group('node{}'.format(info['node']))
        wlog('DB Adding Node Group to Tumor Patches for Patient {}'.format(info['patient']), 'node{}'.format(info['node']))

    if 'node{}'.format(info['node']) not in h5db.get('normal/level{}/centre{}/patient{}'.format(settings['slide_level'], info['centre'], info['patient'])):
        h5db.get('normal/level{}/centre{}/patient{}'.format(settings['slide_level'], info['centre'], info['patient'])).create_group('node{}'.format(info['node']))
        wlog('DB Adding Node Group to Normal Patches for Patient {}'.format(info['patient']), 'node{}'.format(info['node']))
    return

def wlog(tag, info):
    ''' wlog:
        saves the information in info into a log file
    '''
    llg.basicConfig(level=llg.INFO)
    logger = llg.getLogger(__name__)
    logger.info(tag + ': '+ str(info))
    return

def load_slide(slide_path, slide_level=6, verbose = 0):
    ''' load_slide:
        loads the WSI as a Numpy array

        input:
        slide_path, path to the WSI
        slide_level, level of resolution (default = 6)

        output:
        rgb_img, [Numpy array] loaded image
        slide, [OpenSlide object] slide
    '''
    slide = OpenSlide(slide_path)
    rgba_im = slide.read_region((0,0),slide_level,slide.level_dimensions[slide_level])
    rgba_im= np.array(rgba_im)
    rgb_im = cv2.cvtColor(rgba_im,cv2.COLOR_RGBA2RGB)
    if verbose:
        print 'Loading: ', slide_path
        plt.imshow(rgb_im)
    return rgb_im, slide

def rgb2gray(rgb_im):
    ''' Conversion to greyscale '''
    return cv2.cvtColor(rgb_im,cv2.COLOR_RGB2GRAY)

def gray2otsu(gray_im, verbose = 1):
    ''' Otsu thresholding '''
    otsu_im, o_th = otsu_thresholding(gray_im)
    if verbose:
        print 'Otsu threshold: ', o_th
    return otsu_im

def otsu2morph(otsu_im, verbose = 0):
    '''Morphology '''
    kernel_o = np.ones((5,5),np.uint8)
    #kernel_c = np.ones((1,1),np.uint8)
    morp_im = cv2.morphologyEx(otsu_im,cv2.MORPH_OPEN,kernel_o)
    morp_im = morp_im == 0
    morp_im = (morp_im).astype(float)
    if verbose:
        viewer = ImageViewer(morp_im)
        viewer.show()
    return morp_im

def get_morp_im(rgb_im, verbose = 0):
    return otsu2morph(gray2otsu(rgb2gray(rgb_im)))

def get_otsu_im(rgb_im, verbose = 0):
    return gray2otsu(rgb2gray(rgb_im))

'''OLD preprocess: fino a che non te sei accorta che
	hai fatto na cazzata

def preprocess(slide_path,  xml_path, slide_level = 6, patch_size = 256, verbose = 1):
    ''''''function preprocess:
        WSI preprocessing to extract tumor patches

        input:
        slide_path, path to WSI
        slide_level, level of resolution (default = 6)

        output:
        mask, [Numpy array of 0 and 1s] tumor annotation mask
        rgb_img, [Numpy array] loaded image
    ''''''
    rgb_im, slide = load_slide(slide_path, slide_level=slide_level)

    tumor_contours = get_opencv_contours_from_xml(xml_path,slide.level_downsamples[slide_level])
    tum_im = rgb_im

    con =cv2.drawContours(tum_im, tumor_contours,-1,(0,255,0), 5)

    _,annotations,_ = cv2.split(tum_im)
    annotations = annotations == 255
    annotations_mask = annotations.astype(int)

    return slide, annotations_mask, rgb_im, tum_im
'''

def preprocess(slide_path,  xml_path, slide_level = 7, patch_size = 224, verbose = 1):
    '''function preprocess:
        WSI preprocessing to extract tumor patches

        input:
        slide_path, path to WSI
        slide_level, level of resolution (default = 6)

        output:
        mask, [Numpy array of 0 and 1s] tumor annotation mask
        rgb_img, [Numpy array] loaded image
    '''
    print'[functions][data_preprocessing] NEW PREPROCESS FUNCTION'
    rgb_im, slide = load_slide(slide_path, slide_level=slide_level)

    tumor_contours = get_opencv_contours_from_xml(xml_path,slide.level_downsamples[slide_level])
    tum_im=rgb_im

    mask=np.zeros(tum_im[...,0].shape,np.uint8)
    con=cv2.drawContours(mask, tumor_contours,-1,(255,0,0), 2)
    tum =cv2.drawContours(tum_im, tumor_contours,-1,(255,0,0), 3)
    annotations_mask=cv2.fillPoly(mask, pts =[cn for cn in tumor_contours], color=(255,255,255))
    annotations_mask=mask

    return slide, annotations_mask, rgb_im, tum_im

def check_data(centre, source_fld, xml_path):
    pwd = source_fld + str(centre) + '/'
    WSI_file = xml_path[:-3]+'tif'

    print 'Workin with: ', WSI_file
    slide_path = join(pwd,WSI_file)
    return slide_path, None

'''
def get_WSI_path(centre, source_fld, xml_file):
    WSI_file = xml_file[:-3]+'tif'

    print 'Workin with: ', WSI_file
    slide_path = join(source_fld+str(centre),WSI_file)
    return slide_path

def get_annotation_list(centre, xml_source_fld):
    xml_of_selected_centre = []
    xml_list = listdir(xml_source_fld)
    for x in xml_list:
        identifier = x[-13]
        if centre == 0:
            if int(identifier)<=1:
                xml_of_selected_centre.append(x)
        elif centre == 1:
            if int(identifier) == 2 or int(identifier) == 3:
                xml_of_selected_centre.append(x)
        elif centre == 2:
            if int(identifier) == 4 or int(identifier) == 5:
                xml_of_selected_centre.append(x)
        elif centre == 3:
            if int(identifier) == 6 or int(identifier) == 7:
                xml_of_selected_centre.append(x)
        elif centre == 4:
            if int(identifier) == 8 or int(identifier) == 9:
                xml_of_selected_centre.append(x)
    return xml_of_selected_centre
'''
def parseOptions(configFile):
    settings = {}

    config = ConfigParser.RawConfigParser(allow_no_value = True)
    config.read(configFile)

    training_centres = []
    centres = config.get("settings", "training_centres").split(',')
    for c in centres:
        training_centres.append(int(c))

    settings['training_centres'] = training_centres
    settings['source_fld'] = config.get("settings", "source_fld")
    settings['xml_source_fld'] = config.get("settings", "xml_source_fld")
    settings['slide_level'] = int(config.get("settings", "slide_level"))
    settings['patch_size'] = int(config.get("settings", "patch_size"))
    settings['n_samples'] = int(config.get("settings", "n_samples"))

    '''Logging the info'''
    wlog('training_centres', settings['training_centres'])
    wlog('source_fld', settings['source_fld'])
    wlog('xml_source_fld', settings['xml_source_fld'])
    wlog('slide_level', settings['slide_level'])
    wlog('patch_size', settings['patch_size'])
    wlog('n_samples', settings['n_samples'])

    return settings

def parseLoadOptions(configFile):
    settings = {}
    config = ConfigParser.RawConfigParser(allow_no_value = True)
    config.read(configFile)
    settings['PWD'] = config.get("load", "PWD")
    settings['h5file'] = config.get("load", "h5file")
    return settings

def parseTrainingOptions(configFile):
    settings = {}
    config = ConfigParser.RawConfigParser(allow_no_value = True)
    config.read(configFile)
    settings['model_type'] = config.get("train", "model_type")
    settings['activation'] = config.get("train", "activation")
    settings['loss'] = config.get("train", "loss")
    settings['lr'] = float(config.get("train", "lr"))
    settings['decay'] = float(config.get("train", "decay"))
    settings['momentum'] = float(config.get("train", "momentum"))
    settings['nesterov'] = config.get("train", "nesterov")

    settings['batch_size'] = int(config.get("train", "batch_size"))
    settings['epochs'] = int(config.get("train", "epochs"))
    settings['verbose'] = int(config.get("train", "verbose"))
    return settings

def getFolderName():
    return str(datetime.datetime.now()).split(' ')[0][-5:].split('-')[0]+str(datetime.datetime.now()).split(' ')[0][-5:].split('-')[1]+'-'+str(datetime.datetime.now()).split(' ')[1][:5].split(':')[0]+str(datetime.datetime.now()).split(' ')[1][:5].split(':')[1]
