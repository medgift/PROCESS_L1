from defaults import HAS_SKIMAGE_VIEW, HAS_TENSORFLOW, valid_pipelines
import ConfigParser, io, sys, re
import pprint as pp
from argparse import ArgumentTypeError
from os import listdir, mkdir
from os.path import join, expanduser
from openslide import OpenSlide
import numpy as np
import cv2
from integral import patch_sampling_using_integral
if HAS_SKIMAGE_VIEW:
    from skimage.viewer import ImageViewer
from extract_xml import *
from util import otsu_thresholding
import logging as llg
import h5py as hd
################################################################################
# globals
debug = False
conf_debug = False
logger = None
################################################################################

'''To-DO
* remove all parse...() functions obsoleted by `parseConfig`
'''

'''** NOT IMPLEMENTED YET** Do not use
'''
def parseOptionsFromLog(folder, logfile):
    raise AppError, 'Not implemeted yet'

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
    # [BUG] there's no dataset created here, so pretty useless
    return hd.File(dataset_file_path, "w")


def get_data_val(data_class, centres, db, db16, dblist, rand_patient):
    pat_idx = []
    cam16_idx = []
    val_idx = []
    train_patches = []
    val_patches = []
    if np.shape(centres)==():
        centres=[centres]

    for centre in centres:
        for d in dblist:
            if data_class in d and ('centre'+str(centre) in d):
                if 'Mask' not in d:
                    pat_idx.append(d)
                else:
                    cam16_idx.append(d)

    logger.debug('[debug][get_data_val] Selected patient index for val: {}'.format(rand_patient))
    val_idx = pat_idx[rand_patient::10]
    pat_idx = [x for x in pat_idx if x not in val_idx]
    for n in pat_idx:
        if db[n].shape[0]>0:
            train_patches.append(db[n])
    for n in cam16_idx:
        if db16[n].shape[0]>0:
            train_patches.append(db16[n])
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

def get_dataset_val_split(centres, db, db16, dblist):
    '''
    get_dataset:
       centres: selected centres to build the dataset
       db: h5py database of Patches
       dblist: list of records per patient case with info about tissue type
                               (normal, tumor), resolution level, node, etc
    '''
    rand_patient=np.random.randint(0,9)
    nor_pat_idx, nor_patches, val_nor_idx, val_nor_patches = get_data_val('normal', centres, db, db16, dblist, rand_patient)
    tum_pat_idx, tum_patches, val_tum_idx, val_tum_patches = get_data_val('tumor', centres, db, db16, dblist, rand_patient)
    x_tr = np.concatenate([nor_patches, tum_patches])
    y_tr = np.concatenate([np.zeros((len(nor_patches),1)),np.ones((len(tum_patches),1))]).T[0]
    x_val = np.concatenate([val_nor_patches, val_tum_patches])
    y_val = np.concatenate([np.zeros((len(val_nor_patches),1)),np.ones((len(val_tum_patches),1))]).T[0]
    logger.debug('training data shape: {}, {}'.format(np.shape(x_tr), np.shape(y_tr)))
    logger.debug('validation data shape: {}, {}'.format(np.shape(x_val), np.shape(y_val)))
    logger.debug('trining class balance: {}, {}'.format(len(nor_patches), len(tum_patches)))
    logger.debug('val class balance: {}, {}'.format(len(val_nor_patches), len(val_tum_patches)))
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
    logger.debug('loaded data shape: {}, {}'.format(np.shape(x), np.shape(y)))
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
    logger.info(
        'Checking the shuffle: {}, {}'.format(
            np.sum(x_shuffled[0]-x[indexes[0]]),
            y_shuffled[0] - y[indexes[0]]
        )
    )
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



def log_init(
        log_fname='info.log',
        log_level='info'
):
    """Init a global logger

    Arguments
    +++++++++

    :param str log_fname: log file name. Can be a path which must exist

    :param str log_level: logging level

    TO-DO
    +++++

    * replace with logreport
    """
    global debug, logger

    try:
        numeric_level = getattr(llg, log_level.upper())
    except Exception as e:
        raise ValueError('Invalid log level: "%s" (exception: %s)' % (log_level, str(e)))

    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)


    log_fmt = '%(asctime)s %(levelname)-8s'
    # more fuss with 'DEBUG'... looks like there's no way to discriminate by tag name
    if numeric_level <= 10:
        log_fmt += ' %(funcName)-12s'
        debug = True

    llg.basicConfig(
        format   = log_fmt + ' %(message)s',
        level    = numeric_level,
        datefmt  = '%Y-%m-%d %H:%M:%S',
        filename = log_fname,
        filemode = 'w'
    )

    # console logging of messages with priority higher than INFO to the sys.stderr
    console = llg.StreamHandler()
    console.setLevel(numeric_level)
    log_fmt = llg.Formatter('%(levelname)-8s %(message)s')
    console.setFormatter(log_fmt)
    llg.getLogger(__name__).addHandler(console)

    llg.captureWarnings(True)
    logger = llg.getLogger(__name__)

    logger.info("log file is '%s'" % log_fname)

    return logger

def wlog(subj, msg):
    ''' wlog: saves the information in info into a log file
    '''
    logger.info(subj + ': '+ msg)


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
        logger.info('Loading: {}'.format(slide_path))
        plt.imshow(rgb_im)
    return rgb_im, slide

def rgb2gray(rgb_im):
    ''' Conversion to greyscale '''
    return cv2.cvtColor(rgb_im,cv2.COLOR_RGB2GRAY)

def gray2otsu(gray_im, verbose=1):
    ''' Otsu thresholding '''
    otsu_im, o_th = otsu_thresholding(gray_im)
    if verbose:
        logger.info('Otsu threshold: {}'.format(o_th))
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

def preprocess(
        slide_path,  xml_path,
        slide_level=7,
        patch_size=224,
        verbose=1
):
    """WSI preprocessing to extract tumor patches

    preprocess takes the WSI path, and the slide_level and returns the
    the WSI openslide obj, the tumor annotation mask, the WSI image and
    the tumor contours


    input:
    slide_path, path to WSI
    slide_level, level of resolution (default = 6)

    output:
    mask, [Numpy array of 0 and 1s] tumor annotation mask
    rgb_img, [Numpy array] loaded image
    """
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

    logger.info('Workin with: {}'.format(WSI_file))
    slide_path = join(pwd,WSI_file)
    return slide_path, None

def parseOptions(configFile):
    raise AppError, 'Obsoleted by `parseConfig()`'

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

'''Parse an ini-style 'configFile', merge its contents with dict 'defConfig'
and return a new 'config' dict (same structure as `defaults.def_config`)

This is the only function needed to store program's config once for all.
'''
def parseConfig(configFile, defConfig):
    def _typeof(obj):
        '''Return a list of stringified type(obj) and types of its elements, if any
        (only lists are supported).
        '''
        fre = lambda v: re.match("<type\s+'(\w+)'>", str(type(v))).group(1)
        types = [fre(obj)]
        if isinstance(obj, list):
            types += map(fre, obj)
        return types

    def _typify(stuff, types):
        '''Type-cast the `stuff` string according to list `types` as returned by
        `_typeof()`. types[0] is the container type, followin elements, if
        any, specify the types of comma-splitted `stuff`
        '''
        # t is a stringified type, like 'str'
        if conf_debug:
            sys.stderr.write("[debug] stuff={} <=> types={}\n".format(stuff, types))
        tf = lambda t, v: eval(t)(v)
        items = stuff
        if len(types) > 1:
            # supposedly an uniform type list, e.g. all int
            sstuff = stuff.split(',')
            stypes = types[1:]
            dlen = len(sstuff) - len(stypes)
            if dlen > 0:
                # more stuff than types, discard some stuff
                sstuff = sstuff[:dlen+1]
            elif dlen < 0:
                stypes = stypes[:dlen]

            try:
                items = map(tf, stypes, sstuff)
            except TypeError as e:
                sys.exit("Configuration mismatch: {} (given values) <=> {} (default types): {}\n".format(stuff, types, e))

        return tf(types[0], items)


    config = {}
    # parser = ConfigParser.RawConfigParser(defConfig, allow_no_value=True)
    parser = ConfigParser.SafeConfigParser(defConfig)
    try:
        parser.readfp(open(configFile))
    except IOError as e:
       sys.stderr.write("[warn] {}: can't read config file -- {}. Will use defaults\n"
                        .format(configFile, e))

    # merge. defaults() with "sections" is rather useless: indeed, `get(sct,
    # opt)` won't get the corresponding default value, unless 'opt' is a
    # top-level key in defaults().
    for sct in parser.defaults():
        config[sct] = {}
        try:
            parser.add_section(sct)
        except ConfigParser.DuplicateSectionError:
            pass

        for opt, val in parser.defaults()[sct].items():
            # ConfigParser is type-agnostic (uhm... great), thus types are
            # inferred from defaults
            t = _typeof(val)
            opt_from_file = False
            try:
                # may be empty, and will stay so
                val = parser.get(sct, opt)
                opt_from_file = True
            except ConfigParser.NoOptionError:
                pass

            # sys.stderr.write("[dbg] parser set: %s, %s, %s (%s => %s)\n" % (sct, opt, val, type(val), t))

            config[sct][opt] = val
            if opt_from_file:
                # type-recast
                config[sct][opt] = _typify(val, t)

            # expand ~ in paths. $HOME & Co. might be also supported... TO-DO
            if isinstance(config[sct][opt], str) and '/' in config[sct][opt]:
                config[sct][opt] = expanduser(config[sct][opt])

    return config

def parseLoadOptions(configFile):
    raise AppError, 'Obsoleted by `parseConfig()`'

    settings = {}
    config = ConfigParser.RawConfigParser(allow_no_value = True)
    config.read(configFile)
    settings['PWD'] = config.get("load", "PWD")
    settings['h5file'] = config.get("load", "h5file")
    return settings

def parseTrainingOptions(configFile):
    # [BUG] should we review and add any missing option??
    raise AppError, 'Obsoleted by `parseConfig()`'

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



'''
Validata list:`pipeline`. Allowed values are defined in `defaults::valid_pipelines`
'''
def validate_pipeline(pipeline):
    if not pipeline in valid_pipelines:
        raise ArgumentTypeError("{}: must be in {}".format(pipeline, valid_pipelines))

    return pipeline

def validate_non_neg_int(string):
    value=int(string)
    if value < 0:
        raise ArgumentTypeError("{}: must be non negative integer".format(string))

    return value
