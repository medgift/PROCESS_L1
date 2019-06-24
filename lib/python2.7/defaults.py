# -*- coding: utf-8 -*-
################################################################################
# For copyright see the `LICENSE` file.
#
# This file is part of PROCESS_UC1.
################################################################################
# default values, costants. etc.
import sys, os
from datetime import datetime as dt

# Flags to exclude loading of some modules -- for testing purposes
# only. Please *revert* the def values once ready for production
HAS_SKIMAGE_VIEW = True if os.getenv('PROCESS_UC1__HAS_SKIMAGE_VIEW') else False  # depends on Qt
HAS_TENSORFLOW =  True if os.getenv('PROCESS_UC1__HAS_TENSORFLOW') else False     # depends on CUDA
TEST_MODE = not (HAS_SKIMAGE_VIEW and HAS_TENSORFLOW)
# no logging yet
if TEST_MODE:
    sys.stderr.write("[warn] TEST_MODE is on. Skimage, tensorflow won't be loaded!\n")

CONFIG_FILE = os.getenv('PROCESS_UC1__CONFIG_FILE') or 'config.cfg'

time_stamp = dt.now()

def_config = {
    'settings'          : {
        'data_dir'              : 'data',
        'results_dir'           : 'results',
        'GPU'                   : 0,
        'training_centres'      : [0, 1, 2, 3],
        'source_fld'            : 'data/centre_',
        'xml_source_fld'        : 'data/lesion_annotations',
        'slide_level'           : 5,
        'patch_size'            : 224,
        'n_samples'             : 500,
    },
    # pipeline steps
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
    # TO-DO: clarify items' meaning
    'load'              : {
        'PWD'                   : 'results/intermediate',
        'h5file'                : 'patches.hdf5',
    },
}


valid_pipelines = [
    ['extract'],
    ['extract',  'train'],
    ['load', 'train']
]

################################################################################
