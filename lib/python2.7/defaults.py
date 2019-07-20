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
        # globals for all datasets
        'project_root'          : '~/EnhanceR/PROCESS_UC1',
        'data_dir'              : 'data',       # subdir of 'project_root'
        'results_dir'           : 'results',    # (ditto)
        # TO-DO: this stuff below should go elsewhere
        'method'                : 'random',
        'window'                : [],
        'gpu_id'                : 0,
        'slide_level'           : 5,
        'patch_size'            : 224,
        'n_samples'             : 500,
        'white_level'           : 200,          # RGB reference level for the white mask
        'white_threshold'       : .3,           # [0, 1]. Discard patches "whiter" than this (**clarify units**)
        'white_threshold_incr'  : .05,          # increment of `white_threshold` when too many bad patches
                                                # are found. i.e. >= `bad_batch_size`
        'white_threshold_max'   : .7,           # max allowed before bailing out
        'gray_threshold'        : 90,           # retain patches whose RGB mean is bigger than this
        'area_overlap'          : .6,           # total patch area should covers at least this % of the annotation
                                                # region
        'bad_batch_size'        : 1000,         # every these "bad" patches, increase the white threshold
        'margin_width_x'        : 0,            # (pixel number) discard mask points falling within this margins.
        'margin_width_y'        : 0,            # Set to 0 to grab everything
    },
    # Datasets come in different families. For now we support only 'camelyon17'
    'camelyon16'        : {
        # see example @ <https://drive.google.com/drive/folders/0BzsdkU4jWx9Bb19WNndQTlUwb2M>
        # TO-DO
    },
    'camelyon17'        : {
        # file system structure (stored under`[settings][data_dir]`) is like
        #
        # |-- centre_1
        # |   `-- patient_021_node_3.tif
        # | ...
        # |-- centre_N
        # |   `-- patient_086_node_4.tif
        # `-- lesion_annotations
        #     |-- patient_004_node_4.xml
        #     |-- ...
        #     `-- patient_086_node_4.xml
        #
        # basename for 'centre_<CID>', where <CID>s are listed in `training_centres`
        'source_fld'            : 'data/centre_',               # 'data' is interpolated in `config.ini`
        'xml_source_fld'        : 'data/lesion_annotations',    # (ditto)
        'training_centres'      : [0, 1, 2, 3, 4],              # CID list. Declare *all* of them here,
                                                                # else type conversion will fail!
        'centre_name_regex'     : 'centre_(?P<CID>\d+)',        # match center ID
        # slide files and annotation files bear names like
        #     patient_<PID>_node_<NID>.{xml, tif}
        # we want to match <PID> (patient ID) and <NID> (node ID)
        'patient_name_regex'    : 'patient_(?P<PID>\d+)_node_(?P<NID>\d+)',
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
        'PWD'                   : 'results/intermediate',       # 'results' is interpolated in `config.ini`
        'h5file'                : 'patches.hdf5',
    },
}


valid_pipelines = [
    ['extract'],
    ['extract',  'train'],
    ['load', 'train']
]

################################################################################
