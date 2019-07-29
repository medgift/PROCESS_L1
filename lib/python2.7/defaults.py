# -*- coding: utf-8 -*-
################################################################################
# For copyright see the `LICENSE` file.
#
# This file is part of PROCESS_UC1.
################################################################################
"""Default values, costants, etc. for PROCESS_UC1 cnn.

Some option can be overridden by their command line equivalent, see
:ref:`cnn`.  **Warning!** Only really working stuff (mainly related to patch
extraction) is fully documented here.

Legend: <type>(<default value> or '...' to signify to read below/source code)

Data
====

.. data:: def_config
        <dict>(...). The main configuration dictionary, where each
        nested dict corresponds to a section in the program's configuration
        file `CONFIG_FILE`.
        ::

        settings :

            project_root
                <str>. A directory wherein to look for input data and to store
                results.

            data_dir
                <str>('data'). Subdir of 'project_root' wherein to look for
                input

            results_dir
                <str>('results'). Subdir of 'project_root' wherein to look
                store output

            method
                <str>('random'). Sampling method in ['random', 'linear']

            window
                <list>([MIN, MAX]). Nonzero point index window in [0,
                100]). This will limit the sampling to the %range [MIN, MAX)
                of all available indices

            slide_level
                <int>(5). WSI resolution level to work at

            patch_size
                <int>(224). Size of a square patch in pixels

            n_samples
                <int>(500). The number of patches to extract in a batch
                (a.k.a. batch size)

            white_level
                <int>(200). RGB reference level for the white mask

            white_threshold
                <float>(.6). Discard patches "whiter" than this value in [0,
                1] (**clarify units**)

            bad_batch_size
                <int>(1000). Batch size of discarded ("bad") patches. See
                :ref:`Adaptive white threshold mechanism`

            white_threshold_incr
                <float>(.0). Increment of `white_threshold`. See
                :ref:`Adaptive white threshold mechanism`. No increment
                (default) == disable the mechanism

            white_threshold_max
                <float>(.7). Max allowed before bailing out sampling. See
                :ref:`Adaptive white threshold mechanism`.

            gray_threshold
                <int>(90). Retain patches whose RGB mean is bigger than this

            area_overlap
                <float>(.6). Total patch area should covers at least this
                fraction in (0, 1] of the annotation region

            margin_width_x
                <int>(0). Discard mask points falling within a margin this
                pixel-number wide (X-axis). Set to 0 to grab everything

            margin_width_y
                <int>(0). Same as for `margin_width_x` for Y-axis


        camelyon17 :    (see :ref:`Datasets`)

            source_fld'
                <str>('data/centre_'). Basename path for 'centre_<CID>'
                subdirs in `data_dir` where input WSI are stored.  <CID>s are
                listed in `training_centres`. The 'data' prefix is usually
                interpolated in `config.ini`

            xml_source_fld
                <str>('data/lesion_annotations'). Subdir where XML annotations
                files are stored. The 'data' prefix is usually interpolated in
                `config.ini`

            training_centres
                <list>([0, 1, 2, 3, 4]). CID list. Declare *all* of them here,
                else type conversion from `config.ini` will fail!

            centre_name_regex
                <str>('centre_(?P<CID>\d+)'). Regex to match and extract the
                center ID. Used for building the internal dataset
                representation. See :ref:`datasets.py`

            patient_name_regex
                <str>('patient_(?P<PID>\d+)_node_(?P<NID>\d+)'). Regex to
                match and extract the patient ID and the node ID. Used for
                building the internal dataset representation. See
                :ref:`datasets.py`

.. data:: test_mode
        <bool>(True). Signal that we're running with some feature disabled --
        see :ref:`Environment`

.. data:: time_stamp
        <obj>(now). A :mod:`datetime` time stamp.

.. data:: valid_pipelines
        <list>(see below...). List of valid pipeline definitions. Used to
        validate the command line.


Datasets
--------

Only the Camelyon17 dataset family is supported. The expected structure is as
follows (looked up `def_config[settings][data_dir]`):

    |-- centre_<CID>
    |   `-- patient_<PID>_node_<NID>.tif
    ...
    |-- centre_<CID>
    |   ...
    `-- lesion_annotations
        |-- patient_<PID>_node_<NID>.xml
        |-- ...

Where the <CID> range to take into account is specified by
`def_config[camelyon17][training_centres]`.


Adaptive white threshold mechanism
----------------------------------

This mechanism can be enabled _only_ with `def_config[settings][method]` :=
'random' and is disabled by default. This is somewhat overdoing: use with
caution and for tuning purposes only! Rationale: using a too low
`white_threshold` might discard valuable patches, hence the mechanism: ::

    bad_ones := count how many white patches we got so far
    white_threshold := def_config[settings][white_threshold]
    if bad_ones > def_config[settings][bad_batch_size]
        increse white_threshold of def_config[settings][white_threshold_incr]
        if white_threshold >= def_config[settings][white_threshold_max]
            bail out (stop sampling)

**Warnings:**

* if the total nonzero mask points is <= def_config[settings][bad_batch_size],
  the increment will never be triggered;

* if the total nonzero mask points is <= def_config[settings][n_samples],
  you'll probably get duplicated pathes.

In order to see how the white_threshold changes, a logging level <= DEBUG must
be set by the caller (i.e., :ref:`cnn`).


Environment
===========

The following data symbols have their corresponding ENV variable prefixed by
'PROCESS_UC1__'.

.. data:: CONFIG_FILE
        <str>('config.cfg'). Path to the program's configuration file.

.. data:: HAS_SKIMAGE_VIEW
        <bool>(False). Enable loading extra module Skimage.

.. data:: HAS_TENSORFLOW
        <bool>(False). Enable loading extra module Tensorflow. Warning: can't
        do NN training without this.

When at least one of `HAS_SKIMAGE_VIEW` and `HAS_TENSORFLOW` is
False, `test_mode` is True. This is just advisory.

Meta
====

:Authors:
    Marco E. Poleggi L<mailto:marco-emilio.poleggi -AT- hesge.ch>
"""
import sys, os
from datetime import datetime as dt

# Flags to exclude loading of some modules -- for testing purposes
# only. Please *revert* the def values once ready for production
HAS_SKIMAGE_VIEW = True if os.getenv('PROCESS_UC1__HAS_SKIMAGE_VIEW') else False  # depends on Qt
HAS_TENSORFLOW =  True if os.getenv('PROCESS_UC1__HAS_TENSORFLOW') else False     # depends on CUDA
test_mode = not (HAS_SKIMAGE_VIEW and HAS_TENSORFLOW)
# no logging yet
if test_mode:
    sys.stderr.write(
        "[defaults] Warning! Test_mode is on. Some (extra) modules won't be loaded.\n"
    )

CONFIG_FILE = os.getenv('PROCESS_UC1__CONFIG_FILE') or 'config.ini'

time_stamp = dt.now()

def_config = {
    'settings'          : {
        # globals for all datasets
        'project_root'          : '~/EnhanceR/PROCESS_UC1',
        'data_dir'              : 'data',
        'results_dir'           : 'results',
        # TO-DO: this stuff below should go elsewhere
        'method'                : 'random',
        'window'                : [],
        'gpu_id'                : 0,
        'slide_level'           : 5,
        'patch_size'            : 224,
        'n_samples'             : 500,
        'white_level'           : 200,
        'white_threshold'       : .6,
        'bad_batch_size'        : 1000,
        'white_threshold_incr'  : .0,
        'white_threshold_max'   : .7,
        'gray_threshold'        : 90,
        'area_overlap'          : .6,
        'margin_width_x'        : 0,
        'margin_width_y'        : 0,
    },
    # Datasets come in different families. For now we support only 'camelyon17'
    'camelyon16'        : {
        # see example @ <https://drive.google.com/drive/folders/0BzsdkU4jWx9Bb19WNndQTlUwb2M>
        # TO-DO
    },
    'camelyon17'        : {
        'source_fld'            : 'data/centre_',
        'xml_source_fld'        : 'data/lesion_annotations',
        'training_centres'      : [0, 1, 2, 3, 4],
        'centre_name_regex'     : 'centre_(?P<CID>\d+)',
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
        # removed in favor of per-slide DB files
        # 'h5file'                : 'patches.hdf5',
    },
}


valid_pipelines = [
    ['extract'],
    ['extract',  'train'],
    ['load', 'train']
]

################################################################################
if __name__ == '__main__':
    print(__doc__)
