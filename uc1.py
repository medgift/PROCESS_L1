from __future__ import print_function

import os
import sys
import argparse

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path2 = os.path.abspath(os.path.join(file_path, '..', 'Benchmarks', 'common'))
sys.path.append(lib_path2)

import candle_keras as candle

additional_definitions = [
{'name':'training_centres',
    'action':'store',
    'nargs':'+',
    'type':int,
    'help':'training center indices'},
{'name':'source_fld_',
    'action':'store',
    'help':'source file directory'},
{'name':'xml_source_fld',
    'action':'store',
    'help':'xml source file directory'},
{'name':'PWD',
    'action':'store',
    'help':'intermediate data directory'},
{'name':'slide_level',
    'action':'store',
    'type':int,
    'help':'slide level'},
{'name':'patch_size',
    'action':'store',
    'type':int,
    'help':'patch size'},
{'name':'gpu',
    'action':'store',
    'help':'index of the GPU (as a string)'},
{'name':'n_samples',
    'action':'store',
    'type':int,
    'help':'number of samples per slide'},
{'name':'multinode',
    'action':'store',
    'type':candle.str2bool,
    'help':'use Horovod flag'},
{'name':'color',
    'action':'store',
    'default':True,
    'type':candle.str2bool,
    'help':'use color flag'},
{'name':'load',
    'action':'store',
    'default':True,
    'type':candle.str2bool,
    'help':'use color flag'},
{'name':'train',
    'action':'store',
    'default':True,
    'type':candle.str2bool,
    'help':'use color flag'}
]

required = None



class BenchmarkUC1(candle.Benchmark):

    def set_locals(self):
        """Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the
        benchmark.
        """

        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions
