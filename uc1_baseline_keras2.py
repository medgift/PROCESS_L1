from __future__ import print_function
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', 'Benchmarks', 'common'))
sys.path.append(lib_path)
#from keras import backend as K

import uc1 as bmk
from candle_run import run
import candle_keras as candle
from keras import backend as K

def initialize_parameters():

    # Build benchmark object
    uc1Bmk = bmk.BenchmarkUC1(bmk.file_path, 'uc1_default_model.txt', 'keras',
    prog='ucb1_baseline', desc='Multi-task (DNN) for data extraction from clinical reports - Pilot 3 Benchmark 1')
    
    # Initialize parameters
    gParameters = candle.initialize_parameters(uc1Bmk)
    #bmk.logger.info('Params: {}'.format(gParameters))

    return gParameters


def main():

    settings = initialize_parameters()
    run(settings)

if __name__ == '__main__':
    main()
    try:
        K.clear_session()
    except AttributeError:      # theano does not have this function
        pass
