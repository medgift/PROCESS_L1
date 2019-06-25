import time, os

from functions import (
    wlog
)
'''Pipeline steps defined by function names (ugly, I now)... This is just a
basic prototype: each step would go into a separate module/class for better
control.
'''

def extract(config, results_dir, logger):
    '''Patch extaction

    Patches are extracted and saved in a HDF5 database patches.hdf5
    with the following structure:

            Tumor / Level N / Centre C / Patient P / Node No/  patches
            Tumor / Level N / Centre C / Patient P / Node No/ locations

            Normal / Level N / Centre C / Patient P / Node No/ patches
            Normal / Level N / Centre C / Patient P / Node No/ locations

    where N, C, P and No are respectively the slide_level, the current centre,
    the current patient and the current node.

    Arguments
    +++++++++

    :param config
        dict(). Program config as defined in :py:mod:`defaults`.

    :results_dir
        str(). Directory where to store the results
    '''
    logger.info('[extract] step starting...')

    from functions import createH5Dataset
    from datasets import Dataset

    h5db = createH5Dataset(
        os.path.join(results_dir, config['load']['h5file'])
    )

    c17_cfg = config['camelyon17'];
    data_dir = config['settings']['data_dir']
    camelyon17 = Dataset(
        name='camelyon17',
        slide_source_fld=os.path.join(data_dir, 'camelyon17', c17_cfg['source_fld']),
        xml_source_fld=os.path.join(data_dir, 'camelyon17', c17_cfg['xml_source_fld']),
        centres=c17_cfg['training_centres'],
        logger=logger,
        settings=config['settings']        # **FIX-ME** possibly overkill
    )

    ############################################################################
    # Core processing. Monitoring running time... [BUG] this is not the best way
    start_time = time.time()
    camelyon17.extract_patches(h5db, results_dir)
    patch_extraction_elapsed = time.time() - start_time
    #
    ############################################################################

    tot_patches = camelyon17.tum_counter +  camelyon17.nor_counter
    time_per_patch = patch_extraction_elapsed / tot_patches
    logger.info('[extract] Total elapsed time: {}'.format(patch_extraction_elapsed))
    logger.info('[extract] Time per patch: {}'.format(time_per_patch))

    h5db.close()

    logger.info('[extract] step done!')


def load(config, results_dir, logger):
    '''Load already extracted patches from an existing HDF5 database'''

    raise ApplicationError('load: pipeline step under revision')

    logger.info('[load] step starting...')

    load_settings = config['load']

    PWD = load_settings['PWD']
    h5file = load_settings['h5file']

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

    logger.info('[load] step done!')


def train(config, results_dir, logger):
    '''NN training on GPU.

    Use CNN model to classify tumor patches from normal ones.

    TO-DO: should load pretrained weights /and this is URGENT

    Also, this part might be split from the patch extraction Module as one
    could use pre-extracted patches to train the model
    '''

    raise ApplicationError('train: pipeline step under revision')

    logger.info('[train] Initialising Horovod...')
    if not HAS_TENSORFLOW:
        logger.warning('[train] bailing out as tensorflow is not available')
        return

    tf.set_random_seed(args['seed'])

    logger.info('[parallel][train] Initialising Horovod...')
    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    tfconfig.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.Session(config=config))


    # Selecting the GPU device
    GPU_DEVICE = config['settings']['GPU']
    os.putenv("CUDA_VISIBLE_DEVICES", str(GPU_DEVICE))
    logger.info('[train] Using GPU no. ', GPU_DEVICE)

    # else is greyscale...
    COLOR = True


    print '[cnn][train] Training Network...'
    net_settings = config['train']
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

    #fitModel(model, net_settings, Xtrain, Ytrain, Xval, Yval, save_history_path=results_dir)
    history = fitModel(model, net_settings, Xtrain, Ytrain, Xval, Yval, save_history_path=results_dir)
    wlog('[training] accuracy: ', history.history['acc'])
    wlog('[validation] accuracy: ', history.history['val_acc'])
    wlog('[training] loss: ', history.history['loss'])
    wlog('[validation] loss: ', history.history['val_loss'])

    model.save_weights(os.path.join(results_dir, 'tumor_classifier.h5'))


    return
