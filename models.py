from keras.preprocessing import image
from keras.applications import resnet50
import sys
sys.path.insert(0, '/mnt/nas2/results/Models')
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from resnet101 import *
from keras.preprocessing import image
import tensorflow as tf
from functions import *
import matplotlib.pyplot as plt
import os
from image import *
from normalizers import *
#import horovod.keras as hvd
from keras.callbacks import Callback
from keras.callbacks import CSVLogger
import time

class SGDLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr * (1. / (1. + tf.cast(optimizer.decay, tf.float32) * tf.cast(optimizer.iterations, tf.float32))))
        logs['lr'] = lr
        print('\n LR:  {:.6f}\n'.format(lr))

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

### TO DO: Change num_classes to 1 and categorical_crossentropy to binary_crossentropy
def getModel(net_settings, settings, num_classes=1):
    '''
    Should be modified with model type as input and returns the desired model
    '''
    if net_settings['model_type'] == 'resnet':
        base_model = resnet50.ResNet50(include_top=True, weights='imagenet')
        finetuning = Dense(1, activation='sigmoid', name='predictions')(base_model.layers[-2].output)
        model = Model(input=base_model.input, output=finetuning)

        ## Adjust learning rate based on number of GPUs
        #hv_lr = net_settings['lr'] * hvd.size()
        #opt = optimizers.SGD(lr = hv_lr,  momentum=0.9, decay=1e-6, nesterov=True)
        ## Adding Horovod DistributedOptimizer
        #opt = hvd.DistributedOptimizer(opt)
        opt = optimizers.SGD(lr = net_settings['lr'],  momentum=net_settings['momentum'], decay=net_settings['decay'], nesterov=net_settings['nesterov'])
        model.compile(loss=net_settings['loss'],
        optimizer= opt, metrics=['accuracy'])
        if settings['multinode']=='True':
            callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0),]
            if hvd.rank() == 0:
                callbacks.append(keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))
        #return model
    elif net_settings['model_type'] == 'resnet101':
        model = resnet101_model(net_settings, settings['patch_size'], settings['patch_size'], 3, 1)
        hv_lr = net_settings['lr']
        if settings['multinode']=='True':
            ## Adjust learning rate based on number of GPUs
            hv_lr = net_settings['lr'] * hvd.size()
        opt = optimizers.SGD(lr = hv_lr,  momentum=net_settings['momentum'], decay=net_settings['decay'], nesterov=net_settings['nesterov'])
        if settings['multinode']=='True':
            ## Adding Horovod DistributedOptimizer
            opt = hvd.DistributedOptimizer(opt)
        model.compile(loss=net_settings['loss'],
                      optimizer= opt,
                      metrics=['accuracy'])
        if settings['multinode']=='True':
            callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0),]
            if hvd.rank() == 0:
                callbacks.append(keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))
        #return model
    else:
        print('[models] Ugggh. Not ready for this yet.')
        exit(0)
        return None
    return model    
def standardPreprocess(data):

    print('[models] Appling some standard preprocessing to the data. ')
    preprocessedData = np.asarray([resnet50.preprocess_input(x) for x in data])
    print('[models] data mean: ', np.mean(preprocessedData))
    print('[models] data std: ', np.std(preprocessedData))

    return preprocessedData

def get_normalizer(patch):
    normalizer = ReinhardNormalizer()
    normalizer.fit(patch)
    np.save('normalizer',normalizer)
    np.save('normalizing_patch', patch)
    print('Normalisers saved to disk.')
    return normalizer

def normalize_patch(patch, normalizer):
    return np.float64(normalizer.transform(np.uint8(patch)))

def normalize_batch(batch, normalizer):
    for i in range(len(batch)):
        for p in range(len(batch[i][0])):
            normalized_patch = normalize_patch(np.uint8(batch[i][0][p]), normalizer)
            batch[i][0][p] = normalized_patch
    return batch

def fitModel(model, net_settings, X_train, y_train, X_val, y_val, save_history_path='', batch_size=1, epochs=2, data_augmentation=1, verbose=1):

    normalizer = get_normalizer(np.uint8(X_train[0]))
    X_train = [normalize_patch(np.uint8(x), normalizer) for x in X_train]
    X_val = [normalize_patch(np.uint8(x), normalizer) for x in X_val]

    X_train = standardPreprocess(X_train)
    X_val = standardPreprocess(X_val)
    
    time_callback = TimeHistory()

    if not data_augmentation:
        history = model.fit(X_train, y_train,
                batch_size=net_settings['batch_size'],
                epochs=net_settings['epochs'],
                verbose = net_settings['verbose'],
                validation_data=(X_val, y_val),
                callbacks = [time_callback, SGDLearningRateTracker(), CSVLogger(os.path.join(save_history_path, 'train_stats.log'))]
                )

        print('[models] Training history keys stored: ', history.history.keys())
        # Plotting info about accuracy
        plt.figure()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        #plt.show()
        plt.savefig(os.path.join(save_history_path,'trainingAccuracy.png'))
        plt.close()
        # Plotting info about loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        #plt.show()
        plt.savefig(os.path.join(save_history_path,'trainingLoss.png'))
        plt.close()
        return history

    else:
        print('[models] [NEW!] Using real-time data augmentation.')
        # Data Augmentation: new module!
        datagen = ImageDataGenerator(
            contrast_stretching=False, #####
            histogram_equalization=False,#####
            random_hue=True, #####
            random_saturation=False, #####
            random_brightness=True, #####
            random_contrast=False, #####
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0.0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.0,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.0,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True)  # randomly flip images


        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train)
        os.mkdir(os.path.join(save_history_path,'adata'))
        # Fit the model on the batches generated by datagen.flow().
        history = model.fit_generator( #normalize_batch(
                      datagen.flow(X_train,
                                   y_train,
                                   batch_size=net_settings['batch_size']),
                                   epochs=net_settings['epochs'],
                                   steps_per_epoch= len(X_train) // net_settings['batch_size'] * 3,
                                   validation_data=(X_val, y_val),
                                   callbacks = [time_callback,
                                                SGDLearningRateTracker(),
                                                CSVLogger(os.path.join(save_history_path, 'train_stats.log'))]
                                )



        #verbose=net_settings['verbose'], max_q_size=200) #,
        #callbacks=[lr_reducer, early_stopper, csv_logger]) validation_data=(X_val, y_val),
        print(history.history.keys())
        # Plotting info about accuracy
        plt.figure()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        #plt.show()
        plt.savefig(os.path.join(save_history_path,'trainingAccuracy.png'))
        plt.close()
        # Plotting info about loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        #plt.show()
        plt.savefig(os.path.join(save_history_path,'trainingLoss.png'))
        plt.close()

        # Plotting info about training loss
        plt.plot(history.history['loss'])
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        #plt.show()
        plt.savefig(os.path.join(save_history_path,'ONLYtrainingLoss.png'))
        plt.close()

        model.save_weights('model.h5')
        print('Model saved to disk')

    return history, time_callback.times

def resizePatches(train, val):
    ## not used so far
    Xtrain = np.zeros(len(train), 224, 224)
    Xval = np.zeros(len(val), 224, 224)
    c = 0
    for patch in train:
        img = cv2.resize(patch.astype('float32'), (224, 224))
        Xtrain[c] = img
        c += 1
        #Xtrain.append(img)
    c=0
    for patch in val:
        img = cv2.resize(patch.astype('float32'), (224, 224))
        Xval[c] = img
        #Xval.append(img)
        c += 1

    return Xtrain, Xval
