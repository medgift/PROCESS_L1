import matplotlib
matplotlib.use('Agg')
from models import *
import keras
from models import getModel, standardPreprocess
from functions import parseTrainingOptions, parseLoadOptions
import h5py as hd
import os
from tflearn.data_utils import shuffle, to_categorical
import matplotlib.pyplot as plt
import sklearn.metrics
import numpy as np
import sys

os.environ["CUDA_VISIBLE_DEVICES"]="0"
folder = sys.argv[1]

CONFIG_FILE=os.path.join(folder, 'config.cfg')
print 'System Configuration: ', CONFIG_FILE

COLOR = True

settings = parseTrainingOptions(CONFIG_FILE)

print settings
model=getModel(settings)
model.load_weights('./'+folder+'tumor_classifier.h5')


load_settings = parseLoadOptions(CONFIG_FILE)

PWD = '../aprildb/'#0212-1131/' #load_settings['PWD'] #/home/mara/CAMELYON.exps/dev05/'
h5file = 'patches.hdf5' #'0109-1415/patches.hdf5'

h5db = hd.File(os.path.join(PWD, h5file), 'r')

all_tumor_patches = h5db['all_tumor_patches']
all_normal_patches = h5db['all_normal_patches']

patch_size=224
if COLOR:
    tum_patch_list = [np.uint8(patch_entry) for patch_entry in all_tumor_patches]
    nor_patch_list = [np.uint8(patch_entry) for patch_entry in all_normal_patches]
else: 
    tum_patch_list = [np.uint8(cv2.cvtColor( np.uint8(patch_entry), cv2.COLOR_RGB2GRAY)) for patch_entry in all_tumor_patches]
    nor_patch_list = [np.uint8(cv2.cvtColor( np.uint8(patch_entry), cv2.COLOR_RGB2GRAY)) for patch_entry in all_normal_patches]
     
train_ix = int(len(tum_patch_list) * 0.95)

print  'Number of training tumor patches: ', train_ix

# replicating the grayscale image on each channel
train_patches = np.zeros((2*train_ix, patch_size, patch_size, 3))

if COLOR:
    train_patches[:train_ix]= tum_patch_list[:train_ix]
    train_patches[train_ix:]= nor_patch_list[:train_ix]
else:
    train_patches[:train_ix,:,:, 0]= train_patches[:train_ix,:,:, 1]= train_patches[:train_ix,:,:, 2]= tum_patch_list[:train_ix]
    train_patches[train_ix:,:,:,0]= train_patches[train_ix:,:,:,1]= train_patches[train_ix:,:,:,2]= nor_patch_list[:train_ix]



'''Labeling:
1 for Tumor patch
0 for Normal patch'''

y_train = np.zeros((2*train_ix))
y_train[:train_ix]=1

''' Validation'''
'''#mode1
stop_idx = min(len(tum_patch_list), len(nor_patch_list))
val_ix=( stop_idx - train_ix)

val_patches = np.zeros((2*val_ix, patch_size, patch_size, 3))
if COLOR:
    val_patches[:val_ix] = tum_patch_list[-val_ix:]
    val_patches[val_ix:] = nor_patch_list[-val_ix:]
else:
    val_patches[:val_ix,:,:, 0]= val_patches[:val_ix,:,:, 1]= val_patches[:val_ix,:,:, 2]= tum_patch_list[-val_ix:]
    val_patches[val_ix:,:,:,0]= val_patches[val_ix:,:,:,1]= val_patches[val_ix:,:,:,2]= nor_patch_list[-val_ix:]

y_val = np.zeros((2*val_ix))
y_val[:val_ix]=1

'''
#mode2
#stop_idx = min(len(tum_patch_list), len(nor_patch_list))
#val_ix=( stop_idx - train_ix)

val_tum = len(tum_patch_list) - train_ix
val_nor = len(nor_patch_list) - train_ix

val_len = val_tum + val_nor

val_patches = np.zeros((val_len, patch_size, patch_size, 3))

if COLOR:
    val_patches[:val_tum] = tum_patch_list[-val_tum:]
    val_patches[val_tum:] = nor_patch_list[-val_nor:]
else:
    val_patches[:val_tum,:,:, 0]= val_patches[:val_tum,:,:, 1]= val_patches[:val_tum,:,:, 2]= tum_patch_list[-val_tum:]
    val_patches[val_tum:,:,:,0]= val_patches[val_tum:,:,:,1]= val_patches[val_tum:,:,:,2]= nor_patch_list[-val_nor:]

y_val = np.zeros((val_len))
y_val[:val_tum]=1

#End mode 2
''''''


## Appling the same preprocessing to the val patches. really importanttt.
val_patches=standardPreprocess(val_patches)
y_pred=model.predict(val_patches)
#y_val=to_categorical(y_val,2)

np.save(folder+'val_patches', val_patches)
np.save(folder+'y_labels', y_val)
np.save(folder+'y_pred', y_pred)

auc = sklearn.metrics.roc_auc_score(y_val, y_pred)
print 'Model AUC: ', auc

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_val[:], y_pred[:])
    roc_auc[i] =  sklearn.metrics.auc(fpr[i], tpr[i])
fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(np.asarray(y_val).ravel(), np.asarray(y_pred).ravel())
roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.title('ROC curve, AUC = '+str(auc))

fp=open(folder+'ROCrates.txt', 'w')
fp.write('FPR: '+str(tpr))
fp.write('\n TPR: '+str(tpr))
fp.close()

plt.savefig(folder+'modelROC.png')