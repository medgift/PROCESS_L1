# PROCESS Initial Architectures

This repository contains the first Use Case Application for UC#1 of the PROCESS project, http://www.process-project.eu/.

# UC1_medicalImaging

The use case tackles cancer detection and tissue classification on the latest challenges in cancer research using histopathology images, such as CAMELYON and TUPAC.

**CAMNET:

Camelyon17 network for binary classification of breast tissue patches.

**Dependencies

The code is written in Python 2.7 and requires Keras 2.1.5 with Tensorflow 1.4.0 as backend
Further dependencies are in Requirements.txt  
See Requirements.txt

**Usage

The main script is cnn.py, which reads three input variables and one config file.
Input variables specify the execution modality for the script:
-load: accepted values are {load, -}. When 'load', the patches database is loaded from a separated storage folder. If '-' then the high resolution patches are extracted from the raw Whole Slide Images (WSIs).
-train: accepted values are {train, -}. When 'train', network training is performed.
-GPU number: accepted values are integer numbers {0..9}+ which specify the GPU to use to run computations
Main sections in the CONFIG.cfg file:
-[settings]: general settings for the patch extraction module, such as the hospital centres to use, the path to WSIs data and annotations, the slides resolution level, etc.  
-[train]: training settings for the network (model type, loss, activation function etc.)
-[load]: settings for loading the preprocessed dataset

There are two main execution modalities:
<item>High resolution patch extraction</item>
<item>Network training</item>

The script always outputs a folder with the following naming convention basing on the system time: DDMM-HHMM.
The folder will contain a INFO.log file with a recap of the pipeline steps and a copy of the config.cfg file. The folder will also store further outputs of the pipeline.
For each WSI we save binary masks of tumor and nontumor tissue regions and a png map with the patch sampling locations.
At the end of training the model weights are stored in tumor_classifier.h5, and the training curves are stored in png images.

***High resolution patch extraction

example: python cnn.py - - 0

main output: patches.h5

The example command launches the patch extraction module on the raw WSIs using GPU #0.
Regions of Interest are extracted from the manual annotation regions and encoded as low resolution binary masks (one mask for nontumor tissue and one mask for tumor tissue). High resolution patches (i.e. level 1) are randomly sampled from the tumor tissue and the normal tissue.
Tumor and tissue coverage is computed as the integral over the selected region and patches with less than 80% of informative content are rejected. Mostly white and black patches are rejected as well.
Patches are hierarchically stored in a h5 file with the following tree structure:
            Tumor / Level N / Centre C / Patient P / Node No/  patches
            Tumor / Level N / Centre C / Patient P / Node No/ locations
            Normal / Level N / Centre C / Patient P / Node No/ patches
            Normal / Level N / Centre C / Patient P / Node No/ locations

***Network training

example: python cnn.py load train 0 

main output: tumor_classifier.h5

The example command launches the training of the network on the patches dataset, whose path needs to be specified in the [load] section of the config.cfg file.

See train.sh for more information


NOTE: All the systems are currently under development and may need further debugging and troubleshooting. 
