# PROCESS Initial Architectures

This repository contains the first Use Case Application for UC#1 of the
PROCESS project, http://www.process-project.eu/.

# UC1_medicalImaging

The use case tackles cancer detection and tissue classification on the latest
challenges in cancer research using histopathology images, such as CAMELYON
and TUPAC.

**This code is work-in-progress: only a limited subset of functions are
implemented and correctly working!**

## CAMNET

Camelyon17 network for binary classification of breast tissue
patches. Camelyon16 is currently **not supported**.

Dataset are fully described in module `dataset.py`. They're essentially a
collection of raw Whole Slide Images (WSIs), each corresponding to a *patient
case* from a given *centre*:

    centre_C
        patient_P_node_N.{tif,xml}
        ...

# Installation

No packaging support for now. Please **FIX ME**.

## Dependencies

The code is written in Python 2.7 and requires Keras 2.1.5 with Tensorflow
1.4.0 as backend. Further dependencies are in `Requirements.txt`.

## Configuration

Configuration files are ini-based. A full template is in
`doc/config.template.ini`. Example configurations are available in
`etc/`. Please, adjust your configuration and use it by calling one of the
master scripts with option `--config-file` whose default value is `config.ini`
(which doesn't exist). Alternatively, a configuration file can be set via the
env variable `PROCESS-UC1__CONFIG_FILE`. F.i. on a *nix-like system, run

    $ PROCESS-UC1__CONFIG_FILE=<path-to-your-config-file> <master-script>

See below, fo configuration details.

## Usage

The only working _master script_ is `bin/cnn`. Get the full help by calling:

    $ ./bin/cnn --help

This master script is a sort of pipeline-based program, in the form

    cnn [OPTIONS] [STEP1 [STEP2...]]

For now only two pipeline steps are allowed and at least one is mandatory. Here is the
valid step sequences (or pipelines):

    #  pipeline                        step1    step2
    -------------------------------------------------
    1   patch extraction               extract
    2   patch extraction + training    extract  train
    3   training existing patches      load     train

**Development status** For now only the `extract` step is working in a
sequential random sampling fashion, i.e., patches are sequentially extracted
over a uniform distribution of image (WSI) points. The remaining steps are
implemented but not tested with the latest refactored code.


### Configuration file

Only a subset of all options are available as command line arguments, thus a
configuration file is usually needed.  An ini-style configuration file has the
following section-based structure:

    [settings]
    ...
    [camelyon17]
    # dataset parameters
    ...
    [load]
    # step parameters
    ...
    [train]
    # step parameters
    ...

Where:

* **[settings]**: common parameters and settings for the *patch extraction*
    module, such as paths to the input dataset and where to store results,
    etc.;
* **[camelyon17]**: *dataset-specific* parameters, such as subpaths to WSIs
  and annotations, the hospital centres to use, etc.;
* **[train]**: training settings for the network, such as model type, loss,
  activation function etc.;
* **[load]**: settings for loading the preprocessed dataset.

Full description is available in `doc/config.template.ini`.

### Pipeline steps

* **extract** high resolution patches from any WSI in the input
  datasets. Performed on CPU;
* **load** a pre-existing patch database from storage;
* **train** the neural network model. Performed on GPGPU, wehere available.

#### High resolution patch extraction ###

The command

    cnn [OPTIONS] -c my_config.ini extract

launches the patch extraction module on the raw WSIs of a Camelyon17
dataset. Regions of Interest are extracted from the manual annotation regions
and encoded as low resolution binary masks (one mask for nontumor tissue and
one mask for tumor tissue). High resolution patches (i.e. level 1) are
randomly sampled from the tumor tissue and the normal tissue.  Tumor and
tissue coverage is computed as the integral over the selected region and
patches with less than 80% of informative content are rejected. Mostly white
and black patches are rejected as well.

Patches are hierarchically stored in a h5 file with the following tree structure:

    Tumor  / Level N / Centre C / Patient P / Node No / patches
    Tumor  / Level N / Centre C / Patient P / Node No / locations
    Normal / Level N / Centre C / Patient P / Node No / patches
    Normal / Level N / Centre C / Patient P / Node No / locations
    ...

Results are stored under the directory specified by

    <results_dir>/<results_sdir>

where `<results_dir>` is defined under config file's section `[settings]`, and
`<results_sdir>` is either auto-generated as a "MMDD-hhmm" timestamp or
defined by command line argument `--results-sdir`. So, supposing that
`results_dir = /foo/bar`, to always use the same subdirectory, one would call

    cnn -c my_config.ini --results-sdir=test extract

then `/foo/bar/test` would contain something like:

    my_config.ini
    INFO.log
    levelL_centreC_patientP_nodeN_annotation_mask.png
    levelL_centreC_patientP_nodeN_normal_tissue_mask.png
    levelL_centreC_patientP_nodeN_tumor_locations.png
    ...
    patches.hdf5

where `my_config.ini` is a copy of the input configuration file, `INFO.log` is
the execution log, then images of extracted patches (binary masks of tumor and
nontumor tissue regions plus a png map with the patch sampling locations, for
each WSI -- L, C, P and N are integers) follow, and finally the h5 patch DB
`patches.hdf5`.

By default, all the patient cases found in the input dataset are processed. To
explicitly process only a patient subset, append a `-p` or `--patients`
(without the '=' sign) option to the command line:

    cnn -c my_config.ini --results-sdir=test extract --patients patient_P_node_N ...

Note that each patient string `patient_P_node_N` has no XML/TIF extension.

Full examples of usage are in [Section Slurm Scripts](#slurm).


### Network training ###

**Warning. This section needs reviewing.**

Before training the NN model, an h5 patch DB must be available, either by
extracting patches from a datest with command

    cnn [OPTIONS] -c my_config.ini extract train

or by loading a pre-computed h5 DB file with command

    cnn [OPTIONS] -c my_config.ini load train

In both cases the input to the `train` step is specified in the `[load]`
section of `my_config.ini` -- **PLEASE REVIEW**. Whereas output is composed of

* the model new weights stored in file `tumor_classifier.h5`,
* the training curves as png images,

all stored under directory `<results_dir>/<results_sdir>` (**PLEASE VERIFY**).

Training is performed on GPGPU, by default the first one in the array (index
'0'). This can be either specified in `my_config.ini`

    [settings]
    GPU = <index in [0, N]>

or on the command line

    cnn -c my_config.ini --gpu-id=1 load train

See `bin/train.sh` for a full example.


## <a name="#slurm"></a>Slurm scripts

Some scripts are available in `bin/` to launch parallel `cnn` runs via the
Slurm batch system. These are working examples which can achieve linear
speed-up (minus system overhead).

### <a name="#slurm_rnd_seed"></a>Parallel patch extraction by randomized seed

The file `bin/runner-slurm_rnd_seed` uses a Slurm job array where each task is
configured with a different random seed. The goal is to exploit randomness in
the WSI point distribution in order to extract a different patch set at each
batch run. Here's the relevant parts of the bash script (slightly simplified),
where the random seed is set by the batch scheduler task ID in an array of [0,
N]. Note also that results are stored in different subdirs.

    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=1
    #SBATCH --array=0-9

    task_id=$(printf '%03d' $SLURM_ARRAY_TASK_ID)

    cnn --config-file=... \
        --results-sdir=.../${task_id} \
        --seed=${task_id}\
        extract

Please, review the full script, then call it like this:

    some-hpc$ sbatch runner-slurm_rnd_seed

Under the hypothesis of infinite resources (i.e., available CPUs), this script
achieves linear speed-up as each task is assigned a different Slurm "node",
i.e. a CPU. No GPGPU is involved.


### <a name="#slurm_onep-onep"></a>Parallel slide processing

The file `bin/runner-onep-onep` uses a Slurm job array where each task is
assigned a different WSI (a single "patient case") while the random seed is
the same. Here's the relevant parts of the bash script (slightly simplified).
Note also that results are stored in different subdirs.

    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=1
    #SBATCH --array=0-<NUMBER_OF_XML_FILES>

    task_id=$(printf '%03d' $SLURM_ARRAY_TASK_ID)

    declare -a patients
    patients=($(ls --color=none some-dir/with-xml-annotations/ | sed -nr 's/\.xml// p'))
    patient=${patients[${SLURM_ARRAY_TASK_ID}]}

    cnn --config-file=... \
        --results-sdir=.../${task_id} \
        --patients=${patient}\
        extract

**N.B.** The the parameter `<NUMBER_OF_XML_FILES>` should be set to the total
number of "patients". Since the `patients` array is built by listing the
content of a directory with Camelyon17-style XML annotation files, their total
number should be known in advance. Failing to do so will result in either
running useless tasks (more task slots than patients), or overloading the
batch system (more patients than task slots) with subsequent less-than-linear
speed-up.

Please, review the full script, then call it like this:

    some-hpc$ sbatch runner-onep-onep

Under the hypothesis of infinite resources (i.e., available CPUs), this script
achieves linear speed-up. No GPGPU is involved.
