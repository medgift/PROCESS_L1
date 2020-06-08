# Layer 1 of PROCESS Use Case 1 application

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
        patient_P_node_N.tif
        ...
    centre_...
        ...
    lesion_annotations
        patient_P_node_N.xml
        ...


# Installation

No packaging support for now.

## Dependencies

The code is written in Python 2.7 and requires Keras 2.1.5 with Tensorflow
1.4.0 as backend. Further dependencies are in `requirements.txt`.

## Configuration

Configuration files are ini-based. A full template is in
`doc/config.template.ini`. Example configurations are available in
`etc/`. Please, adjust your configuration and use it by calling one of the
master scripts with option `--config-file` whose default value is `config.ini`
(which doesn't exist). Alternatively, a configuration file can be set via the
env variable `PROCESS_UC1__CONFIG_FILE`. F.i. on a *nix-like system, run

    $ PROCESS_UC1__CONFIG_FILE=<path-to-your-config-file> <master-script>

See below for configuration details.

## Usage

For now, the only working _master script_ is `bin/cnn`. Get the full help by calling:

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

#### High resolution patch extraction

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

Patches are hierarchically stored in `h5` DB files with the following tree
structure (blank spaces added for readability sake):

    tumor  / l<level> / c<centre> / p<patient> / n<node> / patches   [/ <batch>]
    tumor  / l<level> / c<centre> / p<patient> / n<node> / locations [/ <batch>]
    normal / l<level> / c<centre> / p<patient> / n<node> / patches   [/ <batch>]
    normal / l<level> / c<centre> / p<patient> / n<node> / locations [/ <batch>]
    ...

Where the extra key component `[/ <batch> ]` is added when there are more
patches to extract than `config[settings] : n_samples`.

Results are stored in the directory specified by config file option `results_dir` defined
under in section `[settings]`, which can be overridden by command line argument
`--results-dir`, f.i.:

    cnn -c my_config.ini --results-dir=path/to/my/results extract

then `path/to/my/results` would contain subdirectories for each patient case;
something like:

    ├── my_config.ini
    ├── INFO.log
    └── l<level>_c<centre>_p<patient>_n<node>
        ├── annotation_mask.png
        ├── normal_tissue_mask.png
        ├── patches.h5
        └── tumor_locations.<batch>.png
    └── l..._c..._p..._n...
        ...

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


#### Sampling methods

Patch extraction is based on sampling of WSI points in the interesting locations
defined by lesion annotation information.  Two methods can be used: random or linear.


##### Random sampling

This (default) is set by CL option `--method=random`. The maximum number of
samples is specified by `config[settings] : n_samples`. Since the sampling is
governed by a pseudo-random number generator, a "seed" can be set via CL
option `--seed=<int>`. By default where, `<int>`, is set to some (possibly
changing) value chosen by the operating system. Thus, in order to repeat the
same experiment, be sure to always specify the same seed through different
runs! On the other hand, by using different seeds, one can extract different
patch sets (of size `n_samples`), provided that the WSI has enough annotation
points -- otherwise, duplicated patches may come out.

Also, an adaptive "white" threshold (a pretty sensitive parameter) mechanism
is available. This is intended for tuning purposes (not recommended for
production runs) and is governed by the following configuration options:

    [settings]
    white_threshold = <float>
    white_threshold_incr = <float>
    white_threshold_max = <float>
    bad_batch_size = <int>

To see how the `white_threshold` increases during the run, debugging must be
enabled via CL option `--log-level=debug` -- the last used value is noted in
the file `INFO.log`. The mechanism can be disabled by setting
`white_threshold_incr = 0.`.

For more details, see

    pydoc lib/python2.7/defaults.py


##### Linear sampling

This method is set by CL option `--method=linear` and allows processing
**all** the WSI's annotation points. Since there can be millions of these
(especially for the normal tissues), sampling is done in batches of size
`config[settings] : n_samples`; each batch has its output stored under a
different key in the patient case's `patches.h5`. Of course, `n_samples` must
be adjusted to allow a batch to fit in RAM. As a rule of thumb, a size of 1000
should consume 7-8GB of RAM (included the loaded slide). The size of the
resulting `patches.h5` depends on how many batches are needed to scan the
whole slide: it can grow up to ~30GB. A portion of the whole annotation set
can be processed by setting the CL option `--window <MIN%> <MAX%>`, whose
bounds define, respectively, the minimum and maximum percent indices (think of
two "cursors") in the annotation set. Thus:

    cnn --method=linear --window 0 10 ...

instructs the program to process only the first 10% of annotations.

The adaptive white threshold mechanism is not available with linear sampling.


### Network training ###

**Warning! This section needs reviewing. Network training is under
development.**

Before training the NN model, an h5 patch DB must be available, either by
extracting patches from a datest with command

    cnn [OPTIONS] -c my_config.ini extract train

or by loading a pre-computed h5 DB file with command

    cnn [OPTIONS] -c my_config.ini load train

In both cases the input to the `train` step is specified in the `[load]`
section of `my_config.ini` -- Whereas output is composed of

* the model new weights stored in file `tumor_classifier.h5`,
* the training curves as png images,

all stored under directory `<results_dir>/`

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
configured with a different random generator seed. The goal is to exploit
randomness in the WSI point distribution in order to extract a different patch
set at each batch run. Here's the relevant parts of the bash script (slightly
simplified), where the random seed is set by the batch scheduler task ID in an
array of [0, N]. Note also that results are stored in different directories.

    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=1
    #SBATCH --array=0-9

    task_id=$SLURM_ARRAY_TASK_ID

    cnn --config-file=... \
        --results-dir=.../${task_id} \
        --seed=${task_id}\
        extract

Please, review the full script, then call it like this:

    some-hpc$ sbatch runner-slurm_rnd_seed

Under the hypothesis of infinite resources (i.e., available CPUs), this script
achieves linear speed-up as each task is assigned a different Slurm "node",
i.e. a CPU. No GPGPU is involved.


### <a name="#slurm_onep-onep"></a>Parallel slide processing

The file `bin/runner-slurm_onep-onep` uses a Slurm job array where each task is
assigned a different WSI (a single "patient case") while the random seed is
the same. Here's the relevant parts of the bash script (slightly simplified).
Note also that results are stored in different directories.

    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=1
    #SBATCH --array=0-<NUMBER_OF_SLIDES>

    task_id=$SLURM_ARRAY_TASK_ID

    declare -a patients
    # should have <NUMBER_OF_SLIDES> items
    patients=($(ls --color=none some-dir/with-xml-annotations/ | sed -nr 's/\.xml// p'))
    patient=${patients[${SLURM_ARRAY_TASK_ID}]}

    cnn --config-file=... \
        --results-dir=.../${task_id} \
        --patients=${patient}\
        extract

**N.B.** The the parameter `<NUMBER_OF_SLIDES>` should be set to the total
number of "patients". Since the `patients` array is built by listing the
content of a directory with Camelyon17-style XML annotation files, their total
number should be known in advance. Failing to do so will result in either
running useless tasks (more task slots than patients), or overloading the
batch system (more patients than task slots) with subsequent less-than-linear
speed-up.

Please, review the full script, then call it like this:

    some-hpc$ sbatch runner-slurm_onep-onep

Under the hypothesis of infinite resources (i.e., available CPUs), this script
achieves linear speed-up. No GPGPU is involved.


### <a name="#slurm_onep-mrnd"></a>Parallel slide processing with several random generator seeds

As a combination of the two previous techniques, the file
`bin/runner-slurm_onep-mrnd` uses a Slurm job array where each task is
assigned a different WSI (a single "patient case") while the random generator seed is
varied over M parallel subtasks. Here's the relevant parts of the bash script
(slightly simplified).  Note also that results are stored in different
directories.

    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=<NUMBER_OF_SEEDS>
    #SBATCH --array=0-<NUMBER_OF_SLIDES>

    declare -a patients
    # should have <NUMBER_OF_SLIDES> items
    patients=($(ls --color=none some-dir/with-xml-annotations/ | sed -nr 's/\.xml// p'))
    patient=${patients[${SLURM_ARRAY_TASK_ID}]}

    for s in $(seq 0 <NUMBER_OF_SEEDS>); do
        srun cnn --config-file=... \
            --results-dir=.../${SLURM_ARRAY_TASK_ID}.$s \
            --seed=${s}
            --patients=${patient}\
            extract
    done



**N.B.** The same caveat as above for `<NUMBER_OF_SLIDES>` applies.
Please, review the full script, then call it like this:

    some-hpc$ sbatch runner-slurm_onep-mrnd

Under the hypothesis of infinite resources (i.e., available CPUs), this script
achieves linear speed-up of `<NUMBER_OF_SLIDES> x <NUMBER_OF_SEEDS>`. No
GPGPU is involved.


### <a name="#slurm_onep-mrnd"></a>Parallel slide processing with several linear sampling windows

All the above techniques allow only for a fixed number `config[settings] :
n_samples` of samples. To analyze the whole slide, the batch-based linear
sampling method is applied by the script `bin/runner-slurm_onep-wind` which
uses a Slurm job array where each task is assigned a different WSI (a single
"patient case") while a window (range) of mask indices is varied over M
parallel subtasks. Here's the relevant parts of the bash script (slightly
simplified).  Note also that results are stored in different directories.

    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=<NUMBER_OF_WINDOWS>
    #SBATCH --array=0-<NUMBER_OF_SLIDES>

    declare -a patients
    # should have <NUMBER_OF_SLIDES> items
    patients=($(ls --color=none some-dir/with-xml-annotations/ | sed -nr 's/\.xml// p'))
    patient=${patients[${SLURM_ARRAY_TASK_ID}]}

    win_step=10
    win_start=0
    win_end=10
    # should loop no more than <NUMBER_OF_WINDOWS> times, 10 in this case
    while [[ $win_end -le 100 ]]; do
        srun cnn --config-file=... \
            --results-dir=.../${SLURM_ARRAY_TASK_ID}.${win_start}-${win_end} \
            --seed=0
            --patients=${patient} \
            --method=linear \
            --window ${win_start} ${win_end} \
            extract

        win_sta=$win_end
        win_end=$((win_end + win_step))
    done


**N.B.** The same caveat as above for `<NUMBER_OF_SLIDES>` and
`<NUMBER_OF_WINDOWS>` applies.  Please, review the full script, then call it
like this:

    some-hpc$ sbatch runner-slurm_onep-wind

Under the hypothesis of infinite resources (i.e., available CPUs), this script
achieves linear speed-up of `<NUMBER_OF_SLIDES> x <NUMBER_OF_WINDOWS>`. No
GPGPU is involved.

# Model Interpretability Layer
see https://github.com/maragraziani/iMIMIC-RCVs.git
