#!/bin/bash -l
# -*- mode: shell-script -*-
################################################################################
# runner-slurm_rnd_seed: cnn runner for SLURM with random seed --
# proof-of-concept for testing parallel patch extraction as patch coordinates
# are sampled over a uniform distribution.
#
# Use a job array where each task is scheduled automatically on a (possibly)
# different node/CPU
################################################################################
# For copyright see the `LICENSE` file.
#
# This file is part of PROCESS_UC1.
################################################################################
# Slurm batch control
#SBATCH --job-name=cnn-extr_rnd_seed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=5GB
#SBATCH --time=01:00:00
#SBATCH --account=process2
#SBATCH --partition=plgrid-testing
#SBATCH --array=0-9
#SBATCH --error=%x-%A_%3a.err
#SBATCH --output=%x-%A_%3a.out
#SBATCH --mail-type=END,FAIL
################################################################################
# Job conf
# Env:
#   $HOME      == $PLG_USER_STORAGE
#   $SCRATCH   == $PLG_USER_SCRATCH  == $PLG_USER_SCRATCH_SHARED
#                 $PLG_GROUPS_STORAGE == $PLG_GROUPS_SHARED
#
# Expected experiment sandbox structure:
#
# <experiment_root>/
#     code/           symlink to the program's code
#     conf/           configuration files
#     data/           (symlink to) directory with input data
#     results/        persistent results (to be kept/archived)
#     scratch/        symlink to batch job dir where last runs results are
#                     stored.  Warning! This is temporary storage; make sure to
#                     mv important stuff to permanent storage 'results/'
################################################################################
TSTAMP=$(date '+%m%d')
PROJECT=PROCESS_UC1
PROJECT_STORAGE=${PLG_GROUPS_STORAGE}/plggprocess/UC1
EXPERIMENT_ROOT=${PROJECT_STORAGE}/experiments/cnn-patch-extraction
CONF_DIR=${EXPERIMENT_ROOT}/conf
DATA_DIR=${EXPERIMENT_ROOT}/data/camelyon17/dev_data
# Make sure this points to $SCRATCH!
SLURM_SUBMIT_DIR=${EXPERIMENT_ROOT}/scratch
RESULTS_DIR=${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}

mkdir -p $RESULTS_DIR
echo 'Directory names: <MMDD>.<slurm-job-id>.<slurm-task-id>' > ${RESULTS_DIR}/README

run_id=$(printf '%03d' $SLURM_ARRAY_TASK_ID)
srun_id=${SLURM_ARRAY_JOB_ID}.${run_id}
myself=${SLURM_JOB_NAME}.${srun_id}
results_dir=${RESULTS_DIR}/${TSTAMP}.${srun_id}

################################################################################
# modules
################################################################################
mods="
plgrid/tools/python/2.7.14
plgrid/libs/openslide/3.4.1
"

errs=0
for m in $mods; do
    module load $m || {
        echo 2>&1 "[ERROR] ${myself}: $m: can't load module"
        ((errs++))
    }
done

[[ $errs -gt 0 ]] && exit $errs

################################################################################
# Start job
################################################################################
echo >&2 "[WARN] ${myself}: some modules won't be loaded -- only for patch extraction"

PYTHONPATH=${HOME}/projects/EnhanceR/${PROJECT}/code/${PROJECT}/lib/python2.7
# Note: env unset == False, anything else == True (avoid setting '0' for False)
PROCESS_UC1__HAS_SKIMAGE_VIEW=
PROCESS_UC1__HAS_TENSORFLOW=

export PYTHONPATH PROCESS_UC1__HAS_SKIMAGE_VIEW PROCESS_UC1__HAS_TENSORFLOW

# no container support for now, take stuff under ~/myroot/bin + Git clone
# dir. Executable is on $PATH, no need to ppek in the ${EXPERIMENT_ROOT} dir
cmnd="cnn --config-file=${CONF_DIR}/config.${SLURM_JOB_NAME}.ini --results-dir=${results_dir} --seed=${SLURM_ARRAY_TASK_ID} --log-level=debug extract"

echo -e "[INFO] ${myself}: command line:\n    ${cmnd}"
eval $cmnd
