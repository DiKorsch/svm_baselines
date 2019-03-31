source ${HOME}/.anaconda3/etc/profile.d/conda.sh
conda activate chainer4

PYTHON=python
SCRIPT="../part_locations_from_L1_SVM.py"

GPU=${GPU:-0}
BATCH_SIZE=${BATCH_SIZE:-4}
N_JOBS=${N_JOBS:-0}

MODEL_TYPE=${MODEL_TYPE:-inception}
PREPARE_TYPE=${PREPARE_TYPE:-model}
SUBSET=${SUBSET:-"test"}

OPTS=${OPTS:-""}
OPTS="${OPTS} --gpu $GPU"
OPTS="${OPTS} --batch_size $BATCH_SIZE"
OPTS="${OPTS} --n_jobs $N_JOBS"

OPTS="${OPTS} --model_type $MODEL_TYPE"
OPTS="${OPTS} --prepare_type $PREPARE_TYPE"
OPTS="${OPTS} --subset $SUBSET"

# OPTS="${OPTS} --scale_features"
OPTS="${OPTS} --init_from_maximas"
