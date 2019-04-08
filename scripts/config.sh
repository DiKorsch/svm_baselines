source ${HOME}/.anaconda3/etc/profile.d/conda.sh
conda activate chainer4

PYTHON=python

MODEL_TYPE=${MODEL_TYPE:-inception}

OPTS="${OPTS} --model_type $MODEL_TYPE"
