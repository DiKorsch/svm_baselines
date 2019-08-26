#!/usr/bin/env bash

BASE_FOLDER=/home/korsch/Repos/PhD/00_L1_SVM_Parts

export N_PARTS=${N_PARTS:-4}

export DATASET=${DATASET:-CUB200}
export MODEL_TYPE=${MODEL_TYPE:-inception}

export N_JOBS=0
export SVM_OUTPUT=${BASE_FOLDER}/output/results_C0.1
export WEIGHTS=${BASE_FOLDER}/models/ft_${DATASET}_${MODEL_TYPE}.npz

LABEL_SHIFT=1
INPUT_SIZE=427

export OPTS="${OPTS} --input_size ${INPUT_SIZE}"
export OPTS="${OPTS} --label_shift ${LABEL_SHIFT}"
export OPTS="${OPTS} --prepare_type model"

./locs_from_L1_SVM.sh ${OPTS} \
	--weights ${WEIGHTS} \
	--K $N_PARTS \
	--thresh_type otsu \
