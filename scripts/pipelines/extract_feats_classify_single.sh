#!/usr/bin/env bash

function check_return {
	ret_code=$?

	msg=$1
	if [[ $ret_code -ne 0 ]]; then
		echo "[$ret_code] Error occured during ${msg}!"
		exit $ret_code
	fi
}

function check_dir {
	if [[ ! -d $1 ]]; then
		echo "Creating \"${1}\""
		mkdir -p $1
	fi
}


export GPU=${GPU:-0}

export OMP_NUM_THREADS=2

_home=$(pwd)
BASE_FOLDER=/home/korsch/Repos/PhD

SVM_FOLDER=${BASE_FOLDER}/00_SVM_Baseline
EXTRACTOR_FOLDER=${BASE_FOLDER}/02_Feature_Extraction

BASE_DATA_FOLDER=${SVM_FOLDER}/datasets2
BASE_MODELS_FOLDER=${EXTRACTOR_FOLDER}/models2

C=${C:-1}
export N_JOBS=3

export SVM_OUTPUT=${SVM_FOLDER}/.results/C_${C}
echo "Trained SVMs and features are saved under $SVM_OUTPUT"

check_dir "${SVM_OUTPUT}"

# CLASSIFY_ONLY=${CLASSIFY_ONLY:-0}

error=0
if [[ -z $NAME ]]; then
	echo "NAME not set!"
	error=1
fi
if [[ -z $PART_TYPE ]]; then
	echo "PART_TYPE not set!"
	error=1
fi

if [[ $error == 1 ]]; then
	exit -1
fi

echo "Extraction pipeline for \"${NAME}\" starts in 5s, time for last checks ..."
sleep 5s

export DATASET=$NAME
export DATA=${BASE_DATA_FOLDER}/data.yaml

LOGDIR=${SVM_OUTPUT}/logs/${DATASET}
check_dir $LOGDIR

echo "Logs are saved under ${LOGDIR}"

DATASET_FOLDER=${BASE_DATA_FOLDER}/${DATASET}
OPTS=""

if [[ ! -d $DATASET_FOLDER ]]; then
	echo "Could not find dataset folder \"$DATASET_FOLDER\""
	continue
fi

if [[ $NAME == "CARS" ]]; then
	export BATCH_SIZE=${BATCH_SIZE:-24}
	export MODEL_TYPE=resnet
	INPUT_SIZE=448
	OPTS="${OPTS} --swap_channels"
else
	export BATCH_SIZE=${BATCH_SIZE:-32}
	export MODEL_TYPE=inception
	INPUT_SIZE=427
fi

if [[ $NAME == "NAB" ]]; then
	label_shift=0
else
	label_shift=1
fi

export WEIGHTS=${BASE_MODELS_FOLDER}/ft_${DATASET}_${MODEL_TYPE}.npz

# OPTS="${OPTS} --no_center_crop_on_val"
OPTS="${OPTS} --input_size ${INPUT_SIZE}"
OPTS="${OPTS} --label_shift ${label_shift}"
OPTS="${OPTS} --prepare_type model"


echo "Running pipeline for \"${DATASET}\" with data folder: \"${DATASET_FOLDER}\" and opts: \"${OPTS}\"..."


export PARTS=$PART_TYPE

###############################################
# feature extraction
###############################################

if [[ -z $CLASSIFY_ONLY ]]; then
	export OUTPUT=${DATASET_FOLDER}/features

	cd ${EXTRACTOR_FOLDER}/scripts

	./extract.sh \
		--logfile ${LOGDIR}/00_extraction_${PARTS}.log \
		${OPTS}

	check_return "Global Feature Extraction (${DATASET}: ${PARTS})"
else
	echo "==== CLASSIFY_ONLY was set. Feature extraction is skipped! ===="
fi

###############################################
# SVM classification
###############################################

export OUTPUT=${SVM_OUTPUT}

cd ${SVM_FOLDER}/scripts

./train.sh \
	-clf svm \
	--logfile ${LOGDIR}/01_svm_training_${PARTS}.log \
	--eval_local_parts \
	--no_dump

check_return "SVM Training (${DATASET}: ${PARTS})"

# go back, where the script was invoked
cd $_home
