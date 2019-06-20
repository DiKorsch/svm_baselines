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
export BATCH_SIZE=${BATCH_SIZE:-12}
export MODEL_TYPE=${MODEL_TYPE:-inception}

export OMP_NUM_THREADS=2

_home=$(pwd)
BASE_FOLDER=/home/korsch/Repos/PhD

SVM_FOLDER=${BASE_FOLDER}/00_SVM_Baseline
EXTRACTOR_FOLDER=${BASE_FOLDER}/02_Feature_Extraction



C=${C:-0.1}
# N_PARTS=4

export SVM_OUTPUT=${SVM_FOLDER}/.results_C${C}
echo "Trained SVMs and features are saved under $SVM_OUTPUT"

check_dir "${SVM_OUTPUT}"


export DATA=${SVM_FOLDER}/datasets/data.yaml

echo "Pipeline starts in 10s, time for last checks ..."
sleep 10s

label_shift=1
export DATASET=CUB200
export WEIGHTS=${EXTRACTOR_FOLDER}/models2/inat.inceptionV3.ckpt.npz

DATASET_FOLDER=${SVM_FOLDER}/datasets/${DATASET}
OPTS=""

if [[ ! -d $DATASET_FOLDER ]]; then
	echo "Could not find dataset folder \"$DATASET_FOLDER\""
	continue
fi

echo "Running pipeline for \"${DATASET}\" with label_shift=${label_shift} and data folder: \"${DATASET_FOLDER}\"..."

LOGDIR=${SVM_OUTPUT}/logs/${DATASET}
check_dir $LOGDIR

OPTS="${OPTS} --no_center_crop_on_val"
OPTS="${OPTS} --input_size 427"
OPTS="${OPTS} --label_shift ${label_shift}"
OPTS="${OPTS} --prepare_type model"
#OPTS="${OPTS} --swap_channels"

###############################################
# extract part features
###############################################
# for parts in GT GT2 NAC ; do
for parts in GT GT2 NAC ; do

	export N_JOBS=3
	export OUTPUT=${DATASET_FOLDER}/features2
	export PARTS=$parts

	cd ${EXTRACTOR_FOLDER}/scripts
	./extract.sh \
		--logfile ${LOGDIR}/03_part_extraction_${parts}.log \
		${OPTS}

	check_return "Part Feature Extraction ($parts)"
	continue

	###############################################
	# train SVM on these parts
	###############################################
	export OUTPUT=${SVM_OUTPUT}
	export PARTS=$parts

	cd ${SVM_FOLDER}/scripts
	./train.sh \
		-clf svm \
		--logfile ${LOGDIR}/04_svm_training_${parts}.log \
		--eval_local_parts
		# --no_dump

	check_return "SVM Training ($parts)"
done

# go back, where the script was invoked
cd $_home
