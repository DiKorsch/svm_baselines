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
export BATCH_SIZE=${BATCH_SIZE:-24}
export MODEL_TYPE=${MODEL_TYPE:-inception}

export OMP_NUM_THREADS=2

_home=$(pwd)
BASE_FOLDER=/home/korsch/Repos/PhD

SVM_FOLDER=${BASE_FOLDER}/00_SVM_Baseline
EXTRACTOR_FOLDER=${BASE_FOLDER}/02_Feature_Extraction



C=${C:-0.1}
N_PARTS=4

export SVM_OUTPUT=${SVM_FOLDER}/.results_C${C}
echo "Trained SVMs and features are saved under $SVM_OUTPUT"

check_dir "${SVM_OUTPUT}"


export DATA=${SVM_FOLDER}/datasets/data.yaml

echo "Pipeline starts in 10s, time for last checks ..."
sleep 10s

for NAME in CUB200 NAB FLOWERS CARS ; do

	export DATASET=$NAME
	export WEIGHTS=${EXTRACTOR_FOLDER}/models/ft_${DATASET}_inceptionV3.npz

	DATASET_FOLDER=${SVM_FOLDER}/datasets/${DATASET}
	OPTS=""

	if [[ ! -d $DATASET_FOLDER ]]; then
		echo "Could not find dataset folder \"$DATASET_FOLDER\""
		continue
	fi

	if [[ $NAME == "CUB200" ]]; then
		label_shift=1
	elif [[ $NAME == "NAB" ]]; then
		label_shift=0
	elif [[ $NAME == "CARS" ]]; then
		label_shift=1
	elif [[ $NAME == "FLOWERS" ]]; then
		label_shift=1
	else
		echo "Unknown dataset: $NAME"
		exit 1
	fi


	echo "Running pipeline for \"${DATASET}\" with label_shift=${label_shift} and data folder: \"${DATASET_FOLDER}\"..."

	LOGDIR=${SVM_OUTPUT}/logs/${DATASET}
	check_dir $LOGDIR

	OPTS="${OPTS} --no_center_crop_on_val"
	OPTS="${OPTS} --input_size 427"
	OPTS="${OPTS} --label_shift ${label_shift}"
	OPTS="${OPTS} --prepare_type model"
	OPTS="${OPTS} --swap_channels"

	###############################################
	# extract global features
	###############################################

	export N_JOBS=3
	export OUTPUT=${DATASET_FOLDER}/features
	export PARTS=GLOBAL

	cd ${EXTRACTOR_FOLDER}/scripts
	./extract.sh \
		--logfile ${LOGDIR}/00_global_extraction.log \
		${OPTS}

	check_return "Global Feature Extraction"

	###############################################
	# Train L1 SVM
	###############################################

	export OUTPUT=${SVM_OUTPUT}
	export PARTS=GLOBAL

	cd ${SVM_FOLDER}/scripts
	./train.sh \
		--sparse \
		--C $C \
		--logfile ${LOGDIR}/01_L1_training.log

	check_return "L1 Training"

	###############################################
	# extract L1-SVM parts
	###############################################
	export N_JOBS=0

	cd ${SVM_FOLDER}/scripts
	./locs_from_L1_SVM.sh ${OPTS} \
		--logfile ${LOGDIR}/02_part_estimation.log \
		--weights ${WEIGHTS} \
		--topk 1 \
		--K $N_PARTS \
		--thresh_type otsu \
		--extract \
			${DATASET_FOLDER}/L1_pred/parts/part_locs.txt \
			${DATASET_FOLDER}/L1_full/parts/part_locs.txt

	check_return "Part Estimation"

	###############################################
	# extract part features
	###############################################
	export N_JOBS=3
	export OUTPUT=${DATASET_FOLDER}/features
	export PARTS=L1_pred

	cd ${EXTRACTOR_FOLDER}/scripts
	./extract.sh \
		--logfile ${LOGDIR}/03_part_extraction.log \
		${OPTS}

	check_return "Part Feature Extraction"

	###############################################
	# train SVM on these parts
	###############################################
	export OUTPUT=${SVM_OUTPUT}
	export PARTS=L1_pred

	cd ${SVM_FOLDER}/scripts
	./train.sh \
		-clf svm \
		--logfile ${LOGDIR}/04_svm_training.log \
		# --no_dump

	check_return "SVM Training"

done


# go back, where the script was invoked
cd $_home
