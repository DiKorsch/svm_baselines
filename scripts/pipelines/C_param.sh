#!/usr/bin/env bash

export GPU=${GPU:-0}
export BATCH_SIZE=${BATCH_SIZE:-24}
export MODEL_TYPE=${MODEL_TYPE:-inception}

export DATASET=${DATASET:-CUB200_2FOLD}
export OMP_NUM_THREADS=4

_home=$(pwd)
BASE_FOLDER=/home/korsch/Repos/PhD

SVM_FOLDER=${BASE_FOLDER}/00_SVM_Baseline
EXTRACTOR_FOLDER=${BASE_FOLDER}/02_Feature_Extraction
DATASET_FOLDER=/home/korsch/Data/DATASETS/birds/cub200_2fold

OPTS="--input_size 427 --label_shift 1 --prepare_type model"

function check_return {
	ret_code=$?

	msg=$1
	if [[ $ret_code -ne 0 ]]; then
		echo "[$ret_code] Error occured during ${msg}!"
		exit $ret_code
	fi
}

LOGDIR=$(realpath logs)
echo "Logs are saved under $LOGDIR"
if [[ ! -d ${LOGDIR} ]]; then
	mkdir -p ${LOGDIR}
fi

export WEIGHTS=${EXTRACTOR_FOLDER}/models/ft_${DATASET}_inceptionV3.npz

###############################################
# extract global features
###############################################

# export N_JOBS=3
# export OUTPUT=${DATASET_FOLDER}/features
# export PARTS=GLOBAL

# cd ${EXTRACTOR_FOLDER}/scripts
# ./extract.sh \
# 	--logfile ${LOGDIR}/00_global_extraction.log \
# 	${OPTS}

# check_return "Global Feature Extraction"

export SVM_OUTPUT=$(realpath ${SVM_FOLDER}/.grid_search)
if [[ ! -d $SVM_OUTPUT ]]; then
	echo "Creating \"${SVM_OUTPUT}\""
	mkdir -p $SVM_OUTPUT
fi

echo "Trained SVMs are saved under $SVM_OUTPUT"

for C in 5e-3 1e-2 5e-2 1e-1 5e-1 1 5 1e1 5e1 1e2 5e2 1e3 ; do

	LOGDIR=$(realpath logs/C_$C)
	if [[ ! -d ${LOGDIR} ]]; then
		mkdir -p ${LOGDIR}
	fi

	echo "Running pipeline for C=$C ..."

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
		--K 4 \
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
		-clf logreg \
		--logfile ${LOGDIR}/04_svm_training.log \
		# --no_dump

	check_return "SVM Training"

done


# go back, where the script was invoked
cd $_home
