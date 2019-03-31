#!/usr/bin/env bash

# OPTS="${OPTS} --visualize_coefs"

source config.sh

DATA=${DATA:-/home/korsch/Data/info.yml}

if [[ -z $DATASET ]]; then
	echo "DATASET variable is missing!"
	exit -1
fi

TRAINED_SVM="../clf_${DATASET}.GLOBAL.${MODEL_TYPE}_glob_only_sparse_coefs.npz"

$PYTHON $SCRIPT \
	${DATA} \
	${DATASET} \
	${TRAINED_SVM} \
	${OPTS} \
	$@
