#!/usr/bin/env bash

source config.sh
DATA=${DATA:-/home/korsch/Data/info.yml}

SCRIPT="../train.py"

if [[ -z $DATASET ]]; then
	echo "DATASET variable is missing!"
	exit -1
fi

PARTS=${PARTS:-"GLOBAL"}
OUTPUT=${OUTPUT:-"../.out"}

# OPTS="${OPTS} --show_feature_stats"
OPTS="${OPTS} --output ${OUTPUT}"

$PYTHON $SCRIPT \
	${DATA} \
	${DATASET} \
	${DATASET}_${PARTS} \
	${OPTS} \
	$@
