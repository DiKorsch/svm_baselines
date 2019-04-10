#!/usr/bin/env bash

source config.sh
DATA=${DATA:-/home/korsch/Data/info.yml}

SCRIPT="../predict.py"

if [[ -z $DATASET ]]; then
	echo "DATASET variable is missing!"
	exit -1
fi

if [[ -z $WEIGHTS ]]; then
	echo "WEIGHTS variable is missing!"
	exit -1
fi

if [[ -z $SUBSET ]]; then
	echo "SUBSET variable is missing!"
	exit -1
fi

PARTS=${PARTS:-"GLOBAL"}
OUTPUT=${OUTPUT:-"${SUBSET}.csv"}

OPTS="${OPTS} --subset ${SUBSET}"
OPTS="${OPTS} --output ${OUTPUT}"

$PYTHON $SCRIPT \
	${DATA} \
	${DATASET} \
	${DATASET}_${PARTS} \
	${WEIGHTS} \
	${OPTS} \
	$@
