#!/usr/bin/env bash
PYTHON=python

GPU=${GPU:-0}

export DATA=${DATA:-${HOME}/Data/info.yml}
DATASET=${DATASET:-CUB200}
# NAC GT GT2 L1_pred L1_full
PARTS=${PARTS:-GT2}
WEIGHTS="${HOME}/Data/MODELS/inception/ft_CUB200/rmsprop.g_avg_pooling/model.inat.ckpt/model_final.npz"

OPTS="${OPTS} --input_size 299"
OPTS="${OPTS} --gpu ${GPU}"
OPTS="${OPTS} --load ${WEIGHTS}"

#export OMP_NUM_THREADS=2

$PYTHON ../main.py \
	${DATA} \
	${DATASET} \
	${DATASET}_${PARTS} \
	${OPTS} \
	$@
