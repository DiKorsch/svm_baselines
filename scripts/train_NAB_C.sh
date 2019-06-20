#!/usr/bin/env bash
PYTHON=python

export DATA=../datasets/data.yaml
export PARTS=GLOBAL
export DATASET=NAB

for C in 100 10 1 0.1 0.01; do
	./train.sh \
		-clf logreg \
		--C $C \
		--no_dump \
		--logfile logs/NAB/C_${C}.log
done

