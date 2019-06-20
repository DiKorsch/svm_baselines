#!/usr/bin/env bash

for dname in logs/C*; do
	svm_training=${dname}/04_svm_training.log
	l1_training=${dname}/01_L1_training.log

	if [[ ! -f $svm_training ]]; then
		continue
	fi
	if [[ ! -f $l1_training ]]; then
		continue
	fi

	C=$(echo $dname | grep -oE "[[:digit:]]e[[:digit:]]|[[:digit:]]e-[[:digit:]]|[[:digit:]]")

	accu=$(cat $svm_training | grep "Accuracy all_parts:" | grep -oE "[[:digit:]]+\\.+[[:digit:]]+")
	sparsity=$(cat $l1_training | grep "Percentage" | grep -oE "[[:digit:]]{1,2}\.[[:digit:]]{2}%")


	printf "${C}\t${accu}%%\t"
	echo ${sparsity}

done
