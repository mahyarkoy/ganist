#!/bin/bash
set -e
fname=$1
target=$2
eval_step=5000
mkdir -p $target
for i in {0..2}
do
	mkdir -p $fname/run_$i
	python run_ganist.py -l $fname/run_$i -e $eval_step -s $i
done
cp -r $fname $target
cp run_ganist.py $target/$fname
cp tf_ganist.py $target/$fname
