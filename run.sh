#!/bin/bash
set -e
fname=$1
target=$2
eval_step=20000
mkdir -p $target
for i in {0..2}
do
	mkdir -p $fname/run_$i
	python fig_draw.py -l $fname/run_$i -s $i
	#python run_ganist.py -l $fname/run_$i -e $eval_step -s $i -g 0
done
cp -r $fname $target
#cp run_ganist.py $target/$fname
#cp tf_ganist.py $target/$fname
