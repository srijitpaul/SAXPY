#!/bin/bash

make clean
make

rm  -rf ../measurements/CUPTI/*.out
END=25
NEVAL=10
#Make Bandwidth and execution time measurements
for i in $(seq 10 $END); 
	do
		mkdir -p ../measurements/CUPTI
		bsub -n 1 -o ../measurements/CUPTI/$i.out -R "select [ngpus>0] rusage[ngpus_shared=1]"  ./saxpy $i $NEVAL  
		sleep 5
	done



