#!/bin/bash

nvcc -arch=sm_60 -o ../bin/saxpy_cputime cputime_saxpy.cu 

rm  -f ../measurements/cpu_timings/*.dat

mkdir -p ../measurements/cpu_timings
mkdir -p ../measurements/cpu_CUPTI


MAX_ARRAY_SIZE=25
MAX_RUNS=100
NEVAL=50
INCREMENT=10
START_ARRAY_SIZE=10
START_RUN=20
#Make Bandwidth and execution time measurements
for i in $(seq $START_ARRAY_SIZE $MAX_ARRAY_SIZE);
	do
		for j in $(seq $START_RUN $INCREMENT $MAX_RUNS); 
			do
				bsub -n 1 -I -R "select [ngpus>0] rusage[ngpus_shared=1]"  ./../bin/saxpy_cputime $i $j $NEVAL > temp.dat
                                sed -i '1d' temp.dat
                                cat temp.dat >>../measurements/cpu_timings/${i}.dat
				#bsub -n 1 -o ../measurements/cpu_CUPTI/$i.out -R "select [ngpus>0] rusage[ngpus_shared=1]"  nvprof --metrics achieved_occupancy,dram_read_throughput,dram_utilization,dram_write_throughput,l2_read_throughput,l2_write_throughput,issued_ipc ./../bin/saxpy_cputime $i $NEVAL
			done
	done



