#!/bin/bash

nvcc -arch=sm_60 -o ../bin/saxpy_cputime cputime_saxpy.cu 
END=25
NEVAL=50
#Make Bandwidth and execution time measurements
for i in $(seq 10 $END);
	do
		for j in $(seq 10 $NEVAL); 
			do
				mkdir -p ../measurements/cpu_timings
				mkdir -p ../measurements/cpu_CUPTI
				bsub -n 1 -I -R "select [ngpus>0] rusage[ngpus_shared=1]"  ./../bin/saxpy_cputime $i $j >temp.dat 
                                sed -i '1d' temp.dat
                                cat temp.dat >>../measurements/cpu_timings/${i}.dat			

				#bsub -n 1 -o ../measurements/cpu_CUPTI/$i.out -R "select [ngpus>0] rusage[ngpus_shared=1]"  nvprof --metrics achieved_occupancy,dram_read_throughput,dram_utilization,dram_write_throughput,l2_read_throughput,l2_write_throughput,issued_ipc ./../bin/saxpy_cputime $i $NEVAL
			done
	done



