#!/bin/bash

END=25
NEVAL=50
#Make Bandwidth and execution time measurements
for i in $(seq 10 $END); 
	do
		mkdir -p ../measurements/cpu_timings
		mkdir -p ../measurements/cpu_CUPTI
		bsub -n 1 -R "select [ngpus>0] rusage[ngpus_shared=1]"  ./../bin/saxpy_cputime $i $NEVAL  
		#bsub -n 1 -o ../measurements/cpu_CUPTI/$i.out -R "select [ngpus>0] rusage[ngpus_shared=1]"  nvprof --metrics achieved_occupancy,dram_read_throughput,dram_utilization,dram_write_throughput,l2_read_throughput,l2_write_throughput,issued_ipc ./../bin/saxpy_cputime $i $NEVAL

	done



