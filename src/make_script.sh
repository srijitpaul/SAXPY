#!/bin/bash

END=25
NEVAL=10
#Make Bandwidth and execution time measurements
for i in $(seq 10 $END); 
	do
		mkdir -p ../measurements/timings
		mkdir -p ../measurements/CUPTI
		bsub -n 1 -o ../measurements/timing/$i.out -R "select [ngpus>0] rusage[ngpus_shared=1]"  ./../bin/saxpy_many $i $NEVAL  
		bsub -n 1 -o ../measurements/CUPTI/$i.out -R "select [ngpus>0] rusage[ngpus_shared=1]"  nvprof --metrics achieved_occupancy,dram_read_throughput,dram_utilization,dram_write_throughput,l2_read_throughput,l2_write_throughput,issued_ipc ./../bin/saxpy_many $i $NEVAL

	done



