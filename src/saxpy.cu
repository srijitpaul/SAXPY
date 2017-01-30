#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>


__global__ void saxpy(int n, float a, float *x, float *y)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		y[i] = a*x[i] + y[i];
	}
}

double min(double* array, int size){
    // returns the minimum value of array
    double val = array[0];
    for (int i = 1; i < size; ++i){
        val = val <= array[i] ? val : array[i];
    }
    return val;
}

double max(double* array, int size){
    // returns the maximum value of array
    double val = array[0];
    for (int i = 1; i < size; ++i){
        val = val >= array[i] ? val : array[i];
    }
    return val;
}

double mean(double* array, int size) {
    double sum=0;
    for(int i=0; i<size; i++)
        sum+=array[i];
    return((double)sum/size);
}

int main(int argc, char * argv[])
{
	unsigned long int arrlength= atoi(argv[1]);
	unsigned long int N = 1<<arrlength;
	unsigned long int nruns = atoi(argv[2]);
	size_t size = N*sizeof(float);
	printf("N = %d\n",N);
	double ntime[nruns],nbandwidth[nruns];
	for(unsigned long int count = 0; count < nruns; count ++){
		// Create CUDA events for timing purposes
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		float *x, *y;		// Host vectors
		float *d_x, *d_y;	// Device vectors

		// Allocate host memory
		x = (float *)malloc(size);
		y = (float *)malloc(size);

		// Allocate device memory
		cudaMalloc(&d_x, size);
		cudaMalloc(&d_y, size);

		for (unsigned long int i = 0; i < N; i++){
			x[i] = 1.0f;
			y[i] = 2.0f;
		}

		cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

		// Perform SAXPY on 1M elements
		cudaEventRecord(start);
		saxpy<<<(N+255)/256, 256>>>(N, 2.0, d_x, d_y);
		cudaEventRecord(stop);

		cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

		cudaEventSynchronize(stop);

		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);

		float maxError = 0.0f;
		for (unsigned long int i = 0; i < N; i++){
			maxError = max(maxError, abs(y[i]-4.0f));
		}
		printf("Max error: %f\n", maxError);
		ntime[count] = milliseconds;
		nbandwidth[count] = N*4*3/milliseconds/1e6;
	        cudaFree(d_x);
		cudaFree(d_y);
		free(x);
		free(y);	
	}
	printf("Average time of execution: %f\n", mean(ntime,nruns));
	printf("Maximum time of execution: %f\n", max(ntime,nruns));
	printf("Mininum time of execution: %f\n", min(ntime,nruns));
	printf("Average Bandwidth: %f\n", mean(nbandwidth,nruns));
	printf("Maximum Bandwidth: %f\n", max(nbandwidth,nruns));
	printf("Mininum Bandwidth: %f\n", min(nbandwidth,nruns));

}
