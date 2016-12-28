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

int main()
{
	int N = 1<<27;
	int size = N*sizeof(float);
	printf("N = %d\n",N);

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

	for (int i = 0; i < N; i++)
	{
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
	for (int i = 0; i < N; i++) 
	{
		maxError = max(maxError, abs(y[i]-4.0f));
	}
	printf("Max error: %f\n", maxError);
	printf("Succesfully performed SAXPY on %d elements in %f milliseconds.\n", N, milliseconds);
	printf("Effective Bandwidth (GB/s): %f\n", N*4*3/milliseconds/1e6);
}
