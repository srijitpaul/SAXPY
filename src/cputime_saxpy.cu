#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
using namespace std;

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

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

int main(int argc, char * argv[])
{
    unsigned long int arrlength= atoi(argv[1]);
    unsigned long int N = 1<<arrlength;
    unsigned long int nruns = atoi(argv[2]);
    unsigned long int neval = atoi(argv[3]);
    size_t size = N*sizeof(float);
    double seconds[neval];
 

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
    saxpy<<<(N+255)/256, 256>>>(N, 2.0, d_x, d_y);
        
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);


    for(unsigned long int run = 0; run < neval; run++)
    {

    	// Allocate device memory
   

    	cudaMalloc(&d_x, size);
    	cudaMalloc(&d_y, size);

	

    	double wall_timestart = get_wall_time();

    	cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    	cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);
    
    	for(unsigned long int count = 0; count < nruns; count ++){

        	// Perform SAXPY on 1M elements
        	saxpy<<<(N+255)/256, 256>>>(N, 2.0, d_x, d_y);
    	}
        
    	cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
    	double wall_timestop = get_wall_time();

    	cudaFree(d_x);
    	cudaFree(d_y);


    	seconds[run] = wall_timestop - wall_timestart;


    }

    free(x);
    free(y);
    cout<<nruns<<"\t\t"<<min(seconds,neval)<<endl;
 
    

}
