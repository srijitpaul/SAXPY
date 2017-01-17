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

int main(int argc, char * argv[])
{
    int arrlength= atoi(argv[1]);
    int N = 1<<arrlength;
    int nruns = atoi(argv[2]);
    int size = N*sizeof(float);
    printf("N = %d\n",N);
    char *output_file;
    output_file = new char[1024];
    output_file = argv[3];
    ofstream outfile(output_file,ios::out);
    //create stop timers
    double wall_timestop[nruns];

    float *x, *y;		// Host vectors
    float *d_x, *d_y;	// Device vectors

    // Allocate host memory
    x = (float *)malloc(size);
    y = (float *)malloc(size);


    double wall_timestart = get_wall_time();
    // Allocate device memory
    for(int count = 0; count < nruns; count ++){

    	cudaMalloc(&d_x, size);
    	cudaMalloc(&d_y, size);

    	for (int i = 0; i < N; i++){
        	x[i] = 1.0f;
        	y[i] = 2.0f;
    	}


   
        cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

        // Perform SAXPY on 1M elements
        saxpy<<<(N+255)/256, 256>>>(N, 2.0, d_x, d_y);
        cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
        wall_timestop[count] = get_wall_time();
    }
    double seconds[nruns];

    for(int i = 0; i < nruns; i++){
        seconds[i] = wall_timestop[i] - wall_timestart;
        outfile<<i+1<<"\t\t"<<seconds[i]<<endl;
    }

}
