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
 
    //create stop timers
    double wall_timestop_1, wall_timestop_2, wall_timestop_3;

    float *x, *y;		// Host vectors
    float *d_x, *d_y;	// Device vectors

    // Allocate host memory
    x = (float *)malloc(size);
    y = (float *)malloc(size);


  
    // Allocate device memory
   

    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);

    for (int i = 0; i < N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    
    double wall_timestart_1 = get_wall_time();

    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);
    
    for(int count = 0; count < nruns; count ++){

        // Perform SAXPY on 1M elements
        saxpy<<<(N+255)/256, 256>>>(N, 2.0, d_x, d_y);
    }
        
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
    wall_timestop_1 = get_wall_time();

    double seconds_1;


    seconds_1 = wall_timestop_1 - wall_timestart_1;

    double wall_timestart_2 = get_wall_time();

    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);
    
    for(int count = 0; count < nruns; count ++){

        // Perform SAXPY on 1M elements
        saxpy<<<(N+255)/256, 256>>>(N, 2.0, d_x, d_y);
    }
        
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
    wall_timestop_2 = get_wall_time();

    double seconds_2;


    seconds_2 = wall_timestop_2 - wall_timestart_2;

    double wall_timestart_3 = get_wall_time();

    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);
    
    for(int count = 0; count < nruns; count ++){

        // Perform SAXPY on 1M elements
        saxpy<<<(N+255)/256, 256>>>(N, 2.0, d_x, d_y);
    }
        
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
    wall_timestop_3 = get_wall_time();

    double seconds_3;


    seconds_3 = wall_timestop_3 - wall_timestart_3;

    double mean_seconds = (seconds_1 + seconds_2 + seconds_3)/3;

    cout<<nruns<<"\t\t"<<mean_seconds<<"\t\t"<<sqrt((pow((seconds_1 - mean_seconds),2) + pow((seconds_2 - mean_seconds),2) + pow((seconds_3 - mean_seconds),2))/3)<<endl;

}
