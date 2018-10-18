#include <iostream>
#include <math.h>
#include <cstdlib>
#include <ctime>

// CUDA kernel to multiply elements of two arrays
#define A_HEIGHT 1000
#define B_WIDTH 1000
#define AB_SHARED 1000

//declare global variables
float* A;
float* B;
float* B;

__global__
void mm()
{
	int i,j,k;
	int threadID = blockDim.x*gridDim.x+threadIdx.x;
	int gridStride = gridDim.x*blockDim.x;
	
	int start_row=threadID*num_rows_per_thread;
	int stop_row=start_row+num_rows_per_thread;

	for(i=threadID ; i<N; i+=gridStride){
		for(j=0;j<B_WIDTH;j++){
			for(k=0;k<AB_SHARED;k++){
				C[i*1000+j]+=A[k*1000+j]*B[i*1000+k];
			}
		}
	}
}

int main(void)
{
	// Allocate Unified Memory -- accessible from CPU or GPU
	int N = A_HEIGHT*AB_SHARED;
	cudaMallocManaged(&A, N*sizeof(float));
	cudaMallocManaged(&B, N*sizeof(float));
	cudaMallocManaged(&C, N*sizeof(float));
	float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

	// initialize x and y arrays on the host
	for (int i = 0; i < N; i++) {
		x[i] = r;
		y[i] = r;
	}

	// Launch kernel on 4*256 threads
	
	mm<<<4, 1024>>>();

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	// Free memory
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);

	return 0;
}
