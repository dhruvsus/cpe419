#include <iostream>
#include <math.h>
#include <cstdlib>
#include <ctime>

// CUDA kernel to multiply elements of two arrays
#define A_HEIGHT 1000
#define B_WIDTH 1000
#define AB_SHARED 1000

//declare global variables
float* x;
float* y;
float* z;

__global__
void mm()
{
	int i,j,k;
	int num_rows_per_thread=A_HEIGHT/threads_available;
	int leftover_rows=A_HEIGHT%threads_available;
	int start_row=threadID*num_rows_per_thread;
	int stop_row=start_row+num_rows_per_thread;
	if(threadID==threads_available-1)
		stop_row=stop_row+leftover_rows;
	for(i=start_row;i<stop_row;i++){
		for(j=0;j<B_WIDTH;j++){
			for(k=0;k<AB_SHARED;k++){
				C[i][j]+=A[i][k]*B[k][i];
			}
		}
	}
}

int main(void)
{
	// Allocate Unified Memory -- accessible from CPU or GPU
	int N = A_HEIGHT*AB_SHARED;
	cudaMallocManaged(&x, N*sizeof(float));
	cudaMallocManaged(&y, N*sizeof(float));
	cudaMallocManaged(&z, N*sizeof(float));
	float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

	// initialize x and y arrays on the host
	for (int i = 0; i < N; i++) {
		x[i] = r;
		y[i] = r;
	}

	// Launch kernel on 1M elements on the GPU
	int blockSize = 1024;
	int numBlocks = (N + blockSize - 1) / blockSize;

	mm<<<numBlocks, blockSize>>>();

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	// Free memory
	cudaFree(x);
	cudaFree(y);
	cudaFree(z);

	return 0;
}
