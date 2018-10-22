//imports
#include <iostream>
#include <math.h>
#include <cstdlib>
#include <ctime>

//constants for dimensions of matrices
#define A_HEIGHT 1000
#define B_WIDTH 1000
#define AB_SHARED 1000

//declare global variables
float* A;
float* B;
float* C;
int N;

//threaded across cuda enabled GPU for matrix multiplication
__global__ void matrix_mult_threaded(float* A, float* B, float* C, int N)
{
	int i,j,k;
	int threadID = blockDim.x*blockIdx.x+threadIdx.x;
	int gridStride = gridDim.x*blockDim.x;
	//each thread does work[thread] and work[thread+gridStride]
	//until thread+GridStride<N because N >> #threads, so each
	//thread does more work
	//rows of M1
	for(i=threadID ; i<1000; i+=gridStride){
		//columsn of M2
		for(j=0;j<B_WIDTH;j++){
			//columns of M1 = rows of M2
			for(k=0;k<AB_SHARED;k++){
				C[i*1000+j]+=A[i*1000+k]*B[k*1000+i];
			}
		}
	}
}

int main(void)
{	
	//memory allocation
	//host:
	N = A_HEIGHT*AB_SHARED;
	float r;
	int deviceID;
	//GPU specific variables
	cudaDeviceProp gpuProps;
	//get GPU properties
	cudaGetDevice(&deviceID);
	cudaGetDeviceProperties(&gpuProps, deviceID);
	int numSM=gpuProps.multiProcessorCount;
	int maxThreadsPerBlock=gpuProps.maxThreadsPerBlock;
	int maxThreadsPerMultiProcessor=gpuProps.maxThreadsPerMultiProcessor;
	//unified:
	cudaMallocManaged(&A, N*sizeof(float));
	cudaMallocManaged(&B, N*sizeof(float));
	cudaMallocManaged(&C, N*sizeof(float));
	
	//prefetch A and B to CPU
	cudaMemPrefetchAsync(&A, N*sizeof(float), cudaCpuDeviceId);
	cudaMemPrefetchAsync(&B, N*sizeof(float), cudaCpuDeviceId);


	//Initialize A and B with random values between 0 and 1.0
	for (int i = 0; i < N; i++) {
		r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		A[i] = r;
		B[i] = r;	
	}
		

	//prefetch A, B, and C to GPU
	cudaMemPrefetchAsync(&A, N*sizeof(float), deviceID);
	cudaMemPrefetchAsync(&B, N*sizeof(float), deviceID);
	cudaMemPrefetchAsync(&C, N*sizeof(float), deviceID);

	std::cout<<"SM's "<<numSM<<", maxThreadsPerBlock "<<maxThreadsPerBlock<<", maxThreadsPerMultiProcessor "<<maxThreadsPerMultiProcessor;
	// Launch kernel
	matrix_mult_threaded<<<2*numSM, 128>>>(A,B,C,N);
	
	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();
	
	//fetch C to CPU
	cudaMemPrefetchAsync(&C, N*sizeof(float), cudaCpuDeviceId);

	// Free memory
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	
	return 0;
}
