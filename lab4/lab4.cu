//imports
#include <iostream>
#include <math.h>
#include <cstdlib>
#include <ctime>
#include <curand.h>
#include <curand_kernel.h>

//constants for dimensions of matrices
#define A_HEIGHT 1000
#define B_WIDTH 1000
#define AB_SHARED 1000

//init matrix: initialize A and B with value from 0.0 to 1.0
__global__ void init_matrix(float *X, float *Y, int N){
	int i;
	float r;
	int threadID=blockDim.x*blockIdx.x+threadIdx.x;
	int gridStride=gridDim.x*blockDim.x;
	for(i=threadID;i<N;i+=gridStride){
		r=1.12f;
		//r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		X[i] = r;
		Y[i] = r;
	}
}

//threaded across cuda enabled GPU for matrix multiplication
__global__ void matrixMult(float* A, float* B, float* C, int N)
{
	int i,j,k;
	int threadID = blockDim.x*blockIdx.x+threadIdx.x;
	int gridStride = gridDim.x*blockDim.x;
	//each thread does work[thread] and work[thread+gridStride]
	//until thread+GridStride<N because N >> #threads, so each
	//thread does more work
	//rows of M1
	for(i=threadID ; i<A_HEIGHT; i+=gridStride){
		//columsn of M2
		for(j=0;j<B_WIDTH;j++){
			//columns of M1 = rows of M2
			for(k=0;k<AB_SHARED;k++){
				C[i*AB_SHARED+j]+=A[i*AB_SHARED+k]*B[k*AB_SHARED+i];
			}
		}
	}
}

//threaded across cuda enabled GPU for matrix addition
__global__ void matrixAdd(float* A, float* B, float* C, int N)
{	
	int i, j;
	int xLoc=blockDim.x*blockIdx.x+threadIdx.x;
	int yLoc=blockDim.y*blockIdx.y+threadIdx.y;
	int gridStrideX=blockDim.x*gridDim.x;
	int gridStrideY=blockDim.y*gridDim.y;

	for(i=xLoc;i<N;i+=gridStrideX){
		for(j=yLoc;j<N;j+=gridStrideY){
			C[i][j]=A[i][j]+B[i][j];
		}
	}
}

int main(void)
{	
	//memory allocation
	//host:
	float* A;
	float* B;
	float* C;
	int N;
	N = A_HEIGHT*AB_SHARED;
	int deviceID;
	//GPU specific variables
	cudaDeviceProp gpuProps;
	
	//get GPU properties
	cudaGetDevice(&deviceID);
	cudaGetDeviceProperties(&gpuProps, deviceID);

	int numSM=gpuProps.multiProcessorCount;
	int maxThreadsPerBlock=gpuProps.maxThreadsPerBlock;
	int maxThreadsPerMultiProcessor=gpuProps.maxThreadsPerMultiProcessor;
	int maxGridSize=gpuProps.maxGridSize[0];
	int maxThreadsDim=gpuProps.maxThreadsDim[0];	
	
	//unified: on maxwell, this get allocated on the GPU
	cudaMallocManaged(&A, N*sizeof(float));
	cudaMallocManaged(&B, N*sizeof(float));
	cudaMallocManaged(&C, N*sizeof(float));
	
	//initialize A and B on the GPU
	init_matrix<<<2*numSM, 128>>>(A,B,N);

	//prefetch A and B to CPU
	cudaMemPrefetchAsync(&A, N*sizeof(float), cudaCpuDeviceId);
	cudaMemPrefetchAsync(&B, N*sizeof(float), cudaCpuDeviceId);

	//prefetch A, B, and C to GPU
	cudaMemPrefetchAsync(&A, N*sizeof(float), deviceID);
	cudaMemPrefetchAsync(&B, N*sizeof(float), deviceID);
	cudaMemPrefetchAsync(&C, N*sizeof(float), deviceID);

	std::cout<<"SM's "<<numSM<<", maxThreadsPerBlock "<<maxThreadsPerBlock<<", maxThreadsPerMultiProcessor "<<maxThreadsPerMultiProcessor<<" maxGridSize "<<maxGridSize<<" maxThreadsDim "<<maxThreadsDim;
	// Launch kernel
	matrixAdd<<<2*numSM, 128>>>(A,B,C,N);
	
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
