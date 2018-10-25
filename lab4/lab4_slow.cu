//imports
#include <iostream>
#include <math.h>
#include <cstdlib>
#include <ctime>
#include <curand.h>
#include <curand_kernel.h>

//constants for dimensions of matrices
#define A_HEIGHT 1000
#define A_WIDTH 1000
#define THREADSIZE 16

__global__ void matrixAddGPU(float* A, float* B, float* C, int nX, int nY)
{	
	int i, j;
	int xLoc=blockDim.x*blockIdx.x+threadIdx.x;
	int yLoc=blockDim.y*blockIdx.y+threadIdx.y;
	int gridStrideX=blockDim.x*gridDim.x;
	int gridStrideY=blockDim.y*gridDim.y;

	for(i=xLoc;i<nX;i+=gridStrideX){
		for(j=yLoc;j<nY;j+=gridStrideY){
			C[i*nX+j]=A[i*nX+j]+B[i*nX+j];
		}
	}
}

int main(void)
{	
cudaDeviceProp gpuProps;
	//memory allocation
	float* A;
	float* B;
	float* C;
	float r;
	int nX;
	int nY;
	nX=A_WIDTH;
	nY=A_HEIGHT;
	int deviceID;
	int N=nX*nY;

	//GPU specific variables
	//get GPU properties
	cudaGetDevice(&deviceID);
	cudaGetDeviceProperties(&gpuProps, deviceID);
	
	int numSM=gpuProps.multiProcessorCount;
	int maxThreadsPerBlock=gpuProps.maxThreadsPerBlock;
	int maxThreadsPerMultiProcessor=gpuProps.maxThreadsPerMultiProcessor;
	int maxGridSize=gpuProps.maxGridSize[0];
	int maxThreadsDim=gpuProps.maxThreadsDim[0];	

	//unified:
	cudaMallocManaged(&A, N*sizeof(float));
	cudaMallocManaged(&B, N*sizeof(float));
	cudaMallocManaged(&C, N*sizeof(float));
	
	//prefetch A and B to CPU
	//cudaMemPrefetchAsync(&A, N*sizeof(float), cudaCpuDeviceId);
	//cudaMemPrefetchAsync(&B, N*sizeof(float), cudaCpuDeviceId);


	//Initialize A and B with random values between 0 and 1.0
	for (int i = 0; i < N; i++) {
		r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		A[i] = r;
		B[i] = r;	
	}
		

	//prefetch A, B, and C to GPU
	//cudaMemPrefetchAsync(&A, N*sizeof(float), deviceID);
	//cudaMemPrefetchAsync(&B, N*sizeof(float), deviceID);
	//cudaMemPrefetchAsync(&C, N*sizeof(float), deviceID);

	std::cout<<"SM's "<<numSM<<", maxThreadsPerBlock "<<maxThreadsPerBlock<<", maxThreadsPerMultiProcessor "<<maxThreadsPerMultiProcessor<<" maxGridSize "<<maxGridSize<<" maxThreadsDim "<<maxThreadsDim<<'\n';
	// Launch kernel
	matrixAddGPU<<<2*numSM, 128>>>(A,B,C,nX, nY);
	
	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();
	
	//fetch C to CPU
	//cudaMemPrefetchAsync(&C, N*sizeof(float), cudaCpuDeviceId);

	// Free memory
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	
	return 0;
}

