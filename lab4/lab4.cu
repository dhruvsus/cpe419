//imports
#include <iostream>
#include <math.h>
#include <cstdlib>
#include <ctime>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>

//constants for dimensions of matrices
#define A_HEIGHT 1000
#define A_WIDTH 1000
#define THREADSIZE 16


//init matrix: initialize A and B with value from 0.0 to 1.0
__global__ void initMatrixGPU(float *X, float *Y, int N){
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

//matrix addition, non threaded
void matrixAddNonThreaded(float* A, float* B, float* D, int nX, int nY){
	int row, col;
	for (row=0; row<nY; row++){
		for(col=0; col<nX; col++) {
			D[row*nX+col]=A[row*nX+col]+B[row*nX+col];
		}
	}
}

//threaded across cuda enabled GPU for matrix addition
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
	//memory allocation
	float* A;
	float* B;
	float* C;
	float* D;
	int nX;
	int nY;
	nX=A_WIDTH;
	nY=A_HEIGHT;
	int deviceID;
	int N=nX*nY;

	// GPU specific variables
	cudaDeviceProp gpuProps;

	// Get GPU properties
	cudaGetDevice(&deviceID);
	cudaGetDeviceProperties(&gpuProps, deviceID);
	int numSM=gpuProps.multiProcessorCount;
	int maxThreadsPerBlock=gpuProps.maxThreadsPerBlock;
	int maxThreadsPerMultiProcessor=gpuProps.maxThreadsPerMultiProcessor;
	int maxGridSize=gpuProps.maxGridSize[0];
	int maxThreadsDim=gpuProps.maxThreadsDim[0];	

	const dim3 blockSize(THREADSIZE, THREADSIZE, 1);
	const dim3 gridSize(((A_WIDTH-1)/THREADSIZE)+1,((A_HEIGHT-1)/THREADSIZE)+1);

	// Allocate memory on unified heap and host memory
	cudaMallocManaged(&A, nX*nY*sizeof(float));
	cudaMallocManaged(&B, nX*nY*sizeof(float));
	cudaMallocManaged(&C, nX*nY*sizeof(float));
	D = (float*)malloc(N*sizeof(float));

	//current memory status, assuming >Pascal
	//A,B,C allocated on the device
	//nX, nY, deviceID allocated on the host
	//D allocated on the host, as we don't need it on the device.

	// Launch init kernel
	initMatrixGPU<<<2*numSM, 128>>>(A,B,nX*nY);
	cudaDeviceSynchronize();

	// Print GPU info
	std::cout<<"SM's "<<numSM<<", maxThreadsPerBlock "<<maxThreadsPerBlock<<", maxThreadsPerMultiProcessor "<<maxThreadsPerMultiProcessor<<" maxGridSize "<<maxGridSize<<" maxThreadsDim "<<maxThreadsDim<<'\n';
	// Launch add kernel
	matrixAddGPU<<<gridSize, blockSize>>>(A,B,C,nX,nY);
	cudaDeviceSynchronize();

	// Sequential matrix addition
	matrixAddNonThreaded(A,B,D,nX,nY);

	//sanity check
	int row, col;
	float dif=0;
	for (row=0; row<nY; row++){
		for(col=0; col<nX; col++)
			dif+=abs(C[row*nX+col]-D[row*nX+col]);
	}
	if(dif < 10) printf("SUCCESS\n");
	else printf("FAIL\n");
	printf("%f\n",dif);
	
	// Free memory
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	free(D);
	return 0;
}
