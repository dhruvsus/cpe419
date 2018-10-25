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

//matrix addition, non threaded
void matrixAddNonThreaded(float* A, float* B, float* D, int nX, int nY){
        int row, col;
        for (row=0; row<nY; row++){
                for(col=0; col<nX; col++) {
                        D[row*nX+col]=A[row*nX+col]+B[row*nX+col];
                }
        }
}

/*
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
*/
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

	const dim3 blockSize(THREADSIZE, THREADSIZE, 1);
	const dim3 gridSize(((A_WIDTH-1)/THREADSIZE)+1,((A_HEIGHT-1)/THREADSIZE)+1);
	
	//unified: on maxwell, this get allocated on the GPU
	cudaMallocManaged(&A, nX*nY*sizeof(float));
	cudaMallocManaged(&B, nX*nY*sizeof(float));
	cudaMallocManaged(&C, nX*nY*sizeof(float));
	D = (float*)malloc(N*sizeof(float));

	//current memory status, assuming <Pascal
	//A,B,C allocated on the device
	//nX, nY, deviceID allocated on the host
	//D allocated on the host, as we don't need it on the device.

	//initialize A and B on the GPU
	init_matrix<<<2*numSM, 128>>>(A,B,nX*nY);
	
	/*
	//prefetch A and B to CPU
	cudaMemPrefetchAsync(&A, N*sizeof(float), cudaCpuDeviceId);
	cudaMemPrefetchAsync(&B, N*sizeof(float), cudaCpuDeviceId);

	//prefetch A, B, and C to GPU
	cudaMemPrefetchAsync(&A, N*sizeof(float), deviceID);
	cudaMemPrefetchAsync(&B, N*sizeof(float), deviceID);
	cudaMemPrefetchAsync(&C, N*sizeof(float), deviceID);
	*/
	
	std::cout<<"SM's "<<numSM<<", maxThreadsPerBlock "<<maxThreadsPerBlock<<", maxThreadsPerMultiProcessor "<<maxThreadsPerMultiProcessor<<" maxGridSize "<<maxGridSize<<" maxThreadsDim "<<maxThreadsDim<<'\n';
	// Launch kernel
	matrixAddGPU<<<gridSize, blockSize>>>(A,B,C,nX,nY);
	
	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	//fetch C to CPU
	//cudaMemPrefetchAsync(&C, N*sizeof(float), cudaCpuDeviceId);

	// Free memory
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	free(D);
	return 0;
}
