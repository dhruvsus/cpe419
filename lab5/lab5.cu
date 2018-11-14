//imports
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

//constants
#define HEIGHT 3000
#define WIDTH 3000
#define NUM_STREAMS 10
#define NUM_BLOCKS 32
#define THREADS_PER_BLOCK 128
#define CACHEAMT 0

__global__ void addMat(int *d_X, int *d_Y, int * d_Z, int numElements) {
	int myThreadID = (blockIdx.x * blockDim.x) + threadIdx.x;
	int gridStride = blockDim.x*gridDim.x;
	int i;
	for (i=0;myThreadID<numElements;i+=gridStride) {
		// do the add into d_Z.
		d_Z[myThreadID] = d_X[myThreadID] + d_Y[myThreadID];
	}
}

void matrixAddNonThreaded(int *A, int *B, int *D, int nX, int nY){
	int row, col;
	for (row=0; row<nY; row++){
		for(col=0; col<nX; col++) {
			D[row*nX+col]=A[row*nX+col]+B[row*nX+col];
		}
	}
}

int main() {
	int i, *X, *Y, *Z, *D, *d_X, *d_Y, *d_Z, offset, streamSize;
	//clean GPU
	cudaDeviceReset();
	//create streams
	cudaStream_t streams[NUM_STREAMS];
	
	for (i = 0; i<NUM_STREAMS; i++){
		cudaStreamCreate(&streams[i]);
	}

	//allocate matrix X and Y on host
	cudaMallocHost(&X, HEIGHT * WIDTH * sizeof(int));
	cudaMallocHost(&Y, HEIGHT * WIDTH * sizeof(int));
	cudaMallocHost(&Z, HEIGHT*WIDTH*sizeof(int));
	D=(int*)(malloc(HEIGHT*WIDTH*sizeof(int)));

	//initialize X and Y
	srand(time(NULL));
	int r=rand()%100;
	for (int i = 0; i < HEIGHT*WIDTH; i++) {
		X[i] = r;
		Y[i] = r;
		r=rand();
	}

	//allocate device versions of X and Y
	cudaMalloc(&d_X, HEIGHT * WIDTH * sizeof(int));
	cudaMalloc(&d_Y, HEIGHT * WIDTH * sizeof(int));
	
	//allocate matrix d_Z on the device
	cudaMalloc(&d_Z, HEIGHT * WIDTH * sizeof(int));

	// for each stream, copy and add
	// copy first part of X and Y
	// this is sequential because on the default stream
	// maybe use 2 streams, async and synchronize the streams
	streamSize = (HEIGHT * WIDTH) / NUM_STREAMS;
	offset = 0;
	// hopefully NUM_STREAMS>=2
	//cudaMemcpyAsync(&d_X[offset], &X[offset], streamSize, cudaMemcpyHostToDevice, streams[0]);
	//cudaMemcpyAsync(&d_Y[offset], &Y[offset], streamSize, cudaMemcpyHostToDevice, streams[1]);
	//cudaStreamSynchronize(streams[0]);
	//cudaStreamSynchronize(streams[1]);
	for(i=0;i<NUM_STREAMS;i++){
		//copy memory
		cudaMemcpyAsync(&d_X[offset], &X[offset], streamSize*sizeof(int), cudaMemcpyHostToDevice, streams[i]);
		cudaMemcpyAsync(&d_Y[offset], &Y[offset], streamSize*sizeof(int), cudaMemcpyHostToDevice, streams[i]);
		addMat<<<NUM_BLOCKS, THREADS_PER_BLOCK, CACHEAMT, streams[i]>>>(&d_X[offset], &d_Y[offset], &d_Z[offset], streamSize);
		offset+=streamSize;
	}
	
	//synchronize
	cudaDeviceSynchronize();
	for (i = 0; i < NUM_STREAMS; i++){
		cudaStreamDestroy( streams[i]);
	}
	
	//copy d_Z to host
	cudaMemcpy(&Z, &d_Z, HEIGHT*WIDTH*sizeof(int), cudaMemcpyDeviceToHost);
	
	//host only matrix addition
	matrixAddNonThreaded(X,Y,D,WIDTH,HEIGHT);

	int row, col;
	float dif=0;
	for (row=0; row<HEIGHT; row++){
		for(col=0; col<WIDTH; col++)
			dif+=abs(Z[row*WIDTH+col]-D[row*WIDTH+col]);
	}
	if(dif < 1) printf("SUCCESS\n");
	else printf("FAIL\n");
	printf("%f\n",dif);

	cudaFree(d_X);
	cudaFree(d_Y);
	cudaFree(d_Z);
	cudaFreeHost(X);
	cudaFreeHost(Y);
	cudaFreeHost(Z);
	free(D);
	return 0;
}
