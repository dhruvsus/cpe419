//imports
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

//constants
#define HEIGHT 3000
#define WIDTH 3000
#define NUM_STREAMS 2
#define NUM_BLOCKS 32
#define THREADS_PER_BLOCK 128
#define CACHEAMT 0

__global__ void addMat(int *d_X, int *d_Y, int * d_Z, int numElements) {
	int myThreadID = (blockIdx.x * blockDim.x) + threadIdx.x;
	int gridStride = blockDim.x*gridDim.x;
	int i;
	for (i=myThreadID;i<numElements;i+=gridStride) {
		// do the add into d_Z.
		d_Z[i] = d_X[i] + d_Y[i];
	}
}

void matrixAddNonThreaded(int *X, int *Y, int *D, int nX, int nY){
	int i;
	for (i=0;i<nX*nY;i++){
		D[i]=X[i]+Y[i];
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
	int r=0;
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

	//streamSize: number of words for each stream to work on
	streamSize = (HEIGHT * WIDTH) / NUM_STREAMS;
	offset = 0;
	
	for(i=0;i<NUM_STREAMS;i++){
		//copy memory. this is synchronous because same stream
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

	int dif=0;
	for (i=0; i<HEIGHT*WIDTH; i++){
		dif+=abs(Z[i]-D[i]);
	}
	if(dif < 1) printf("SUCCESS\n");
	else printf("FAIL\n");
	printf("%d\n",dif);
	
	cudaFree(d_X);
	cudaFree(d_Y);
	cudaFree(d_Z);
	cudaFreeHost(X);
	cudaFreeHost(Y);
	cudaFreeHost(Z);
	free(D);
	return 0;
}
