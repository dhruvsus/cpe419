//imports
#include <stdlib.h>
#include <stdio.h>
//constants
#define HEIGHT 3000
#define WIDTH 3000
#define NUM_STREAMS 10
#define NUM_BLOCKS 32
#define THREADS_PER_BLOCK 128
#define CACHEAMT 0

__global__ void addMat(int * X, int * Y, int * Z, int numElements, int offset) {
  int myThreadID = blockIdx.x * blockDim.x + threadIdx.x;
  if (myThreadID > offset && myThreadID < offset + numElements) {
    // do the add into Z.
    Z[myThreadID] = X[myThreadID] + Y[myThreadID];
  }
}
int main() {
  int i, * X, * Y, * d_X, * d_Y, * Z, offset, streamSize;
  //create streams
  cudaStream_t streams[NUM_STREAMS];
  for (i = 0; i<NUM_STREAMS; i++){
    cudaStreamCreate(&streams[i]);
  }

//allocate matrix X and Y on host
cudaMallocHost((void ** ) & X, HEIGHT * WIDTH * sizeof(int));
cudaMallocHost((void ** ) & Y, HEIGHT * WIDTH * sizeof(int));

//allocate device versions of X and Y
cudaMalloc((void ** ) & d_X, HEIGHT * WIDTH * sizeof(int));
cudaMalloc((void ** ) & d_Y, HEIGHT * WIDTH * sizeof(int));
//allocate matrix Z on the device
cudaMalloc( & Z, HEIGHT * WIDTH * sizeof(int));

// for each stream, copy and add
// copy first part of X and Y
// this is sequential because on the default stream
// maybe use 2 streams, async and synchronize the streams
streamSize = (HEIGHT * WIDTH) / NUM_STREAMS;
offset = 0;
// hopefully NUM_STREAMS>=2
cudaMemcpyAsync( (void *)&d_X[offset], (void *)X[offset], streamSize, cudaMemcpyHostToDevice, streams[0]);
cudaMemcpyAsync( (void *)&d_Y[offset], (void *)Y[offset], streamSize, cudaMemcpyHostToDevice, streams[1]);
cudaStreamSynchronize(streams[0]);
cudaStreamSynchronize(streams[1]);
i = 0;
while (i < NUM_STREAMS) {
  // do the add async
  addMat <<< NUM_BLOCKS, THREADS_PER_BLOCK, CACHEAMT, streams[i] >>> (d_X, d_Y, Z, streamSize, offset);
  if (i < NUM_BLOCKS - 1) {
    offset = (i + 1) * streamSize;
    // copy next segment async but be careful not to overwrite
    cudaMemcpyAsync( (void *)&d_X[offset], (void *)X[offset], streamSize, cudaMemcpyHostToDevice, streams[i]);
    cudaMemcpyAsync( (void *)&d_Y[offset], (void *)Y[offset], streamSize, cudaMemcpyHostToDevice, streams[i]);
    i++;
    cudaStreamSynchronize(streams[i]);
  }
}
for (i = 0; i < NUM_STREAMS; i++){
    cudaStreamDestroy( streams[i]);
  }
return 0;
}