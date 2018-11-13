//imports
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <stdlib.h>

__global__ void printSome(int i){
	printf("%d",i);
}
int main(){
	cudaStream_t streams[5];
	int i;
	for(i=0;i<5;i++){
		cudaStreamCreate(&streams[i]);
	}
	for(i=0;i<5;i++){
		printSome<<<1,1,0,streams[i]>>>(i);
	}
	for(i=0;i<5;i++){
		cudaStreamDestroy(streams[i]);
	}
	return 0;
}
