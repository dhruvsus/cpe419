#include <iostream>
#include <math.h>
#include <cstdlib>
#include <ctime>

// CUDA kernel to multiply elements of two arrays
#define A_HEIGHT 1000
#define B_WIDTH 1000
#define AB_SHARED 1000

//declare global variables
float* A;
float* B;
float* C;
float* D;
int N;

void matrix_mult_nonthreaded(){
	int i,j,k;
	for(i=0;i<A_HEIGHT;i++){
		for(j=0;j<B_WIDTH;j++){
			for(k=0;k<AB_SHARED;k++){
				D[i*1000+j]+=A[i*1000+k]*B[k*1000+i];
			}
		}
	}
	return;
}

__global__ void mm(float* A, float* B, float* C, int N)
{
	int i,j,k;
	int threadID = blockDim.x*blockIdx.x+threadIdx.x;
	int gridStride = gridDim.x*blockDim.x;

	for(i=threadID ; i<1000; i+=gridStride){
		for(j=0;j<B_WIDTH;j++){
			for(k=0;k<AB_SHARED;k++){
				C[i*1000+j]+=A[i*1000+k]*B[k*1000+i];
			}
		}
	}
}

int main(void)
{
	// Allocate Unified Memory -- accessible from CPU or GPU
	N = A_HEIGHT*AB_SHARED;
	cudaMallocManaged(&A, N*sizeof(float));
	cudaMallocManaged(&B, N*sizeof(float));
	cudaMallocManaged(&C, N*sizeof(float));
	cudaMallocManaged(&D, N*sizeof(float));
	float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

	// initialize x and y arrays on the host
	for (int i = 0; i < N; i++) {
		A[i] = r;
		B[i] = r;
	}
	
	matrix_mult_nonthreaded();
	// Launch kernel on 4*256 threads

	mm<<<4, 1024>>>(A,B,C,N);

	int row, col;

        float dif, accum=0;
       	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();
 
	for (row=0; row<1000; row++){
                for(col=0; col<1000; col++) {
                        dif=abs(C[row*1000+col]-D[row*1000+col]);
                        //editing diff formula to account for 2d array on the stack
			//dif=abs(fc[row][col]-fcthread[row][col]);
			if(dif!=0) accum+=dif;
                }
        }
        if(accum < 10) 
	std::cout<<"SUCCESS\n";
        else 
	std::cout<<"FAIL\n";
	std::cout<<accum;


	// Free memory
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	cudaFree(D);
	return 0;
}
