//imports
#include <iostream>
#include <math.h>
#include <cstdlib>
#include <ctime>
#include <chrono>

//constants for dimensions of matrices
#define A_HEIGHT 1000
#define B_WIDTH 1000
#define AB_SHARED 1000

//declare global variables
float* A;
float* B;
float* C;
float* D;
int N;

//non-threaded matrix multiplication for sanity checking
void matrix_mult_nonthreaded(){
	int i,j,k;
	//rows of M1
	for(i=0;i<A_HEIGHT;i++){
		//columns of M2
		for(j=0;j<B_WIDTH;j++){
			//columns of M1 = rows of M2
			for(k=0;k<AB_SHARED;k++){
				D[i*1000+j]+=A[i*1000+k]*B[k*1000+i];
			}
		}
	}
	//void return
	return;
}

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
	D = (float*)malloc(N*sizeof(float));
	float r;
	int row, col, devideID;
	float dif=0;
	//GPU specific variables
	cudaDeviceProp gpuProps;
	//get GPU properties
	cudaGetDevice(&deviceID);
	cudaGetDeviceProperties(&gpuProps, deviceID);

	//unified:
	cudaMallocManaged(&A, N*sizeof(float));
	cudaMallocManaged(&B, N*sizeof(float));
	cudaMallocManaged(&C, N*sizeof(float));
	
	//Initialize A and B with random values between 0 and 1.0
	for (int i = 0; i < N; i++) {
		r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		A[i] = r;
		B[i] = r;	
	}
	
	//carry out non threaded matrix mulitiplication. D=AXB	
	auto tStart=std::chrono::high_resolution_clock::now();
	matrix_mult_nonthreaded();
	auto tStop=std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> tTime=tStop-tStart;
	std::cout<<tTime.count()<<" seconds\n";
	
	tStart=std::chrono::system_clock::now();
	// Launch kernel on 4*256 threads
	matrix_mult_threaded<<<4, 1024>>>(A,B,C,N);
	
	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();
	tStop=std::chrono::system_clock::now();
	tTime=tStop-tStart;
	std::cout<<tTime.count()<<" seconds\n";
 
	//sanity check
	//make sure results of threaded and non threaded multiplication are the same
       	for (row=0; row<1000; row++){
                for(col=0; col<1000; col++) {
                        dif+=abs(C[row*1000+col]-D[row*1000+col]);
                }
        }
        if(dif < 10) 
	std::cout<<"SUCCESS\n";
        else 
	{
		std::cout<<"FAIL\n";
		std::cout<<dif;
	}
	// Free memory
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	free(D);

	return 0;
}
