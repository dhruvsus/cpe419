#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
/* dtime -
utility routine to return 
the current wall clock time
*/
double dtime()
{
    double tseconds = 0.0;
    struct timeval mytime;
    gettimeofday(&mytime,(struct timezone*)0);
    tseconds = (double)(mytime.tv_sec + mytime.tv_usec*1.0e-6);
    return( tseconds );
}

// CUDA kernel to multiply elements of two arrays
#define A_HEIGHT 1000
#define B_WIDTH 1000
#define AB_SHARED 1000

//declare global variables
long threads_available=4;

float* A;
float* B;
// C is the CPU implementation, D is the GPU implementation
float* C;
float* D;
int N;

void mm_cpu_serial(){
	int i,j,k;
	for(i=0;i<A_HEIGHT;i++){
		for(j=0;j<B_WIDTH;j++){
			for(k=0;k<AB_SHARED;k++){
				C[i*1000+j]+=A[i*1000+k]*B[k*1000+i];
			}
		}
	}
	return;
}

void* mm_cpu_parallel(void* id){
	int threadID=(int)id;
	int i,j,k;
	int num_rows_per_thread=A_HEIGHT/threads_available;
	int leftover_rows=A_HEIGHT%threads_available;
	int start_row=threadID*num_rows_per_thread;
	int stop_row=start_row+num_rows_per_thread;
	if(threadID==threads_available-1)
		stop_row=stop_row+leftover_rows;
	for(i=start_row;i<stop_row;i++){
		for(j=0;j<B_WIDTH;j++){
			for(k=0;k<AB_SHARED;k++){
				C[i*1000+j]+=A[i*1000+k]*B[k*1000+i];
			}
		}
	}
	return NULL;
}

__global__ void mm_gpu_global(float* A, float* B, float* C, int N)
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

__global__ void mm_gpu_shared(float* A, float* B, float* C, int N)
{}

int main(void)
{	
	double tstart,tstop,ttime;
	N = A_HEIGHT*AB_SHARED;
	float r;
	// Allocate Memory
	cudaMallocHost(&A, N*sizeof(float));
	cudaMallocHost(&B, N*sizeof(float));
	cudaMallocHost(&C, N*sizeof(float));
	cudaMalloc(&D, N*sizeof(float));

	// initialize A and B arrays on the host
	for (int i = 0; i < N; i++) {
		r= static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		A[i] = r;
		B[i] = r;
	}
	
	//multiply cpu serial
	tstart=dtime();
	mm_cpu_serial();
	tstop=dtime();
	ttime=tstop-tstart;
	//print result for non threaded
	printf("secs serial = %10.3lf\n",ttime);

	int row, col;
    float dif, accum=0;
    // Wait for GPU to finish before accessing on host
	for (row=0; row<1000; row++){
                for(col=0; col<1000; col++) {
                        dif=abs(C[row*1000+col]-D[row*1000+col]);
                        //editing diff formula to account for 2d array on the stack
			//dif=abs(fc[row][col]-fcthread[row][col]);
			if(dif!=0) accum+=dif;
                }
        }
    if(accum < 10) 
		printf("SUCCESS\n");
    else 
		printf("FAIL\n");
	printf("%lf\n",accum);


	// Free memory
	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);
	cudaFree(D);
	return 0;
}
