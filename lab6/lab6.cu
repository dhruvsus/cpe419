#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
/* dtime -
utility routine to return 
the current wall clock time
*/
double dtime()
{
	double tseconds = 0.0;
	struct timeval mytime;
	gettimeofday(&mytime, (struct timezone *)0);
	tseconds = (double)(mytime.tv_sec + mytime.tv_usec * 1.0e-6);
	return (tseconds);
}

// CUDA kernel to multiply elements of two arrays
#define A_HEIGHT 1000
#define B_WIDTH 1000
#define AB_SHARED 1000
#define NTHREADS_X 32
#define NTHREADS_Y 32
#define THREADS_PER_BLOCK NTHREADS_X * NTHREADS_Y

//declare global variables
float *A;
float *B;
// C is the CPU implementation, D is the GPU implementation
float *C;
float *D;
int N;

void mm_cpu_serial()
{
	int i, j, k;
	for (i = 0; i < A_HEIGHT; i++)
	{
		for (j = 0; j < B_WIDTH; j++)
		{
			for (k = 0; k < AB_SHARED; k++)
			{
				C[i * AB_SHARED + j] += A[i * AB_SHARED + k] * B[k * B_WIDTH + i];
			}
		}
	}
	return;
}

__global__ void mm_gpu_global(float *A, float *B, float *C, int N)
{
	int i, j, k;
	int threadID = blockDim.x * blockIdx.x + threadIdx.x;
	int gridStride = gridDim.x * blockDim.x;

	for (i = threadID; i < 1000; i += gridStride)
	{
		for (j = 0; j < B_WIDTH; j++)
		{
			for (k = 0; k < AB_SHARED; k++)
			{
				C[i * AB_SHARED + j] += A[i * AB_SHARED + k] * B[k * B_WIDTH + i];
			}
		}
	}
}

__global__ void mm_gpu_shared(float *A, float *B, float *C, int N)
{
}
double sum(float *X, int N)
{
	double sum = 0;
	// Wait for GPU to finish before accessing on host
	for (int i = 0; i < N; ++i)
	{
		sum += X[i];
	}
	return sum;
}
int main(void)
{
	double tstart, tstop, ttime;
	N = A_HEIGHT * AB_SHARED;
	float r;
	// Allocate Memory
	cudaMallocHost(&A, N * sizeof(float));
	cudaMallocHost(&B, N * sizeof(float));
	cudaMallocHost(&C, N * sizeof(float));
	cudaMalloc(&D, N * sizeof(float));

	// initialize A and B arrays on the host
	// initialize A and B arrays on the host
	for (int i = 0; i < N; i++)
	{
		r = (float)rand() / (float)RAND_MAX;
		A[i] = r;
		B[i] = r;
	}

	//multiply gpu non shared
	tstart=dtime();
	
	//launch kernel on 
	tstop=dtime();
	//multiply cpu serial
	tstart = dtime();
	mm_cpu_serial();
	tstop = dtime();
	ttime = tstop - tstart;
	//print result for non threaded
	printf("Secs serial = %10.3lf\n", ttime);
	printf("Sum:%lf\n", sum(C,N));
	//clear C
	memset(C,0,N*sizeof(C[0]));
	// Free memory
	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);
	cudaFree(D);
	return 0;
}
