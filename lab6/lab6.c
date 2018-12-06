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
	gettimeofday(&mytime, (struct timezone *)0);
	tseconds = (double)(mytime.tv_sec + mytime.tv_usec * 1.0e-6);
	return (tseconds);
}

// CUDA kernel to multiply elements of two arrays
#define A_HEIGHT 1000
#define B_WIDTH 1000
#define AB_SHARED 1000
#define THREADS_AVAILABLE 4

//declare global variables
long threads_available;

float *A;
float *B;
float *C;
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
				C[i * 1000 + j] += A[i * 1000 + k] * B[k * 1000 + i];
			}
		}
	}
	return;
}

void *mm_cpu_parallel(void *id)
{
	int threadID = (int)id;
	int i, j, k;
	int num_rows_per_thread = A_HEIGHT / threads_available;
	int leftover_rows = A_HEIGHT % threads_available;
	int start_row = threadID * num_rows_per_thread;
	int stop_row = start_row + num_rows_per_thread;
	if (threadID == threads_available - 1)
		stop_row = stop_row + leftover_rows;
	for (i = start_row; i < stop_row; i++)
	{
		for (j = 0; j < B_WIDTH; j++)
		{
			for (k = 0; k < AB_SHARED; k++)
			{
				C[i * 1000 + j] += A[i * 1000 + k] * B[k * 1000 + i];
			}
		}
	}
	return NULL;
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
	A = (float *)malloc(N * sizeof(float));
	B = (float *)malloc(N * sizeof(float));
	C = (float *)calloc(N, sizeof(float));

	// initialize A and B arrays on the host
	for (int i = 0; i < N; i++)
	{
		r = (float)rand() / (float)RAND_MAX;
		A[i] = r;
		B[i] = r;
	}

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
	//multiply cpu parallel with 4 phthreads
	long thread;
	pthread_t *thread_handles;
	//get number of threads from argv[1]
	threads_available = THREADS_AVAILABLE;
	thread_handles = malloc(threads_available * sizeof(pthread_t));
	tstart = dtime();
	for (thread = 0; thread < threads_available; thread++)
		pthread_create(&thread_handles[thread], NULL, mm_cpu_parallel, (void *)thread);
	for (thread = 0; thread < threads_available; thread++)
		pthread_join(thread_handles[thread], NULL);

	free(thread_handles);
	tstop = dtime();
	ttime = tstop - tstart;
	printf("Secs threaded = %10.3lf\n", ttime);
	printf("Sum:%lf\n", sum(C,N));
	
	// Free memory
	free(A);
	free(B);
	free(C);
	return 0;
}
