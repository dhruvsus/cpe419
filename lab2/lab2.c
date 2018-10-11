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

//declare constants

#define A_HEIGHT 900
#define B_WIDTH 9000
#define AB_SHARED 100

//declare global variables
float A[A_HEIGHT][AB_SHARED];
float B[AB_SHARED][B_WIDTH];
float C[A_HEIGHT][B_WIDTH];
long threads_available;

void matrix_mult_nonthreaded(){
	int i,j,k;
	for(i=0;i<A_HEIGHT;i++){
		for(j=0;j<B_WIDTH;j++){
			for(k=0;k<AB_SHARED;k++){
				C[i][j]+=A[i][k]*B[k][i];
			}
		}
	}
	return;
}

void* matrix_mult_threaded(void* id){
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
				C[i][j]+=A[i][k]*B[k][i];
			}
		}
	}
	return NULL;
}
int main(int argc, char *argv[]){
	FILE *fp;
	int i, j, k;
	double tstart,tstop,ttime;
	if(argc!=3)
		return 0;
	fp = fopen(argv[2], "r");
	// assuming format of command ./a.out <#threads> <name of file>
	//fill up A
	for(i=0;i<A_HEIGHT;i++){
		for(j=0;j<AB_SHARED;j++){
			fscanf(fp, "%f", &A[i][j]);
		}
	}
	//fill up B
	for(i=0;i<AB_SHARED;i++)
	{
		for(j=0;j<B_WIDTH;j++){
			fscanf(fp, "%f", &B[i][j]);
		}
	}
	//multiply non threaded
	tstart = dtime();
	//multiply 1000 times
	matrix_mult_nonthreaded();
	tstop=dtime();
	ttime = tstop-tstart;
	//Print results for non threaded
	printf("Secs serial = %10.3lf\n",ttime);
	
	//threaded part of the H/W
	long thread;
	pthread_t* thread_handles;
	//get number of threads from argv[1]
	threads_available=strtol(argv[1],NULL, 10);
	thread_handles=malloc(threads_available*sizeof(pthread_t));
	tstart=dtime();
	for(thread=0;thread<threads_available;thread++)
		pthread_create(&thread_handles[thread], NULL, matrix_mult_threaded, (void*) thread);
	for(thread=0;thread<threads_available;thread++)
		pthread_join(thread_handles[thread], NULL);
	
	free(thread_handles);
	tstop=dtime();
	ttime=tstop-tstart;
	printf("Secs threaded = %10.3lf\n", ttime);
	return 0;
}
