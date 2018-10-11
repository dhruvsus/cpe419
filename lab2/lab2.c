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

#define A_HEIGHT 100
#define B_WIDTH 90
#define AB_SHARED 80

//declare global variables
float A[A_HEIGHT][AB_SHARED];
float B[AB_SHARED][B_WIDTH];
float C[A_HEIGHT][B_WIDTH];
long threads_available;

void matrix_mult_nonthreaded(int a_height,int b_width,int ab_shared){
	int i,j,k;
	for(i=0;i<a_height;i++){
		for(j=0;j<b_width;j++){
			for(k=0;k<ab_shared;k++){
				C[i][j]+=A[i][k]*B[k][i];
			}
		}
	}
	return;
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
	for(i=0;i<AB_SHARED;i++){
		for(j=0;j<A_HEIGHT;j++){
			fscanf(fp, "%f", &A[i][j]);
		}
	}
	//fill up B
	for(i=0;i<B_WIDTH;i++)
	{
		for(j=0;j<AB_SHARED;j++){
			fscanf(fp, "%f", &B[i][j]);
		}
	}
	//multiply non threaded
	tstart = dtime();
	//multiply 1000 times
	for(k=0;k<1000;k++)
		matrix_mult_nonthreaded(A_HEIGHT,B_WIDTH,AB_SHARED);
	tstop=dtime();
	ttime = tstop-tstart;
	//Print results for non threaded
	printf("Secs serial = %10.3lf\n",ttime);
	return 1;
}
