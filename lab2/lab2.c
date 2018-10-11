#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthreadh.h>

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
#define B_WIDTH 200
#define AB_SHARED 50

//declare global variables
int A[A_HEIGHT][AB_SHARED];
int B[AB_SHARED][B_WIDTH];
int C[A_HEIGHT][B_WIDTH];
long threads_available;

int main(int argc, char *argv[]){
	FILE *fp;
	int i, j;
	if(argc!=2)
		return 0;
	fp = fopen(argv[2], "r"); // assuming format of command ./a.out <#threads> <name of file>
	//fill up A
	for(i=0;i<AB_SHARED;i++){
		for(j=0;j<A_HEIGHT;j++){
			fscanf(fp, "%1f", &A[i][j]);
		}
	}
	//fill up B
	for(i=0;i<B_WIDTH;i++)
	{
		for(j=0;j<AB_SHARED;j++){
			fscanf(fp, "%1f", &B[i][j]);
		}
	}	
}
