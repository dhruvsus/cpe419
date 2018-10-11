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

#define A_HEIGHT 10
#define B_WIDTH 20
#define AB_SHARED 5

//declare global variables
float A[A_HEIGHT][AB_SHARED];
float B[AB_SHARED][B_WIDTH];
float C[A_HEIGHT][B_WIDTH];
long threads_available;

int main(int argc, char *argv[]){
	FILE *fp;
	int i, j;
	float array_read;
	if(argc!=3)
		return 0;
	fp = fopen(argv[2], "r"); // assuming format of command ./a.out <#threads> <name of file>
	//fill up A
	for(i=0;i<AB_SHARED;i++){
		for(j=0;j<A_HEIGHT;j++){
			fscanf(fp, "%f", &array_read);
			A[i][j]=array_read;
			printf("%f\n",array_read);
		}
	}
	//fill up B
	for(i=0;i<B_WIDTH;i++)
	{
		for(j=0;j<AB_SHARED;j++){
			fscanf(fp, "%f", &B[i][j]);
			printf("%f\n",B[i][j]);;
		}
	}	
}
