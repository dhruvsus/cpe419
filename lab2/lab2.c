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


