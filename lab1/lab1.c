#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <pthread.h>
//
// dtime -
//
// utility routine to return 
// the current wall clock time
//
double dtime()
{
    double tseconds = 0.0;
    struct timeval mytime;
    gettimeofday(&mytime,(struct timezone*)0);
    tseconds = (double)(mytime.tv_sec + mytime.tv_usec*1.0e-6);
    return( tseconds );
}

#define HEIGHT 900
#define WIDTH 100
#define ARRAY_SIZE (HEIGHT*WIDTH) 

// declare input arrays  
int x[HEIGHT][WIDTH];
int y[HEIGHT][WIDTH];

// declare the output array
int fc[HEIGHT][WIDTH];
int fcthread[HEIGHT][WIDTH];

/*global variable accesible to all threads*/
long threads_count;

//
// Matrix Addition non threaded
// 
void matrixAdd(int Hight, int Width){
        int row, col, k;
        float Pvalue=0;
        for (row=0; row<Hight; row++){
                for(col=0; col<Width; col++) {
                        fc[row][col]=x[row][col]+y[row][col];
                }
        }
}

//
//Matrix Addition threaded
//
void* matrixAddTh(void* id){
	int threadID=(int)id;
	int i,j;
	int num_rows_per_thread=HEIGHT/threads_count;
	int start_row=threadID*num_rows_per_thread;
	int stop_row=start_row+num_rows_per_thread;
	for(i=start_row;i<stop_row;i++){
		for(j=0;j<WIDTH;j++){
			fcthread[i][j]=x[i][j]+y[i][j];
		}
	}
        return NULL;
}

int main(int argc, char *argv[] )
{
        int i,j;
	double tstart, tstop, ttime;
	srand(time(NULL));
        //
        // initialize the input arrays 
        //
        printf("Initializing\r\n");
	for(i=0; i<ARRAY_SIZE; i++)
        {
            int rloc=i/WIDTH;
	    int cloc=i%WIDTH;
	    x[rloc][cloc]=rand()%100;
	    y[rloc][cloc]=rand()%100;
        }

        printf("Starting Compute\r\n");

        tstart = dtime();
        // loop many times to really get lots of calculations
        for(j=0; j<1000; j++)
        {
                // add the two arrays 
                matrixAdd(HEIGHT,WIDTH);
        }
        tstop = dtime();

        // elasped time
        ttime = tstop - tstart;

        //
        // Print the results
        //
        if ((ttime) > 0.0)
        {
            printf("Secs Serial = %10.3lf\n",ttime);
        }
	
        //threaded part of HW
        long thread;
        pthread_t* thread_handles;
        double tstartT, tstopT, ttimeT;

        //get number of threads for user input and allocate memory to them
        threads_count=strtol(argv[1], NULL, 10);
	thread_handles=malloc(threads_count*sizeof(pthread_t));
        tstartT = dtime();
        for(thread=0; thread<threads_count; thread++)
                pthread_create(&thread_handles[thread], NULL, matrixAddTh, (void*) thread);
        for(thread=0; thread<threads_count; thread++)
                pthread_join(thread_handles[thread], NULL);
        free(thread_handles);
        tstopT=dtime();
        ttimeT=tstopT-tstartT;
        if ((ttimeT) > 0.0)
        {
            printf("Secs Threaded = %10.3lf\n", ttimeT);
        }
	
        //check the solutions are the same in both implementations
        int row, col;
        float dif, accum=0;
        for (row=0; row<HEIGHT; row++){
                for(col=0; col<WIDTH; col++) {
                        //dif=abs(fc[row*WIDTH+col]-fcthread[row*WIDTH+col]);
                        //editing diff formula to account for 2d array on the stack
			dif=abs(fc[row][col]-fcthread[row][col]);
			if(dif!=0) accum+=dif;
                }
        }
        if(accum < 0.1) printf("SUCCESS\n");
        else printf("FAIL\n");
	return( 0 );
}

