//imports
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <math.h>
#include <cuda.h>

__global__
void printSome(int x){
    printf("%d", x);
}
int main(){
    int i;
    for(i=0;i<50;i++){
        printSome<<<1,1>>>(i);
    }
    return 0;
}
