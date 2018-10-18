#include<stdio.h>
#include<cuda_runtime.h>
#include<helper_cuda.h>

__global__ void HelloWorld()
{
	printf("Hello World");
}

int main(void)
{
	//Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;
	//Launch the Vecotr Add CUDA kKernel
	HelloWorld<<<1,1>>>();
	err = cudaGetLastError();
	if(err!=cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGet ErrorString(err));
		exit(EXIT_FAILURE):
	}
	cudaDeviceSynchronize();
	printf("Done\n");
	return -;
}
