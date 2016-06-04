#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <device_launch_parameters.h>



__device__ void vectorAdd_mod(const float *A, const float *B, float *C, int numElements,
	const int * _mapBlk, int _blkDim, int _gridDim)
{
	if (threadIdx.x < _blkDim)
	{
		int _oldbid = _mapBlk[blockIdx.x];
		//actual kernel
		//////////////////////////////////////////////////////////////////////////
		int i = blockDim.x * _oldbid + threadIdx.x;
		if (i < numElements)
		{
			C[i] = A[i] + B[i];
		}
		//////////////////////////////////////////////////////////////////////////
	}
}


__global__ void scheduler(const float * A_A, const float * B_A, float * C_A, int numElements_A, //parametry A
	const float * A_B, const float * B_B, float * C_B, int numElements_B,		 //parametry B
	const int * mapBlk, const int * mapKernel,								 //mapowanie
	int gridDim_A, int blkDim_A, int gridDim_B, int blkDim_B)				 //wymiary
{
	float a = A_A[blockIdx.x];
	
	//int kernel_id = mapKernel[blockIdx.x];
	//if (kernel_id == 0)
	//{
	//	//launch kernel A
	//	vectorAdd_mod(A_A, B_A, C_A, numElements_A,
	//		mapBlk, blkDim_A, gridDim_A);
	//}
	//else
	//{
	//	//launch kernel B
	//	vectorAdd_mod(A_B, B_B, C_B, numElements_B, 
	//		mapBlk, blkDim_B, gridDim_B);
	//}
}

void compute_mapping(int * mapKernel, int * mapBlk, size_t mapSize, int gridDim_A, int gridDim_B, int smAlloc_A, int smAlloc_B)
{
	int totalSm = smAlloc_A + smAlloc_B;
	for (int i = 0, blkA = 0, blkB = 0; i < mapSize; ++i)
	{
		if (i % totalSm < smAlloc_A)
		{
			mapKernel[i] = 0;
			mapBlk[i] = blkA;
			++blkA;
		}
		else
		{
			mapKernel[i] = 1;
			mapBlk[i] = blkB;
			++blkB;
		}

	}
}

/**
* Host main routine
*/
int main(void)
{
	cudaError_t err = cudaSuccess;

	int numElements = 500000;
	int threadsPerBlock = 1024;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	size_t size = numElements * sizeof(float);

	float *h_A = (float *)malloc(size);
	float *h_B = (float *)malloc(size);
	float *h_C = (float *)malloc(size);

	for (int i = 0; i < numElements; ++i)
	{
		h_A[i] = rand() / (float)RAND_MAX;
		h_B[i] = rand() / (float)RAND_MAX;
	}

	size_t mapSize = blocksPerGrid * 2;
	size_t mapBytes = mapSize * sizeof(int);
	int * mapBlk = (int*)malloc(mapBytes);
	int * mapKernel = (int*)malloc(mapBytes);

	compute_mapping(mapKernel, mapBlk, mapSize, blocksPerGrid, blocksPerGrid, 1, 1);

	int * d_mapBlk = NULL;
	err = cudaMalloc((void **)&d_mapBlk, mapBytes);
	
	int * d_mapKernel = NULL;
	err = cudaMalloc((void **)&d_mapKernel, mapBytes);

	float *d_A_A = NULL;
	err = cudaMalloc((void **)&d_A_A, size);
	float *d_B_A = NULL;
	err = cudaMalloc((void **)&d_B_A, size);

	float *d_A_B = NULL;
	err = cudaMalloc((void **)&d_A_B, size);
	float *d_B_B = NULL;
	err = cudaMalloc((void **)&d_B_B, size);

	float *d_C_A = NULL;
	err = cudaMalloc((void **)&d_C_A, size);
	float *d_C_B = NULL;
	err = cudaMalloc((void **)&d_C_B, size);

	
	err = cudaMemcpy(d_mapBlk, mapBlk, mapSize * sizeof(int), cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_mapKernel, mapKernel, mapSize * sizeof(int), cudaMemcpyHostToDevice);

	err = cudaMemcpy(d_A_A, h_A, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_A_B, h_A, size, cudaMemcpyHostToDevice);

	err = cudaMemcpy(d_B_A, h_B, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_B_B, h_B, size, cudaMemcpyHostToDevice);


	scheduler <<< mapSize, threadsPerBlock >>>(d_A_A, d_B_A, d_C_A, numElements,
		d_A_B, d_B_B, d_C_B, numElements,
		mapBlk, mapKernel, blocksPerGrid, threadsPerBlock, blocksPerGrid, threadsPerBlock
		);
	
	cudaDeviceSynchronize();
	
	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch scheduler kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_C, d_C_A, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Free device global memory
	err = cudaFree(d_A_A);
	err = cudaFree(d_A_B);
	err = cudaFree(d_B_A);
	err = cudaFree(d_B_B);
	err = cudaFree(d_C_A);
	err = cudaFree(d_C_B);
	err = cudaFree(d_mapKernel);
	err = cudaFree(d_mapBlk);

	// Free host memory
	free(h_A);
	free(h_B);
	free(h_C);
	free(mapKernel);
	free(mapBlk);

	// Reset the device and exit
	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	err = cudaDeviceReset();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("Done\n");
	return 0;
}

