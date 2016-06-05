#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


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
	int kernel_id = mapKernel[blockIdx.x];
	if (kernel_id == 0)
	{
		//launch kernel A
		vectorAdd_mod(A_A, B_A, C_A, numElements_A,
			mapBlk, blkDim_A, gridDim_A);
	}
	else
	{
		//launch kernel B
		vectorAdd_mod(A_B, B_B, C_B, numElements_B, 
			mapBlk, blkDim_B, gridDim_B);
	}
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
	HANDLE_ERROR(cudaMalloc((void **)&d_mapBlk, mapBytes));
	
	int * d_mapKernel = NULL;
	HANDLE_ERROR(cudaMalloc((void **)&d_mapKernel, mapBytes));

	float *d_A_A = NULL;
	HANDLE_ERROR(cudaMalloc((void **)&d_A_A, size));
	float *d_B_A = NULL;
	HANDLE_ERROR(cudaMalloc((void **)&d_B_A, size));

	float *d_A_B = NULL;
	HANDLE_ERROR(cudaMalloc((void **)&d_A_B, size));
	float *d_B_B = NULL;
	HANDLE_ERROR(cudaMalloc((void **)&d_B_B, size));

	float *d_C_A = NULL;
	HANDLE_ERROR(cudaMalloc((void **)&d_C_A, size));
	float *d_C_B = NULL;
	HANDLE_ERROR(cudaMalloc((void **)&d_C_B, size));

	
	HANDLE_ERROR(cudaMemcpy(d_mapBlk, mapBlk, mapBytes, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_mapKernel, mapKernel, mapBytes, cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMemcpy(d_A_A, h_A, size, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_A_B, h_A, size, cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMemcpy(d_B_A, h_B, size, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_B_B, h_B, size, cudaMemcpyHostToDevice));


	scheduler <<< mapSize, threadsPerBlock >>>(d_A_A, d_B_A, d_C_A, numElements,
		d_A_B, d_B_B, d_C_B, numElements,
		d_mapBlk, d_mapKernel, blocksPerGrid, threadsPerBlock, blocksPerGrid, threadsPerBlock
		);
	
	cudaDeviceSynchronize();
	
	HANDLE_ERROR(cudaGetLastError());

	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	HANDLE_ERROR(cudaMemcpy(h_C, d_C_A, size, cudaMemcpyDeviceToHost));

	// Free device global memory
	HANDLE_ERROR(cudaFree(d_A_A));
	HANDLE_ERROR(cudaFree(d_A_B));
	HANDLE_ERROR(cudaFree(d_B_A));
	HANDLE_ERROR(cudaFree(d_B_B));
	HANDLE_ERROR(cudaFree(d_C_A));
	HANDLE_ERROR(cudaFree(d_C_B));
	HANDLE_ERROR(cudaFree(d_mapKernel));
	HANDLE_ERROR(cudaFree(d_mapBlk));

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
	HANDLE_ERROR(cudaDeviceReset());

	printf("Done\n");
	return 0;
}

