#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>

#include "BlackScholes_kernel.cuh"


//error printing
static void HandleError(cudaError_t err, const char *file, int line) 
{
	if (err != cudaSuccess) 
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high)
{
	float t = (float)rand() / (float)RAND_MAX;
	return (1.0f - t) * low + t * high;
}

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
	float2 * __restrict d_CallResult,	//
	float2 * __restrict d_PutResult,	//
	float2 * __restrict d_StockPrice,	//
	float2 * __restrict d_OptionStrike,	//
	float2 * __restrict d_OptionYears,	//
	float Riskfree,						//
	float Volatility,					//
	int optN,							//parametry B
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
		BlackScholesGPU(
		d_CallResult,
		d_PutResult,
		d_StockPrice,
		d_OptionStrike,
		d_OptionYears,
		Riskfree,
		Volatility,
		optN,
		mapBlk, blkDim_B, gridDim_B);
	}
}

void compute_mapping(int * mapKernel, int * mapBlk, size_t mapSize, int gridDim_A, int gridDim_B, int smAlloc_A, int smAlloc_B)
{
	int totalSm = smAlloc_A + smAlloc_B;
	for (int i = 0, blkA = 0, blkB = 0; i < mapSize; ++i)
	{
		if ((i % totalSm < smAlloc_A && blkA < gridDim_A) || blkB >= gridDim_B)
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

	//////////////////////////////////////////////////////////////////////////
	//vecAdd init
	int va_numElements = 1000000;
	int va_threadsPerBlock = 1024;
	int va_blocksPerGrid = (va_numElements + va_threadsPerBlock - 1) / va_threadsPerBlock;
	size_t va_size = va_numElements * sizeof(float);
	
	float *h_A = (float *)malloc(va_size);
	float *h_B = (float *)malloc(va_size);
	float *h_C = (float *)malloc(va_size);

	for (int i = 0; i < va_numElements; ++i)
	{
		h_A[i] = rand() / (float)RAND_MAX;
		h_B[i] = rand() / (float)RAND_MAX;
	}
	
	float *d_A = NULL;
	HANDLE_ERROR(cudaMalloc((void **)&d_A, va_size));
	float *d_B = NULL;
	HANDLE_ERROR(cudaMalloc((void **)&d_B, va_size));
	float *d_C = NULL;
	HANDLE_ERROR(cudaMalloc((void **)&d_C, va_size));

	HANDLE_ERROR(cudaMemcpy(d_A, h_A, va_size, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_B, h_B, va_size, cudaMemcpyHostToDevice));
	//endof vecAdd init
	//////////////////////////////////////////////////////////////////////////
	
	//////////////////////////////////////////////////////////////////////////
	//BlackScholes init
	int bs_numOpt = 1000000;
	int bs_threadsPerBlock = 128;
	int bs_blocksPerGrid = (bs_numOpt + bs_threadsPerBlock - 1) / bs_threadsPerBlock;
	int bs_size = bs_numOpt * sizeof(float);
	float bs_riskfree = 0.02f;
	float bs_volatility = 0.30f;
	
	float * h_CallResultGPU;
	float * h_PutResultGPU;
	float * h_OptionYears;
	float * h_OptionStrike;
	float * h_StockPrice;

	h_CallResultGPU = (float *)malloc(bs_size);
	h_PutResultGPU = (float *)malloc(bs_size);
	h_StockPrice = (float *)malloc(bs_size);
	h_OptionStrike = (float *)malloc(bs_size);
	h_OptionYears = (float *)malloc(bs_size);
	
	for (int i = 0; i < bs_numOpt; i++)
	{
		h_StockPrice[i] = RandFloat(5.0f, 30.0f);
		h_OptionStrike[i] = RandFloat(1.0f, 100.0f);
		h_OptionYears[i] = RandFloat(0.25f, 10.0f);
	}
	
	float * d_CallResult = NULL;
	float * d_PutResult = NULL;
	float * d_StockPrice = NULL;
	float * d_OptionStrike = NULL;
	float * d_OptionYears = NULL;

	HANDLE_ERROR(cudaMalloc((void **)&d_CallResult, bs_size));
	HANDLE_ERROR(cudaMalloc((void **)&d_PutResult, bs_size));
	HANDLE_ERROR(cudaMalloc((void **)&d_StockPrice, bs_size));
	HANDLE_ERROR(cudaMalloc((void **)&d_OptionStrike, bs_size));
	HANDLE_ERROR(cudaMalloc((void **)&d_OptionYears, bs_size));
	
	HANDLE_ERROR(cudaMemcpy(d_StockPrice, h_StockPrice, bs_size, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_OptionStrike, h_OptionStrike, bs_size, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_OptionYears, h_OptionYears, bs_size, cudaMemcpyHostToDevice));
	
	//endof BlackScholes init
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
	//Mapping
	size_t mapSize = va_blocksPerGrid + bs_blocksPerGrid;
	size_t mapBytes = mapSize * sizeof(int);
	int * mapBlk = (int*)malloc(mapBytes);
	int * mapKernel = (int*)malloc(mapBytes);

	compute_mapping(mapKernel, mapBlk, mapSize, va_blocksPerGrid, bs_blocksPerGrid, 1, 1);
	
	int * d_mapBlk = NULL;
	HANDLE_ERROR(cudaMalloc((void **)&d_mapBlk, mapBytes));
	
	int * d_mapKernel = NULL;
	HANDLE_ERROR(cudaMalloc((void **)&d_mapKernel, mapBytes));

	HANDLE_ERROR(cudaMemcpy(d_mapBlk, mapBlk, mapBytes, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_mapKernel, mapKernel, mapBytes, cudaMemcpyHostToDevice));
	//endof Mapping
	//////////////////////////////////////////////////////////////////////////



	scheduler <<< mapSize, 1024 >>>(d_A, d_B, d_C, va_numElements,
		(float2 *)d_CallResult,
		(float2 *)d_PutResult,
		(float2 *)d_StockPrice,
		(float2 *)d_OptionStrike,
		(float2 *)d_OptionYears,
		bs_riskfree,
		bs_volatility,
		bs_size,
		d_mapBlk, d_mapKernel, va_blocksPerGrid, va_threadsPerBlock, bs_blocksPerGrid, bs_threadsPerBlock
		);
	
	cudaDeviceSynchronize();
	
	HANDLE_ERROR(cudaGetLastError());

	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	HANDLE_ERROR(cudaMemcpy(h_C, d_C, va_size, cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaMemcpy(h_CallResultGPU, d_CallResult, bs_size, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(h_PutResultGPU, d_PutResult, bs_size, cudaMemcpyDeviceToHost));

	// Free device global memory
	HANDLE_ERROR(cudaFree(d_A));
	HANDLE_ERROR(cudaFree(d_B));
	HANDLE_ERROR(cudaFree(d_C));
	HANDLE_ERROR(cudaFree(d_mapKernel));
	HANDLE_ERROR(cudaFree(d_mapBlk));

	HANDLE_ERROR(cudaFree(d_OptionYears));
	HANDLE_ERROR(cudaFree(d_OptionStrike));
	HANDLE_ERROR(cudaFree(d_StockPrice));
	HANDLE_ERROR(cudaFree(d_PutResult));
	HANDLE_ERROR(cudaFree(d_CallResult));
	
	// Free host memory
	free(h_A);
	free(h_B);
	free(h_C);
	free(mapKernel);
	free(mapBlk);

	free(h_OptionYears);
	free(h_OptionStrike);
	free(h_StockPrice);
	free(h_PutResultGPU);
	free(h_CallResultGPU);
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

