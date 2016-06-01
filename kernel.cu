#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		C[i] = A[i] + B[i];
	}
}


__global__ void dummyKernel(/* float *x, */ int n)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	float result;
	for (int i = tid; i < n; i += blockDim.x * gridDim.x) 
	{
		result = sqrt(pow(3.14159, i));
	}
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
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Print the vector length to be used, and compute its size
	int numElements = 500000;
	size_t size = numElements * sizeof(float);
	printf("[Vector addition of %d elements]\n", numElements);

	//Create streams
	cudaStream_t stream1, stream2;

	err = cudaStreamCreate(&stream1);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to create stream1 (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaStreamCreate(&stream2);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to create stream2 (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the host input vector A
	float *h_A = (float *)malloc(size);

	// Allocate the host input vector B
	float *h_B = (float *)malloc(size);

	// Allocate the host output vector C
	float *h_C = (float *)malloc(size);

	// Verify that allocations succeeded
	if (h_A == NULL || h_B == NULL || h_C == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// Initialize the host input vectors
	for (int i = 0; i < numElements; ++i)
	{
		h_A[i] = rand() / (float)RAND_MAX;
		h_B[i] = rand() / (float)RAND_MAX;
	}

	// Allocate the device input vector A
	float *d_A_A = NULL;
	err = cudaMalloc((void **)&d_A_A, size);

	float *d_A_B = NULL;
	err = cudaMalloc((void **)&d_A_B, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A1 (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector B
	float *d_B_A = NULL;
	err = cudaMalloc((void **)&d_B_A, size);

	float *d_B_B = NULL;
	err = cudaMalloc((void **)&d_B_B, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B1 (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector C
	float *d_C_A = NULL;
	err = cudaMalloc((void **)&d_C_A, size);

	float *d_C_B = NULL;
	err = cudaMalloc((void **)&d_C_B, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C1 (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the host input vectors A and B in host memory to the device input vectors in
	// device memory
	printf("Copy input data from the host memory to the CUDA device - stream1\n");
	err = cudaMemcpyAsync(d_A_A, h_A, size, cudaMemcpyHostToDevice, stream1);
	err = cudaMemcpyAsync(d_A_B, h_A, size, cudaMemcpyHostToDevice, stream2);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpyAsync(d_B_A, h_A, size, cudaMemcpyHostToDevice, stream1);
	err = cudaMemcpyAsync(d_B_B, h_B, size, cudaMemcpyHostToDevice, stream2);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 1024;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);



	size_t mapSize = blocksPerGrid * 2;
	int * mapBlk = (int*)malloc(mapSize * sizeof(int));
	int * mapKernel = (int*)malloc(mapSize * sizeof(int));
	int * d_mapBlk = nullptr;
	int * d_mapKernel = nullptr;
	
	//mapowanie
	compute_mapping(mapKernel, mapBlk, mapSize, blocksPerGrid, blocksPerGrid, 1, 1);
	
	
	cudaMalloc((void **)&d_mapBlk, mapSize);
	cudaMalloc((void **)&d_mapKernel, mapSize);

	cudaMemcpyAsync(d_mapBlk, mapBlk, mapSize, cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(d_mapKernel, mapKernel, mapSize, cudaMemcpyHostToDevice, stream2);

	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream2);

	scheduler <<<blocksPerGrid * 2, threadsPerBlock >>>(d_A_A, d_B_A, d_C_A, numElements,
		d_A_B, d_B_B, d_C_B, numElements,
		nullptr, nullptr, blocksPerGrid, threadsPerBlock, blocksPerGrid, threadsPerBlock
		);
	
	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaStreamSynchronize(stream1);
	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpyAsync(h_C, d_C_A, size, cudaMemcpyDeviceToHost, stream1);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Free device global memory
	err = cudaFree(d_A_A);
	err = cudaFree(d_B_B);
	err = cudaFree(d_C_A);

	// Free host memory
	free(h_A);
	free(h_B);
	free(h_C);

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

