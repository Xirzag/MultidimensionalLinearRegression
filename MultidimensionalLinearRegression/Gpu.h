#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

namespace Gpu {

	extern cublasHandle_t cuBlasHandle;
	extern cusolverDnHandle_t cuSolverHandle;
	extern int device;

	//https://devtalk.nvidia.com/default/topic/796648/cannot-initialize-cublas-library-/
	//http://docs.nvidia.com/cuda/cublas/#cublasxt_create
	void static init() {
		int count;
		cudaGetDeviceCount(&count);
		cudaError_t cudaStatus = cudaGetDevice(0);
		cublasStatus_t cublasStatus = cublasCreate(&Gpu::cuBlasHandle);
		cusolverStatus_t cusolverStatus = cusolverDnCreate(&Gpu::cuSolverHandle);
	};

	void static end() {
		cusolverStatus_t cusolverStatus = cusolverDnDestroy(Gpu::cuSolverHandle);
		cublasStatus_t cublasStatus = cublasDestroy(Gpu::cuBlasHandle);
		cudaError_t cudaStatus = cudaDeviceReset();
	};

	void static synchronize() {
		int cudaStatus = cudaDeviceSynchronize();
	};

};