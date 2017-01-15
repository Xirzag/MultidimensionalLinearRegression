#pragma once
#include "Gpu.h"
#include "MklMatrix.h"


class CudaMatrix
{
public:

	CudaMatrix(int rows, int cols);
	CudaMatrix(MklMatrix &matrixToGpu);
	CudaMatrix(CudaMatrix &&other);


	CudaMatrix multiplyBy(CudaMatrix &matrixB, cublasOperation_t transposeB = CUBLAS_OP_N, cublasOperation_t transposeA = CUBLAS_OP_N);

	void minus(CudaMatrix matrix);

	void get(MklMatrix &matrix);
	MklMatrix get();

	~CudaMatrix();

	double* cudaPointer = nullptr;
	const int rows;
	const int cols;
};
