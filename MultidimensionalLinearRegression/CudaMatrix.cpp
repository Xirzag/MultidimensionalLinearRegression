#include "CudaMatrix.h"


CudaMatrix::CudaMatrix(int rows, int cols) :
	cols(cols), rows(rows)
{
	int cudaStatus = cudaMalloc((void**)&cudaPointer, rows * cols * sizeof(double));
}

CudaMatrix::CudaMatrix(MklMatrix &matrix) :
	cols(matrix.cols), rows(matrix.rows)
{
	int cudaStatus = cudaMalloc((void**)&cudaPointer, matrix.rows * matrix.cols * sizeof(double));
	cudaStatus = cudaMemcpy(cudaPointer, matrix.allocMem, matrix.size() * sizeof(double), cudaMemcpyHostToDevice);
}

CudaMatrix::CudaMatrix(CudaMatrix && matrix) :
	cols(matrix.cols), rows(matrix.rows)
{
	cudaPointer = matrix.cudaPointer;
	matrix.cudaPointer = nullptr;
}


CudaMatrix CudaMatrix::multiplyBy(CudaMatrix & matrixB, cublasOperation_t transposeB, cublasOperation_t transposeA)
{

	int n, m, k, kOther;
	if (transposeA == CUBLAS_OP_N) {
		k = cols;
		m = rows;
	}
	else {
		k = rows;
		m = cols;
	}
	if (transposeB == CUBLAS_OP_N) {
		kOther = matrixB.rows;
		n = matrixB.cols;
	}
	else {
		kOther = matrixB.cols;
		n = matrixB.rows;
	}


	CudaMatrix result(m, n);
	if (k != kOther)
		throw std::invalid_argument("col - rows don't match");
	else {
		double alpha = 1.0, beta = 0.0;
#ifdef CUDA_MORE_MULTIPLICATION
		if (m == 1 && n == 1) {

			int cublasStatus = cublasDdot(Gpu::cuBlasHandle, k, this->cudaPointer, 1, matrixB.cudaPointer, 1, result.cudaPointer);
			if (cublasStatus != 0)
				throw std::invalid_argument("Error in Cuda Multiplication");


		}
		elseif (n == 1) {

			int cublasStatus = cublasDgemv(Gpu::cuBlasHandle, transposeA, this->rows, this->cols, &alpha,
				this->cudaPointer, this->rows,
				matrixB.cudaPointer, 1, 0, result.cudaPointer, 1);
			if (cublasStatus != 0)
				throw std::invalid_argument("Error in Cuda Multiplication");

		}
		else 
#endif
		{

			int cublasStatus = cublasDgemm(Gpu::cuBlasHandle, transposeA, transposeB,
				m, n, k, &alpha, cudaPointer, rows,
				matrixB.cudaPointer, matrixB.rows, &beta, result.cudaPointer, result.rows);
			if (cublasStatus != 0)
				throw std::invalid_argument("Error in Cuda Multiplication");

		}

		
	}
	return result;
}

void CudaMatrix::minus(CudaMatrix matrix)
{
	if (matrix.cols == cols && matrix.rows == rows) {
		double alpha = 1, beta = -1;
		cublasDgeam(Gpu::cuBlasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			matrix.rows, matrix.cols, &alpha,
			cudaPointer, rows, &beta,
			matrix.cudaPointer, matrix.rows,
			cudaPointer, rows
		);
	}
}

void CudaMatrix::get(MklMatrix & matrix)
{
	int cudaStatus = cudaMemcpy(cudaPointer, matrix.allocMem, matrix.size() * sizeof(double), cudaMemcpyDeviceToHost);
}

MklMatrix CudaMatrix::get()
{
	Gpu::synchronize();
	MklMatrix matrix(rows, cols);
	int cudaStatus = cudaMemcpy(matrix.allocMem, cudaPointer, matrix.size() * sizeof(double), cudaMemcpyDeviceToHost);
	return matrix;
}

CudaMatrix::~CudaMatrix()
{
	if (cudaPointer != nullptr)
		cudaFree(cudaPointer);
}
