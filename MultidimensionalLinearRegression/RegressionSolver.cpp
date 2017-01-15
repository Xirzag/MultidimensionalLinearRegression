#include "RegressionSolver.h"

RegressionSolver::Results RegressionSolver::solveWithCuda(RegressionData & data)
{
	CudaMatrix y(data.y);
	CudaMatrix z(data.z);

		
	CudaMatrix coefficients = y.multiplyBy(y, CUBLAS_OP_N, CUBLAS_OP_T);
	auto t1 = coefficients.get();
	CudaMatrix constantTerms = y.multiplyBy(z, CUBLAS_OP_N, CUBLAS_OP_T);
	auto t2 = constantTerms.get();

	Gpu::synchronize();

	if (!solveSystem(coefficients, constantTerms))
		throw std::invalid_argument("can't solve system");

	Gpu::synchronize();

	CudaMatrix &unknowns = constantTerms;
	CudaMatrix &d = z;

	d.minus(y.multiplyBy(unknowns));

	auto t3 = d.get();

	Gpu::synchronize();
	double error = sqrt(d.multiplyBy(d, CUBLAS_OP_N, CUBLAS_OP_T).get().val(0,0));

	return Results{unknowns.get(), error};
}

RegressionSolver::Results RegressionSolver::solve(RegressionData & data)
{

	MklMatrix coefficients, constantTerms;

#pragma omp parallel sections
	{
#pragma omp section
		{
			coefficients = data.y.multiplyBy(data.y, CblasNoTrans, CblasTrans);
		}
#pragma omp section
		{
			constantTerms = data.y.multiplyBy(data.z, CblasNoTrans, CblasTrans);
		}
	}

	if (!solveSystem(coefficients, constantTerms))
		throw std::invalid_argument("can't solve system");

	MklMatrix &unknowns = constantTerms;
	MklMatrix &d = data.z;

	d.minus(data.y.multiplyBy(unknowns));
	double error = sqrt(d.multiplyBy(d, CblasNoTrans, CblasTrans).val(0, 0));

	return Results{ unknowns, error };
}

//Solution is calculated in the constantTerms
bool RegressionSolver::solveSystem(MklMatrix & coefficients, MklMatrix & constantTerms)
{

	int *pivot = new int[coefficients.rows]; //Todo: To MKL
	int result = LAPACKE_dgesv(LAPACK_COL_MAJOR, coefficients.rows, constantTerms.cols,
		coefficients.allocMem, coefficients.rows, pivot,
		constantTerms.allocMem, constantTerms.rows);

	return result == 0;
}

bool RegressionSolver::solveSystem(CudaMatrix & coefficients, CudaMatrix & constantTerms)
{

	int Lwork;
	double * Work;
	int *dev_pivot, *dev_info;

	auto cudaStatus = cudaMalloc((void**)&dev_pivot, coefficients.cols * sizeof(double));
	cudaStatus = cudaMalloc((void**)&dev_info, 1 * sizeof(double));
	auto cusolverStatus = cusolverDnDgetrf_bufferSize(Gpu::cuSolverHandle,
		coefficients.cols, coefficients.rows, coefficients.cudaPointer, coefficients.cols, &Lwork);
	cudaStatus = cudaMalloc((void**)&Work, Lwork * sizeof(double));
	cusolverStatus = cusolverDnDgetrf(Gpu::cuSolverHandle,
		coefficients.cols, coefficients.rows, coefficients.cudaPointer,
		coefficients.cols, Work, dev_pivot, dev_info);
	cusolverStatus = cusolverDnDgetrs(Gpu::cuSolverHandle,
		CUBLAS_OP_T, coefficients.cols, 1, coefficients.cudaPointer, coefficients.cols,
		dev_pivot, constantTerms.cudaPointer, coefficients.cols, dev_info);



	return cusolverStatus == 0;

};



