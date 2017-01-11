#include "RegressionSolver.h"

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
	double error = sqrt(d.multiplyBy(d,CblasNoTrans, CblasTrans).val(0,0));

	return Results{unknowns, error};
}


//Solution is calculated in the constantTerms
bool RegressionSolver::solveSystem(MklMatrix & coefficients, MklMatrix & constantTerms)
{

	int *pivot = new int[coefficients.cols];
	int result = LAPACKE_dgesv(LAPACK_ROW_MAJOR, coefficients.cols, constantTerms.cols,
		coefficients.allocMem, coefficients.cols, pivot,
		constantTerms.allocMem, constantTerms.cols);

	return result == 0;
}
