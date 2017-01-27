#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "RegressionData.h"
#include "CudaMatrix.h"


class RegressionSolver
{
public:
	struct Results {
		MklMatrix solutions;
		double error;
	};
	
	static Results solve(RegressionData &data);
	static Results solveWithCuda(RegressionData &data);

private:
	bool static solveSystem(MklMatrix &coefficients, MklMatrix &constantTerms);
	bool static solveSystem(CudaMatrix &coefficients, CudaMatrix &constantTerms);
};


