#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "RegressionData.h"
#include "eTimer.h"

class RegressionSolver
{
public:
	struct Results {
		MklMatrix solutions;
		double error;
		eTimer timer;
	};
	
	static Results solve(RegressionData &data);

private:
	bool static  solveSystem(MklMatrix &coefficients, MklMatrix &constantTerms);
};

