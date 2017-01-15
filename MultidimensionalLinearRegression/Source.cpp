#include "RegressionSolver.h"
#include "eTimer.h"


int main() {

	RegressionData mklData = RegressionData::readFromFile("mat3.mat");
	RegressionData cudaData(mklData);

	eTimer mklTimer;
	eTimer cudaTimer;


	mklTimer.start();
	RegressionSolver::Results mklResult = RegressionSolver::solve(mklData);
	mklTimer.stop();


	Gpu::init();
	cudaTimer.start();
	RegressionSolver::Results cudaResult = RegressionSolver::solveWithCuda(cudaData);
	cudaTimer.stop();
	Gpu::end();




	std::cout << "Error: " << mklResult.error << "\n"
		<< "Soluciones:\n" <<
		mklResult.solutions;
	mklTimer.report();

	std::cout << "Error: " << cudaResult.error << "\n"
		<< "Soluciones:\n" <<
		cudaResult.solutions;
	cudaTimer.report();
	

	getchar();

}

