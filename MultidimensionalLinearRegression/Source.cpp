#include "RegressionSolver.h"
#include "eTimer.h"

int main() {

	
	RegressionData mklData = RegressionData::readFromFile("data");
	RegressionData cudaData(mklData);

	eTimer mklTimer;
	eTimer cudaTimer;

	for (int i = 0; i < 4; ++i) {
		mklTimer.start();
		RegressionSolver::Results mklResult = RegressionSolver::solve(mklData);
		mklTimer.stop();
	}

	Gpu::init();
	cudaTimer.start();
	RegressionSolver::Results cudaResult = RegressionSolver::solveWithCuda(cudaData);
	cudaTimer.stop();
	Gpu::end();


	mklTimer.start();
	RegressionSolver::Results mklResult = RegressionSolver::solve(mklData);
	mklTimer.stop();

	std::cout << "Error: " << mklResult.error << "\n"
		<< "Soluciones:\n" <<
		mklResult.solutions;
	mklTimer.report();

	std::cout << "Error: " << cudaResult.error << "\n"
		<< "Soluciones:\n" <<
		cudaResult.solutions;
	cudaTimer.report();
	

	getchar();

	return 0;

}

