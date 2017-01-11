#include "RegressionData.h"
#include "RegressionSolver.h"


int main() {

	RegressionData data = RegressionData::readFromFile("test.mat");

	eTimer timer;
	timer.start();
	RegressionSolver::Results result = RegressionSolver::solve(data);
	timer.stop();

	std::cout << "Error: " << result.error << "\n"
		<< "Solciones:\n" <<
		result.solutions;

	timer.report();
	
	getchar();

}

