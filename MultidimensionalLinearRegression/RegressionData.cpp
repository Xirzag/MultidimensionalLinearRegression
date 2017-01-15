#include "RegressionData.h"



RegressionData RegressionData::readFromFile(const char* path)
{
	std::ifstream is(path, std::ifstream::binary);
	if (is.fail())
		throw std::invalid_argument("File not found");

	int cols, rows;
	is >> cols;
	is >> rows;

	RegressionData data = {
		MklMatrix(rows, cols+1),
		MklMatrix(rows, 1)
	};

	for (int y = 0; y < rows; y++) {
		is >> data.z.val(y, 0);

		for (int x = 1; x < cols+1; x++) 
			is >> data.y.val(y, x - 1);
		
		data.y.val(y, cols) = 1;
	}

	return data;
}

RegressionData::RegressionData(MklMatrix && yMat, MklMatrix && zMat)
	:y(std::move(yMat)), z(std::move(zMat)) { }

RegressionData::RegressionData(RegressionData & data)
	: z(data.z), y(data.y) { }
