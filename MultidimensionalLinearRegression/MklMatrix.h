#pragma once
#include <iostream>
#include <istream>
#include <fstream> 
#include <sstream>
#include <stdexcept>
#include <mkl.h>
#include <math.h>

class MklMatrix
{
public:
	MklMatrix();
	MklMatrix(int rows, int cols);
	MklMatrix(int rows, int cols, std::initializer_list<double> list);
	MklMatrix(MklMatrix &&matrix); //move constructor
	MklMatrix(const MklMatrix &matrix); //copy constructor
	void operator=(MklMatrix &&matrix);

	MklMatrix multiplyBy(MklMatrix &other, CBLAS_TRANSPOSE transposeOther = CblasNoTrans, CBLAS_TRANSPOSE transposeMe = CblasNoTrans);
	void minus(MklMatrix matrix);
	
	std::string toString();
	std::string toMatlabString();

	~MklMatrix();


	inline size_t size() const {
		return cols * rows;
	}
	inline double& val(int y, int x) {
		return allocMem[x * rows + y]; 
	}

	inline bool operator==(MklMatrix &matrix) {
		if (matrix.cols != cols && matrix.rows != rows) return false;
		for (int i = 0; i < size(); ++i) 
			if (std::abs(matrix.allocMem[i] - allocMem[i]) > 1)
				return false;

		return true;
	}

	int cols;
	int rows;
	double *allocMem;

};

static std::ostream& operator<<(std::ostream& os, MklMatrix& matrix)
{
	for (int y = 0; y < matrix.rows; y++)
	{
		for (int x = 0; x < matrix.cols; x++)
			os << matrix.val(y, x) << ", ";

		os << "\n";
	}
	return os;
}