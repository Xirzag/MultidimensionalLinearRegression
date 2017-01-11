#include "MklMatrix.h"
#include <exception>

MklMatrix::MklMatrix() : rows(0), cols(0)
{
	allocMem = nullptr;
}

MklMatrix::MklMatrix(int rows, int cols) : rows(rows), cols(cols)
{
	allocMem = (double*)mkl_malloc(cols*rows * sizeof(double), 64);
}

MklMatrix::MklMatrix(int rows, int cols, std::initializer_list<double> list)
	: rows(rows), cols(cols)
{
	allocMem = (double*)mkl_malloc(rows * cols * sizeof(double), 64);
	int i = 0;
	for (double elem : list)
		allocMem[i++] = elem;

}

MklMatrix::MklMatrix(MklMatrix && matrix) :
	cols(matrix.cols), rows(matrix.rows), allocMem(matrix.allocMem)
{
	matrix.allocMem = nullptr;
}

MklMatrix::MklMatrix(const MklMatrix & matrix)
	: rows(matrix.rows), cols(matrix.cols)
{
	allocMem = (double*)mkl_malloc(matrix.size() * sizeof(double), 64);
	//cblas_dcopy(matrix.size(), matrix.allocMem, 0, allocMem, 0);
	std::memcpy(allocMem, matrix.allocMem, matrix.size() * sizeof(double));
}

void MklMatrix::operator=(MklMatrix && matrix)
{
	if (this->allocMem) mkl_free(this->allocMem);
	this->cols = matrix.cols;
	this->rows = matrix.rows;
	this->allocMem = matrix.allocMem;
	matrix.allocMem = nullptr;
}

std::string MklMatrix::toString()
{
	std::string output;
	for (int y = 0; y < rows; y++)
	{
		for (int x = 0; x < cols; x++)
			output += std::to_string(val(y, x)) + ", ";
		
		output += "\n";
	}
	return output;
}

std::string MklMatrix::toMatlabString()
{
	std::string output("[");
	for (int y = 0; y < rows; y++)
	{
		for (int x = 0; x < cols; x++)
			output += " " + std::to_string(val(y, x));

		output += ";";
	}
	output.append("]");
	return output;
}

MklMatrix MklMatrix::multiplyBy(MklMatrix & other, CBLAS_TRANSPOSE transposeOther, CBLAS_TRANSPOSE transposeMe)
{
	//Mat A = Me m*k
	//Mat B = Other k*n
	//Mat C = Result m*n

	//Improve¿?

	int n, m, k, kOther; 
	if (transposeMe == CblasNoTrans) {
		m = rows;
		k = cols;
	}
	else {
		m = cols;
		k = rows;
	}
	if (transposeOther == CblasNoTrans) {
		n = other.cols;
		kOther = other.rows;
	}
	else {
		n = other.rows;
		kOther = other.cols;
	}
	


	MklMatrix result(m, n);
	if (k != kOther)
		throw std::invalid_argument("col - rows don't match"); 
	else {

		if (this == &other) {
			MklMatrix copy(other);
			cblas_dgemm(CblasRowMajor, transposeMe, transposeOther,
				m, n, k, 1.0,
				this->allocMem, this->cols, copy.allocMem, copy.cols,
				0.0, result.allocMem, result.cols);
		}else
			cblas_dgemm(CblasRowMajor, transposeMe, transposeOther,
				m, n, k, 1.0,
				this->allocMem, this->cols, other.allocMem, other.cols,
				0.0, result.allocMem, result.cols);
			

	}
	return result;
}

void MklMatrix::minus(MklMatrix matrix)
{
	if(matrix.cols == cols && matrix.rows == rows)
		for (int i = 0; i < size(); i++)
			allocMem[i] -= matrix.allocMem[i];
}

bool MklMatrix::contains(std::initializer_list<double> list)
{
	int i = 0;
	for (double elem : list)
		if (fabs(allocMem[i++] - elem) > 0.1f)
			return false;

	return true;
}


MklMatrix::~MklMatrix()
{
	if (this->allocMem) 
		mkl_free(this->allocMem);

}

