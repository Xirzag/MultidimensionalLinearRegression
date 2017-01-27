#include "MklMatrix.h"
#include <exception>
#define MKL_MORE_MULTIPLICATION


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
	for (double elem : list) {
		val(i / cols, i%cols) = elem;
		++i;
	}

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

	int n, m, k, kOther; 
	if (transposeMe == CblasNoTrans) {
		k = cols;
		m = rows;
	}
	else {
		k = rows;
		m = cols;
	}
	if (transposeOther == CblasNoTrans) {
		kOther = other.rows;
		n = other.cols;
	}
	else {
		kOther = other.cols;
		n = other.rows;
	}
	
	

	MklMatrix result(m, n);
	if (k != kOther)
		throw std::invalid_argument("col - rows don't match"); 
	else {
#ifdef MKL_MORE_MULTIPLICATION
		if (m == 1 && n == 1) {

			result.val(0,0) = cblas_ddot(k,this->allocMem, 1, other.allocMem, 1);

		}
		else if (n == 1) {

			cblas_dgemv(CblasColMajor, transposeMe, this->rows, this->cols, 1,
				this->allocMem, this->rows,
				other.allocMem, 1, 0, result.allocMem, 1);

		}
		else 
#endif
		{

			cblas_dgemm(CblasColMajor, transposeMe, transposeOther,
				m, n, k, 1.0,
				this->allocMem, this->rows, other.allocMem, other.rows,
				0.0, result.allocMem, result.rows);

		}
		
	}
	return result;
}

void MklMatrix::minus(MklMatrix &matrix)
{
	if(matrix.cols == cols && matrix.rows == rows)
		for (int i = 0; i < size(); i++)
			allocMem[i] = allocMem[i] - matrix.allocMem[i];
}


MklMatrix::~MklMatrix()
{
	if (this->allocMem) 
		mkl_free(this->allocMem);

}

