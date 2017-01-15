#pragma once
#include "MklMatrix.h"

struct RegressionData
{
	MklMatrix y, z;

	static RegressionData readFromFile(const char* path);
	RegressionData(MklMatrix &&y, MklMatrix &&z);
	RegressionData(RegressionData &data);
};

