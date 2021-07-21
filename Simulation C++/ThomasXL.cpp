#include <iostream>
#include <cmath>
#include <algorithm> 
#include "ThomasXL.h"

void print2dArray(double** array, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			std::cout << array[i][j] << "\t";
		}
		std::cout << "\n";
	}
}
void print1dArray(double* array, int n) {
	for (int i = 0; i < n; i++) {
		std::cout << array[i] << "\t";
	}
	std::cout << "\n";
}
ThomasXL::ThomasXL(int n, int numOfDiag) {
	this->n = n;
	nn = (numOfDiag - 1) / 2;
	//initialize a dynamic 2D array: Array of pointers 
	L = new double* [n];
	for (int i = 0; i < n; i++) {
 		L[i] = new double[n];
	}
	U = new double* [n];
	for (int i = 0; i < n; i++) {
		U[i] = new double[n];
	}
	Y = new double[n];
	//Initialize all values in L, Y and U to zero

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			L[i][j] = 0.0;
			U[i][j] = 0.0;
		}
		Y[i] = 0.0;
	}
	//Initialize diagonal value in L to 1

	for (int i = 0; i < n; i++) {
		L[i][i] = 1.0;
	}

}

void ThomasXL::LUDecomposition(double** A) {
	int i = 0, j = 0, k = 0;
	for (j = 0; j < n; j++) {
		for (i = std::max(0,j-nn); i < j + 1; i++) {
 			U[i][j] = A[i][j];
			for (k = std::max(0,j-nn); k < i; k++)
				U[i][j] -= L[i][k] * U[k][j];
		}
		for (i = j + 1; i < std::min(n,j+nn+1); i++) {
 			L[i][j] = A[i][j];

 			for (k = std::max(0,i-nn); k < j; k++) {
				L[i][j] -= L[i][k] * U[k][j];
			}
			L[i][j] /= U[j][j];
			//std::cout << U[j][j]<<"\t"<<L[i][j] << "\n";
		}
		L[j][j] = 1.0;

	}
	//std::cout << "L array is\n ";
	//print2dArray(L, n);

	//std::cout << "\nU array is\n ";
	//print2dArray(U, n);
	
}

void ThomasXL::ForwardSubstitution(double* b) {

	for (int i = 0; i < n; i++) {
		Y[i] = b[i];
		for (int k = std::max(0,i-nn); k < i; k++) {
			Y[i] -= L[i][k] * Y[k];
		}
	}

}
void ThomasXL::BackwardSubstitution(double* x) {
	for (int i = n - 1; i >= 0; i--) {
		x[i] = Y[i];
		for (int k = i + 1; k < std::min(n,i+nn+1); k++) {
			x[i] -= U[i][k] * x[k];
		} 
		x[i] /= U[i][i];
	}

}

void ThomasXL::solve(double** A, double* b, double* x) {
	LUDecomposition(A);
	ForwardSubstitution(b);
	BackwardSubstitution(x);
}

