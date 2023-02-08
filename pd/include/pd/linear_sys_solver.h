#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

// interface
class LinearSystemSolver
{
public:
	LinearSystemSolver() {}
	LinearSystemSolver(int n_itr) : n_itr(n_itr) {}

	// Setting any value from coefficient matrix A that can be precomputed.
	// Precompute A-coefficient for A-Jacobi.
	virtual void set_A(const Eigen::SparseMatrix<float>& A) = 0;

	// Solve the linear system Ax = b, returns x.
	virtual Eigen::VectorXf solve(const Eigen::VectorXf& b) = 0;

	// free memories when system changes
	virtual void clear() = 0;

	// Number of iteration can be tuned realtime
	void set_n_itr(int n_itr) { this->n_itr = n_itr; }
	int n_itr{ 1 };
	
	// Indicates if has memory pointers
	bool is_allocated{ false };
};