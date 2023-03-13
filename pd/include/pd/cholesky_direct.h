#pragma once
#include <pd/linear_sys_solver.h>
#include <Eigen/SparseCholesky>

namespace pd
{
	// parallel jacobi is a GPU-based vanilla Jacobi
	class CholeskyDirect : public LinearSystemSolver
	{
	public:
		void set_A(const Eigen::SparseMatrix<float>& A, const std::unordered_map<int, DeformableMesh>& models) override;
		Eigen::VectorXf solve(const Eigen::VectorXf& b) override;
		void clear() override;

	private:
		Eigen::SimplicialCholesky<Eigen::SparseMatrix<float>> A_cholesky_decomp;
	};
}
