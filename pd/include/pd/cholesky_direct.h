#pragma once
#include <pd/linear_sys_solver.h>
#include <Eigen/SparseCholesky>

namespace pd
{
	// parallel jacobi is a GPU-based vanilla Jacobi
	class CholeskyDirect : public LinearSystemSolver
	{
	public:
		void set_A(const Eigen::SparseMatrix<SimScalar>& A, const std::unordered_map<MeshIDType, DeformableMesh>& models) override;
		SimVectorX solve(const SimVectorX& b) override;
		void clear() override;

	private:
		Eigen::SimplicialCholesky<Eigen::SparseMatrix<SimScalar>> A_cholesky_decomp;
	};
}
