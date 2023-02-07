#pragma once
#include <cuda_runtime.h>

#include <pd/linear_sys_solver.h>

namespace pd
{
	class AJacobi: public LinearSystemSolver
	{
	public:
		void set_A(const Eigen::SparseMatrix<float>& A) override;
		Eigen::VectorXf solve(const Eigen::VectorXf& b) override;

	private:
		
	};
}