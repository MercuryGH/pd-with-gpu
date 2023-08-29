#pragma once
#include <pd/linear_sys_solver.h>

namespace pd
{
	// parallel jacobi is a GPU-based vanilla Jacobi
	class ParallelJacobi : public LinearSystemSolver
	{
	public:
		ParallelJacobi() : LinearSystemSolver(1000) {}

		void set_A(const Eigen::SparseMatrix<SimScalar>& A, const std::unordered_map<MeshIDType, DeformableMesh>& models) override;
		SimVectorX solve(const SimVectorX& b) override;
		void clear() override;

		__global__ friend void itr_shfl_down(SimScalar* __restrict__ next_x, const SimScalar* __restrict__ A, const SimScalar* __restrict__ x, const SimScalar* __restrict__ b, int n_row, int n_col);
		__global__ friend void itr_normal(SimScalar* __restrict__ next_x, const SimScalar* __restrict__ A, const SimScalar* __restrict__ x, const SimScalar* __restrict__ b, int n_row, int n_col);

	private:
		int n{ 0 };

		// device memory pointers, uses dense representation
		SimScalar* d_A{ nullptr };
		SimScalar* d_b{ nullptr };
		SimScalar* d_x{ nullptr };
		SimScalar* d_next_x{ nullptr };
	};
}
