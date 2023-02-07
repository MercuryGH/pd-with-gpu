#pragma once
#include <cuda_runtime.h>
#include <pd/linear_sys_solver.h>

namespace pd
{
	// parallel jacobi is a GPU-based vanilla Jacobi
	class ParallelJacobi : public LinearSystemSolver
	{
	public:
		ParallelJacobi() : LinearSystemSolver(1000) {}

		void set_A(const Eigen::SparseMatrix<float>& A) override;
		Eigen::VectorXf solve(const Eigen::VectorXf& b) override;

		__global__ friend void itr_shfl_down(float* next_x, const float* __restrict__ A, const float* __restrict__  x, const float* __restrict__ b, int n_row, int n_col);
		__global__ friend void itr_normal(float* next_x, const float* __restrict__ A, const float* __restrict__  x, const float* __restrict__ b, int n_row, int n_col);
	private:
		constexpr static int WARP_SIZE{ 32 };
		int n{ 0 };

		// device memory pointers, uses dense representation
		float* d_A{ nullptr };
		float* d_b{ nullptr };
		float* d_x{ nullptr };
		float* d_next_x{ nullptr };
	};
}