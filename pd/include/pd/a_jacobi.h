#pragma once
#include <unordered_map>
#include <cuda_runtime.h>

#include <pd/linear_sys_solver.h>

namespace pd
{
	class AJacobi: public LinearSystemSolver
	{
	public:
		void set_A(const Eigen::SparseMatrix<float>& A, const pd::Constraints& constraints) override;

		Eigen::VectorXf solve(const Eigen::VectorXf& b) override;
		void clear() override;

		__global__ friend void itr_order_1(
			float* next_x,
			const float* __restrict__ x,
			float** __restrict__ d_a_products,
			int** __restrict__ d_a_products_idx,
			const int* __restrict__ d_a_product_sizes,
			const float* __restrict__ d_diagonals,
			const float* __restrict__ b,
			int n  // #Vertex, parallelism is n but not 3n
		);

		__global__ friend void itr_order_2(
			float* next_x_1,
			float* next_x_2,
			const float* __restrict__ x_1,
			const float* __restrict__ x_2,
			float** __restrict__ d_a_products,
			int** __restrict__ d_a_products_idx,
			const int* __restrict__ d_a_product_sizes,
			const float* __restrict__ d_diagonals,
			const float* __restrict__ b,
			int n  // #Vertex
		);

		// not used at runtime
		void set_order(int order)
		{
			this->order = order;
		}

	private:
		void precompute_A_jacobi(const Eigen::SparseMatrix<float>& A, const pd::Constraints& constraints);

	private:
		int n{ 0 };
		// A is an n * n with 3 * 3 block matrix
		// total space for x and b is 3 * #Vertex * sizeof(float) 
		// total space for A = D + B is at most (#Vertex * sizeof(float))^2 but not 3 times of it
		// A is a sparse matrix so there must be some optimization methods

		// CPU mem
		std::vector<std::unordered_map<int, float>> B; // B[i][j] = value
		std::vector<float> D;  // D[i][i] = 1 / A[i][i]

		// bridges between CPU mem and GPU mem.
		float** b_a_products;
		int** b_a_products_idx;

		// GPU mem
		// forall i, B_{is} ... B_{sj} for nonzero terms (reachable j). Total space = O(V + E)
		float** d_a_products; 
		int** d_a_products_idx;
		int* d_a_product_sizes; // indicates the size of d_a_products[i]

		// forall i, D_{ii} ... D_{ss}
		float* d_diagonals;  

		float* d_b;

		int order{ 1 };
		// sizeof(d_x) == sizeof(d_next_x) == order
		//float* d_x;
		std::vector<float*> d_x;
		std::vector<float*> d_next_x;
	};
}