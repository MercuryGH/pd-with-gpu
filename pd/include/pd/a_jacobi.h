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
			float** __restrict__ d_1_ring_neighbors,
			int** __restrict__ d_1_ring_neighbor_indices,
			const int* __restrict__ d_1_ring_neighbor_sizes,
			const float* __restrict__ d_diagonals,
			const float* __restrict__ b,
			int n  // #Vertex, parallelism is n but not 3n
		);

		__global__ friend void itr_order_2(
			float* next_x_1,
			float* next_x_2,
			const float* __restrict__ x_1,
			const float* __restrict__ x_2,
			float** __restrict__ d_1_ring_neighbors,
			int** __restrict__ d_1_ring_neighbor_indices,
			const int* __restrict__ d_1_ring_neighbor_sizes,
			float** __restrict__ d_2_ring_neighbors,
			int** __restrict__ d_2_ring_neighbor_indices,
			const int* __restrict__ d_2_ring_neighbor_sizes,
			const float* __restrict__ d_diagonals,
			const float* __restrict__ d_diagonal_adj_sums,
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

		constexpr static int A_JACOBI_MAX_ORDER = 3;

		// bridges between CPU mem and GPU mem.
		float** k_ring_neighbors[A_JACOBI_MAX_ORDER];
		int** k_ring_neighbor_indices[A_JACOBI_MAX_ORDER];

		// GPU mem
		// forall i, B_{is} ... B_{sj} for nonzero terms (reachable j). Total space = O(V + E)
		float** d_k_ring_neighbors[A_JACOBI_MAX_ORDER];
		int** d_k_ring_neighbor_indices[A_JACOBI_MAX_ORDER];
		int* d_k_ring_neighbor_sizes[A_JACOBI_MAX_ORDER]; // indicates the size of d_1_ring_neighbors

		// forall i, D_{ii} ... D_{ss}
		float* d_diagonals[A_JACOBI_MAX_ORDER];

		// sum of D_{ss} forall s adjacent to i
		//float* d_diagonal_adj_sums; // use only in order = 2
		//float* d_diagonal_adj_sums_2; // use only in order = 3

		float* d_b;

		int order{ 1 };
		// sizeof(d_x) == sizeof(d_next_x) == order
		//float* d_x;
		float* d_x[A_JACOBI_MAX_ORDER];
		float* d_next_x[A_JACOBI_MAX_ORDER];
	};
}