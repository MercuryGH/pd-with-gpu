#pragma once
#include <unordered_map>
#include <cuda_runtime.h>

#include <pd/types.h>

#include <pd/linear_sys_solver.h>

namespace pd
{
	class AJacobi: public LinearSystemSolver
	{
	public:
		AJacobi(int order): order(order) {}
		void set_A(const Eigen::SparseMatrix<SimScalar>& A, const std::unordered_map<MeshIDType, DeformableMesh>& models) override;

		SimVectorX solve(const SimVectorX& b) override;
		void clear() override;

		__global__ friend void itr_order_1(
			SimScalar* __restrict__ next_x,
			const SimScalar* __restrict__ x,
			SimScalar** __restrict__ d_1_ring_neighbors,
			int** __restrict__ d_1_ring_neighbor_indices,
			const int* __restrict__ d_1_ring_neighbor_sizes,
			const SimScalar* __restrict__ d_diagonals,
			const SimScalar* __restrict__ b_term,
			int n_vertex  // #Vertex, parallelism is n but not 3n
		);

		__global__ friend void itr_order_2(
			SimScalar* __restrict__ next_x_1,
			SimScalar* __restrict__ next_x_2,
			const SimScalar* __restrict__ x_1,
			const SimScalar* __restrict__ x_2,

			SimScalar** __restrict__ d_2_ring_neighbors,
			int** __restrict__ d_2_ring_neighbor_indices,
			const int* __restrict__ d_2_ring_neighbor_sizes,

			const SimScalar* __restrict__ d_diagonals, // D_ii

			const SimScalar* __restrict__ b_term,
			int n_vertex  // #Vertex
		);

		__global__ friend void itr_order_3(
			SimScalar* __restrict__ next_x_1,
			SimScalar* __restrict__ next_x_2,
			SimScalar* __restrict__ next_x_3,
			const SimScalar* __restrict__ x_1,
			const SimScalar* __restrict__ x_2,
			const SimScalar* __restrict__ x_3,

			SimScalar** __restrict__ d_1_ring_neighbors,
			int** __restrict__ d_1_ring_neighbor_indices,
			const int* __restrict__ d_1_ring_neighbor_sizes,

			SimScalar** __restrict__ d_2_ring_neighbors,
			int** __restrict__ d_2_ring_neighbor_indices,
			const int* __restrict__ d_2_ring_neighbor_sizes,

			SimScalar** __restrict__ d_3_ring_neighbors,
			int** __restrict__ d_3_ring_neighbor_indices,
			const int* __restrict__ d_3_ring_neighbor_sizes,

			const SimScalar* __restrict__ d_diagonals, // D_ii
			const SimScalar* __restrict__ b,
			int n_vertex  // #Vertex
		);

		// not used at runtime
		void set_order(int order)
		{
			this->order = order;
		}

	private:
		void precompute_A_jacobi(const Eigen::SparseMatrix<SimScalar>& A, const std::unordered_map<MeshIDType, DeformableMesh>& models);

		int n{ 0 };
		// A is an n * n with 3 * 3 block matrix
		// total space for x and b is 3 * #Vertex * sizeof(float) 
		// total space for A = D + B is at most (#Vertex * sizeof(float))^2 but not 3 times of it
		// A is a sparse matrix so there must be some optimization methods

		// CPU mem
		std::vector<std::unordered_map<VertexIndexType, SimScalar>> B; // B[i][j] = value
		std::vector<SimScalar> D;  // D[i][i] = 1 / A[i][i]

		constexpr static int A_JACOBI_MAX_ORDER = 3;

		// bridges between CPU mem and GPU mem.
		SimScalar** k_ring_neighbors[A_JACOBI_MAX_ORDER];
		int** k_ring_neighbor_indices[A_JACOBI_MAX_ORDER];

		// GPU mem
		// forall i, B_{is} ... B_{sj} for nonzero terms (reachable j). Total space = O(V + E)
		SimScalar** d_k_ring_neighbors[A_JACOBI_MAX_ORDER];
		int** d_k_ring_neighbor_indices[A_JACOBI_MAX_ORDER];
		int* d_k_ring_neighbor_sizes[A_JACOBI_MAX_ORDER]; // indicates the size of d_1_ring_neighbors

		// forall i, D_{ii}
		SimScalar* d_diagonals;

		SimScalar* d_b;
		SimScalar* d_b_term;

		int order{ 0 };
		SimScalar* d_x[A_JACOBI_MAX_ORDER];
		SimScalar* d_next_x[A_JACOBI_MAX_ORDER];
	};
}