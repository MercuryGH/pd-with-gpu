#pragma once
#include <unordered_map>

#include <pd/types.h>
#include <pd/linear_sys_solver.h>

namespace pd
{
	class AJacobi: public LinearSystemSolver
	{
	public:
		AJacobi(int order): order(order) {}
		AJacobi(int order, SimScalar rho, SimScalar under_relaxation): order(order), rho(rho), under_relaxation(under_relaxation) {}
		void set_A(const Eigen::SparseMatrix<SimScalar>& A, const std::unordered_map<MeshIDType, DeformableMesh>& models) override;

		SimVectorX solve(const SimVectorX& b) override;
		void clear() override;

		__global__ friend void itr_order_1(
			SimScalar* __restrict__ next_x,
			const SimScalar* __restrict__ x,
			const SimScalar* __restrict__ prev_x,
			SimScalar** __restrict__ d_1_ring_neighbors,
			VertexIndexType** __restrict__ d_1_ring_neighbor_indices,
			const int* __restrict__ d_1_ring_neighbor_sizes,
			const SimScalar* __restrict__ d_diagonals,
			const SimScalar* __restrict__ d_b_term,
			int n_vertex,  // #Vertex, parallelism is n but not 3n
			SimScalar omega, // chebyshev param omega
			SimScalar under_relaxation // under-relaxation coeff
		);

		__global__ friend void itr_order_2(
			SimScalar* __restrict__ next_x_1,
			SimScalar* __restrict__ next_x_2,
			const SimScalar* __restrict__ x_1,
			const SimScalar* __restrict__ x_2,
			const SimScalar* __restrict__ prev_x_1,
			const SimScalar* __restrict__ prev_x_2,

			SimScalar** __restrict__ d_2_ring_neighbors,
			VertexIndexType** __restrict__ d_2_ring_neighbor_indices,
			const int* __restrict__ d_2_ring_neighbor_sizes,

			const SimScalar* __restrict__ d_diagonals, // D_ii

			const SimScalar* __restrict__ b_term,
			int n_vertex,  // #Vertex
			SimScalar omega, // chebyshev param omega
			SimScalar under_relaxation // under-relaxation coeff
		);

		__global__ friend void itr_order_3(
			SimScalar* __restrict__ next_x_1,
			SimScalar* __restrict__ next_x_2,
			SimScalar* __restrict__ next_x_3,
			const SimScalar* __restrict__ x_1,
			const SimScalar* __restrict__ x_2,
			const SimScalar* __restrict__ x_3,
			const SimScalar* __restrict__ prev_x_1,
			const SimScalar* __restrict__ prev_x_2,
			const SimScalar* __restrict__ prev_x_3,

			SimScalar** __restrict__ d_3_ring_neighbors,
			VertexIndexType** __restrict__ d_3_ring_neighbor_indices,
			const int* __restrict__ d_3_ring_neighbor_sizes,

			const SimScalar* __restrict__ d_diagonals, // D_ii
			const SimScalar* __restrict__ b,
			int n_vertex,  // #Vertex
			SimScalar omega, // chebyshev param omega
			SimScalar under_relaxation // under-relaxation coeff
		);

		// not used at runtime
		void set_order(int order)
		{
			this->order = order;
		}

		void set_params(SimScalar rho, SimScalar under_relaxation)
		{
			this->rho = rho;
			this->under_relaxation = under_relaxation;
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
		VertexIndexType** k_ring_neighbor_indices[A_JACOBI_MAX_ORDER];

		// GPU mem
		// forall i, B_{is} ... B_{sj} for nonzero terms (reachable j). Total space = O(V + E)
		SimScalar** d_k_ring_neighbors[A_JACOBI_MAX_ORDER];
		VertexIndexType** d_k_ring_neighbor_indices[A_JACOBI_MAX_ORDER];
		int* d_k_ring_neighbor_sizes[A_JACOBI_MAX_ORDER]; // indicates the size of d_1_ring_neighbors

		// forall i, D_{ii}
		SimScalar* d_diagonals;

		SimScalar* d_b;
		SimScalar* d_b_term;

		int order{ 0 };
		SimScalar* d_prev_x[A_JACOBI_MAX_ORDER];
		SimScalar* d_x[A_JACOBI_MAX_ORDER];
		SimScalar* d_next_x[A_JACOBI_MAX_ORDER];

		// chebyshev params
		SimScalar rho{ 0.9992 };
		SimScalar under_relaxation{ 1 };
	};
}
