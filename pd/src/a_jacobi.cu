#include <array>

#include <pd/a_jacobi.h>
#include <util/gpu_helper.h>
#include <util/adj_list_graph.h>

namespace pd
{
	__global__ void precompute_b_term_1(
		SimScalar* __restrict__ d_b_term,
		const SimScalar* __restrict__ d_b,
		const SimScalar* __restrict__ d_diagonals,
		int n_vertex
	)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < n_vertex)
		{
			SimScalar D_ii_inv = d_diagonals[idx];

			d_b_term[3 * idx] = d_b[3 * idx] * D_ii_inv;
			d_b_term[3 * idx + 1] = d_b[3 * idx + 1] * D_ii_inv;
			d_b_term[3 * idx + 2] = d_b[3 * idx + 2] * D_ii_inv;
		}
	}

	__device__ void chebyshev_acc(
		SimScalar* __restrict__ next_x,
		const SimScalar* __restrict__ x,
		const SimScalar* __restrict__ prev_x,	
		SimScalar omega, // chebyshev param omega 
		SimScalar under_relaxation, // under-relaxation coeff	
		int _3_idx
	)
	{
		for (int i = 0; i < 3; i++)
		{
			next_x[_3_idx + i] = omega * (((next_x[_3_idx + i] - x[_3_idx + i]) * under_relaxation + x[_3_idx + i]) - prev_x[_3_idx + i]) + prev_x[_3_idx + i];
		}	
	}

	__global__ void itr_order_1(
		SimScalar* __restrict__ next_x,
		const SimScalar* __restrict__ x,
		const SimScalar* __restrict__ prev_x,
		SimScalar** __restrict__ d_1_ring_neighbors,
		int** __restrict__ d_1_ring_neighbor_indices,
		const int* __restrict__ d_1_ring_neighbor_sizes,
		const SimScalar* __restrict__ d_diagonals,
		const SimScalar* __restrict__ d_b_term,
		int n_vertex,  // #Vertex, parallelism is n but not 3n
		SimScalar omega, // chebyshev param omega 
		SimScalar under_relaxation // under-relaxation coeff
	)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < n_vertex)
		{
			const SimScalar* B_ijs = d_1_ring_neighbors[idx];
			const int* js = d_1_ring_neighbor_indices[idx];

			SimScalar sum_0 = 0.0f;
			SimScalar sum_1 = 0.0f;
			SimScalar sum_2 = 0.0f;

			int j_cnt = d_1_ring_neighbor_sizes[idx];
			for (int k = 0; k < j_cnt; k++)
			{
				SimScalar B_ij = B_ijs[k];
				int j = js[k];

				sum_0 += B_ij * x[3 * j];
				sum_1 += B_ij * x[3 * j + 1];
				sum_2 += B_ij * x[3 * j + 2];
			}

			SimScalar D_ii_inv = d_diagonals[idx];

			next_x[3 * idx] = sum_0 * D_ii_inv + d_b_term[3 * idx];
			next_x[3 * idx + 1] = sum_1 * D_ii_inv + d_b_term[3 * idx + 1];
			next_x[3 * idx + 2] = sum_2 * D_ii_inv + d_b_term[3 * idx + 2];

			// debug only
			// printf("Debug: %d %f %f %f\n", idx, next_x[3 * idx], next_x[3 * idx + 1], next_x[3 * idx + 2]);

			// chebyshev
			chebyshev_acc(next_x, x, prev_x, omega, under_relaxation, 3 * idx);
		}
	}

	__global__ void precompute_b_term_2(
		SimScalar* __restrict__ d_b_term,
		const SimScalar* __restrict__ d_b,

		SimScalar** __restrict__ d_1_ring_neighbors,
		int** __restrict__ d_1_ring_neighbor_indices,
		const int* __restrict__ d_1_ring_neighbor_sizes,

		const SimScalar* __restrict__ d_diagonals,
		int n_vertex
	)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < n_vertex)
		{
			// b_terms:
			const SimScalar* B_ijs_b = d_1_ring_neighbors[idx];
			const int* js_b = d_1_ring_neighbor_indices[idx];

			SimScalar b_term_0 = 0.0f;
			SimScalar b_term_1 = 0.0f;
			SimScalar b_term_2 = 0.0f;

			int j_cnt_b = d_1_ring_neighbor_sizes[idx];
			for (int k = 0; k < j_cnt_b; k++)
			{
				SimScalar B_ij = B_ijs_b[k];
				int j = js_b[k];

				b_term_0 += B_ij * d_b[3 * j];
				b_term_1 += B_ij * d_b[3 * j + 1];
				b_term_2 += B_ij * d_b[3 * j + 2];
			}

			SimScalar D_ii_inv = d_diagonals[idx];

			// may contains floating point precision problem
			d_b_term[3 * idx] = (d_b[3 * idx] + b_term_0) * D_ii_inv;
			d_b_term[3 * idx + 1] = (d_b[3 * idx + 1] + b_term_1) * D_ii_inv;
			d_b_term[3 * idx + 2] = (d_b[3 * idx + 2] + b_term_2) * D_ii_inv;
		}
	}

	__global__ void itr_order_2(
		SimScalar* __restrict__ next_x_1,
		SimScalar* __restrict__ next_x_2,
		const SimScalar* __restrict__ x_1,
		const SimScalar* __restrict__ x_2,
		const SimScalar* __restrict__ prev_x_1,
		const SimScalar* __restrict__ prev_x_2,

		SimScalar** __restrict__ d_2_ring_neighbors,
		int** __restrict__ d_2_ring_neighbor_indices,
		const int* __restrict__ d_2_ring_neighbor_sizes,

		const SimScalar* __restrict__ d_diagonals, // D_ii

		const SimScalar* __restrict__ d_b_term,
		int n_vertex,  // #Vertex
		SimScalar omega, // chebyshev param omega 
		SimScalar under_relaxation // under-relaxation coeff
	)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < n_vertex)
		{
			const SimScalar* B_issjs = d_2_ring_neighbors[idx];
			const int* js = d_2_ring_neighbor_indices[idx];

			SimScalar sum_0_1 = 0.0f;
			SimScalar sum_1_1 = 0.0f;
			SimScalar sum_2_1 = 0.0f;

			SimScalar sum_0_2 = 0.0f;
			SimScalar sum_1_2 = 0.0f;
			SimScalar sum_2_2 = 0.0f;

			int j_cnt = d_2_ring_neighbor_sizes[idx];
			for (int k = 0; k < j_cnt; k++)
			{
				SimScalar B_issj = B_issjs[k];
				int j = js[k];

				sum_0_1 += B_issj * x_1[3 * j];
				sum_1_1 += B_issj * x_1[3 * j + 1];
				sum_2_1 += B_issj * x_1[3 * j + 2];

				sum_0_2 += B_issj * x_2[3 * j];
				sum_1_2 += B_issj * x_2[3 * j + 1];
				sum_2_2 += B_issj * x_2[3 * j + 2];
			}

			SimScalar D_ii_inv = d_diagonals[idx];

			next_x_1[3 * idx] = sum_0_1 * D_ii_inv + d_b_term[3 * idx]; 
			next_x_1[3 * idx + 1] = sum_1_1 * D_ii_inv + d_b_term[3 * idx + 1];
			next_x_1[3 * idx + 2] = sum_2_1 * D_ii_inv + d_b_term[3 * idx + 2];

			next_x_2[3 * idx] = sum_0_2 * D_ii_inv + d_b_term[3 * idx];
			next_x_2[3 * idx + 1] = sum_1_2 * D_ii_inv + d_b_term[3 * idx + 1];
			next_x_2[3 * idx + 2] = sum_2_2 * D_ii_inv + d_b_term[3 * idx + 2];

			// chebyshev
			chebyshev_acc(next_x_1, x_1, prev_x_1, omega, under_relaxation, 3 * idx);
			chebyshev_acc(next_x_2, x_2, prev_x_2, omega, under_relaxation, 3 * idx);
		}
	}

	__global__ void precompute_b_term_3(
		SimScalar* __restrict__ d_b_term,
		const SimScalar* __restrict__ d_b,

		SimScalar** __restrict__ d_1_ring_neighbors,
		int** __restrict__ d_1_ring_neighbor_indices,
		const int* __restrict__ d_1_ring_neighbor_sizes,

		SimScalar** __restrict__ d_2_ring_neighbors,
		int** __restrict__ d_2_ring_neighbor_indices,
		const int* __restrict__ d_2_ring_neighbor_sizes,

		const SimScalar* __restrict__ d_diagonals,
		int n_vertex
	)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < n_vertex)
		{
			// b_terms_1:
			const SimScalar* B_ijs_b = d_1_ring_neighbors[idx];
			const int* js_b_1 = d_1_ring_neighbor_indices[idx];

			SimScalar b_term_0_1 = 0.0f;
			SimScalar b_term_1_1 = 0.0f;
			SimScalar b_term_2_1 = 0.0f;

			int j_cnt_b_1 = d_1_ring_neighbor_sizes[idx];
			for (int k = 0; k < j_cnt_b_1; k++)
			{
				SimScalar B_ij = B_ijs_b[k];
				int j = js_b_1[k];

				b_term_0_1 += B_ij * d_b[3 * j];
				b_term_1_1 += B_ij * d_b[3 * j + 1];
				b_term_2_1 += B_ij * d_b[3 * j + 2];
			}

			// b_terms_2:
			const SimScalar* B_issjs_b = d_2_ring_neighbors[idx];
			const int* js_b_2 = d_2_ring_neighbor_indices[idx];

			SimScalar b_term_0_2 = 0.0f;
			SimScalar b_term_1_2 = 0.0f;
			SimScalar b_term_2_2 = 0.0f;

			int j_cnt_b_2 = d_2_ring_neighbor_sizes[idx];
			for (int k = 0; k < j_cnt_b_2; k++)
			{
				SimScalar B_issj = B_issjs_b[k];
				int j = js_b_2[k];

				b_term_0_2 += B_issj * d_b[3 * j];
				b_term_1_2 += B_issj * d_b[3 * j + 1];
				b_term_2_2 += B_issj * d_b[3 * j + 2];
			}

			SimScalar D_ii_inv = d_diagonals[idx];

			// may contains SimScalaring point precision problem
			d_b_term[3 * idx] = (d_b[3 * idx] + b_term_0_1 + b_term_0_2) * D_ii_inv;
			d_b_term[3 * idx + 1] = (d_b[3 * idx + 1] + b_term_1_1 + b_term_1_2) * D_ii_inv;
			d_b_term[3 * idx + 2] = (d_b[3 * idx + 2] + b_term_2_1 + b_term_2_2) * D_ii_inv;
		}
	}

	__global__ void itr_order_3(
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
		int** __restrict__ d_3_ring_neighbor_indices,
		const int* __restrict__ d_3_ring_neighbor_sizes,

		const SimScalar* __restrict__ d_diagonals, // D_ii
		const SimScalar* __restrict__ d_b_term,
		int n_vertex,  // #Vertex
		SimScalar omega, // chebyshev param omega 
		SimScalar under_relaxation // under-relaxation coeff
	)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < n_vertex)
		{
			const SimScalar* B_ittssjs = d_3_ring_neighbors[idx];
			const int* js = d_3_ring_neighbor_indices[idx];

			SimScalar sum_0_1 = 0.0f;
			SimScalar sum_1_1 = 0.0f;
			SimScalar sum_2_1 = 0.0f;

			SimScalar sum_0_2 = 0.0f;
			SimScalar sum_1_2 = 0.0f;
			SimScalar sum_2_2 = 0.0f;

			SimScalar sum_0_3 = 0.0f;
			SimScalar sum_1_3 = 0.0f;
			SimScalar sum_2_3 = 0.0f;

			int j_cnt = d_3_ring_neighbor_sizes[idx];
			for (int k = 0; k < j_cnt; k++)
			{
				SimScalar B_ittssj = B_ittssjs[k];
				int j = js[k];

				sum_0_1 += B_ittssj * x_1[3 * j];
				sum_1_1 += B_ittssj * x_1[3 * j + 1];
				sum_2_1 += B_ittssj * x_1[3 * j + 2];

				sum_0_2 += B_ittssj * x_2[3 * j];
				sum_1_2 += B_ittssj * x_2[3 * j + 1];
				sum_2_2 += B_ittssj * x_2[3 * j + 2];

				sum_0_3 += B_ittssj * x_3[3 * j];
				sum_1_3 += B_ittssj * x_3[3 * j + 1];
				sum_2_3 += B_ittssj * x_3[3 * j + 2];
			}

			SimScalar D_ii_inv = d_diagonals[idx];

			next_x_1[3 * idx] = sum_0_1 * D_ii_inv + d_b_term[3 * idx];
			next_x_1[3 * idx + 1] = sum_1_1 * D_ii_inv + d_b_term[3 * idx + 1];
			next_x_1[3 * idx + 2] = sum_2_1 * D_ii_inv + d_b_term[3 * idx + 2];

			next_x_2[3 * idx] = sum_0_2 * D_ii_inv + d_b_term[3 * idx];
			next_x_2[3 * idx + 1] = sum_1_2 * D_ii_inv + d_b_term[3 * idx + 1];
			next_x_2[3 * idx + 2] = sum_2_2 * D_ii_inv + d_b_term[3 * idx + 2];

			next_x_3[3 * idx] = sum_0_3 * D_ii_inv + d_b_term[3 * idx];
			next_x_3[3 * idx + 1] = sum_1_3 * D_ii_inv + d_b_term[3 * idx + 1];
			next_x_3[3 * idx + 2] = sum_2_3 * D_ii_inv + d_b_term[3 * idx + 2];

			// chebyshev
			chebyshev_acc(next_x_1, x_1, prev_x_1, omega, under_relaxation, 3 * idx);
			chebyshev_acc(next_x_2, x_2, prev_x_2, omega, under_relaxation, 3 * idx);
			chebyshev_acc(next_x_3, x_3, prev_x_3, omega, under_relaxation, 3 * idx);
		}
	}

	void AJacobi::clear()
	{
		if (is_allocated)
		{
			// free memory
			B.clear();
			D.clear();
			checkCudaErrors(cudaFree(d_b));
			checkCudaErrors(cudaFree(d_b_term));
			for (int k = 0; k < order; k++)
			{
				checkCudaErrors(cudaFree(d_prev_x[k]));
				checkCudaErrors(cudaFree(d_x[k]));
				checkCudaErrors(cudaFree(d_next_x[k]));
			}
			checkCudaErrors(cudaFree(d_diagonals));

			const int n_vertex = n / 3;
			for (int k = 0; k < order; k++)
			{
				for (int i = 0; i < n_vertex; i++)
				{
					checkCudaErrors(cudaFree(k_ring_neighbors[k][i]));
					checkCudaErrors(cudaFree(k_ring_neighbor_indices[k][i]));
				}
				free(k_ring_neighbors[k]);
				free(k_ring_neighbor_indices[k]);
			}
			for (int k = 0; k < order; k++)
			{
				checkCudaErrors(cudaFree(d_k_ring_neighbors[k]));
				checkCudaErrors(cudaFree(d_k_ring_neighbor_indices[k]));
				checkCudaErrors(cudaFree(d_k_ring_neighbor_sizes[k]));
			}

			is_allocated = false;
		}
	}

	// Eigen::SparseMatrix can be converted to CUDA sparse matrix but it's quite tricky.
	// Instead we construct sparse matrix using adjacent table by ourselves
	void AJacobi::set_A(const Eigen::SparseMatrix<SimScalar>& A, const std::unordered_map<MeshIDType, DeformableMesh>& models)
	{
		n = A.rows();
		assert((n / 3) * 3 == n);
		if (order > A_JACOBI_MAX_ORDER || order < 1)
		{
			printf("Error: Invalid A-Jacobi order!\n");
			assert(false);
		}

		checkCudaErrors(cudaMalloc((void**)&d_b, sizeof(SimScalar) * n));
		checkCudaErrors(cudaMalloc((void**)&d_b_term, sizeof(SimScalar) * n));
		for (int i = 0; i < order; i++)
		{
			checkCudaErrors(cudaMalloc((void**)&d_prev_x[i], sizeof(SimScalar) * n));
			checkCudaErrors(cudaMalloc((void**)&d_x[i], sizeof(SimScalar) * n));
			checkCudaErrors(cudaMalloc((void**)&d_next_x[i], sizeof(SimScalar) * n));
		}
		// set precomputation values
		precompute_A_jacobi(A, models);

		is_allocated = true;
	}

	void AJacobi::precompute_A_jacobi(const Eigen::SparseMatrix<SimScalar>& A, const std::unordered_map<MeshIDType, DeformableMesh>& models)
	{
		// fill the member
		const int n_vertex = n / 3;

		const auto get_B_ij = [&](int i, int j) {
			assert(A.coeff(i * 3, j * 3) == A.coeff(i * 3 + 1, j * 3 + 1));
			assert(A.coeff(i * 3 + 1, j * 3 + 1) == A.coeff(i * 3 + 2, j * 3 + 2));
			return A.coeff(i * 3, j * 3);
		};

		const auto get_D_ii = [&](int i) {
			return A.coeff(i * 3, i * 3);
		};

		std::vector<std::pair<int, int>> edges;

		int acc = 0;
		for (const auto& [id, model] : models)
		{
			int n = model.positions().rows();
			const Eigen::MatrixX2i model_edges = model.get_edges();

			for (int i = 0; i < model_edges.rows(); i++)
			{
				const auto edge = model_edges.row(i);
				const auto vi = edge(0);
				const auto vj = edge(1);
				
				edges.emplace_back(acc + vi, acc + vj);
			}

			acc += n;
		}
		util::AdjListGraph adj_list_graph(edges, n_vertex);
		const std::vector<std::unordered_set<VertexIndexType>>& adj_list = adj_list_graph.get_adj_list();

		D.resize(n_vertex);
		B.resize(n_vertex);

		// B and D precomputation
		for (int i = 0; i < n_vertex; i++)
		{
			for (VertexIndexType j : adj_list[i])
			{
				// Note: negative sign!!!
				B[i][j] = -get_B_ij(i, j);
			}
			D[i] = 1.0f / get_D_ii(i);
			// if (std::abs(D[i]) < 1e-6f)
			// 	printf("Warning: i = %d, D[i] = %f\n", i, D[i]);
		}

		// --- Compute d_diagonals
		checkCudaErrors(cudaMalloc((void**)&d_diagonals, sizeof(SimScalar) * n_vertex));
		checkCudaErrors(cudaMemcpy(d_diagonals, D.data(), sizeof(SimScalar) * n_vertex, cudaMemcpyHostToDevice));

		const auto get_vertex_k_ring_neighbors = [&](
			int i,
			int ring_width,
			std::vector<VertexIndexType>& neighbor_indices,
			std::vector<SimScalar>& neighbor_B_vals
		) {
			if (ring_width == 1)
			{
				neighbor_indices.reserve(B[i].size());
				neighbor_B_vals.reserve(B[i].size());
				for (const auto& [k, v] : B[i])
				{
					neighbor_indices.push_back(k);
					neighbor_B_vals.push_back(order >= 2 ? v * D[k] : v);
				}
			}
			// compute D_{ss}^{-1} * B_{is}B_{sj} for l = 2
			if (ring_width == 2)
			{
				std::unordered_map<VertexIndexType, SimScalar> a_products_on_vertex;
				for (const auto& [s, v1] : B[i])
				{
					for (const auto& [j, v2] : B[s])
					{
						// If D cannot be precomputed, times D is done later.
						if (order == 2)
						{
							a_products_on_vertex[j] += v1 * v2 * D[s];
						}
						else if (order == 3)
						{
							a_products_on_vertex[j] += v1 * v2 * D[s] * D[j];
						}
						else
						{
							a_products_on_vertex[j] += v1 * v2;
						}
					}
				}

				neighbor_indices.reserve(a_products_on_vertex.size());
				neighbor_B_vals.reserve(a_products_on_vertex.size());
				for (const auto& [k, v] : a_products_on_vertex)
				{
					neighbor_indices.push_back(k);
					neighbor_B_vals.push_back(v);
				}
			}
			if (ring_width == 3)
			{
				std::unordered_map<VertexIndexType, SimScalar> a_products_on_vertex;
				for (const auto& [t, v1] : B[i])
				{
					for (const auto& [s, v2] : B[t])
					{
						for (const auto& [j, v3] : B[s])
						{
							// If D cannot be precomputed, times D is done later.
							if (order == 3)
							{
								a_products_on_vertex[j] += v1 * v2 * v3 * D[t] * D[s];
							}
							else
							{
								a_products_on_vertex[j] += v1 * v2 * v3;
							}
						}
					}
				}

				neighbor_indices.reserve(a_products_on_vertex.size());
				neighbor_B_vals.reserve(a_products_on_vertex.size());
				for (const auto& [k, v] : a_products_on_vertex)
				{
					neighbor_indices.push_back(k);
					neighbor_B_vals.push_back(v);
				}
			}
		};

		std::array<std::vector<int>, A_JACOBI_MAX_ORDER> k_ring_neighbor_sizes;
		for (int k = 0; k < order; k++)
		{
			// --- Compute d_a_products, d_a_products_idx and d_a_products_sizes
			checkCudaErrors(cudaMalloc((void***)&d_k_ring_neighbors[k], sizeof(SimScalar*) * n_vertex));
			checkCudaErrors(cudaMalloc((void***)&d_k_ring_neighbor_indices[k], sizeof(int*) * n_vertex));
			checkCudaErrors(cudaMalloc((void**)&d_k_ring_neighbor_sizes[k], sizeof(int) * n_vertex));
			// to avoid indexing device memory on the host, use tmp host memory
			k_ring_neighbors[k] = (SimScalar**)malloc(sizeof(SimScalar*) * n_vertex);
			k_ring_neighbor_indices[k] = (int**)malloc(sizeof(int*) * n_vertex);
			k_ring_neighbor_sizes[k].resize(n_vertex);

			for (int i = 0; i < n_vertex; i++)
			{
				std::vector<int> neighbor_indices;
				std::vector<SimScalar> neighbor_B_vals;
				get_vertex_k_ring_neighbors(i, k + 1, neighbor_indices, neighbor_B_vals);
				const int n_adj_vertex = neighbor_indices.size();
				checkCudaErrors(
					cudaMalloc((void**)&k_ring_neighbors[k][i], sizeof(SimScalar) * n_adj_vertex));
				checkCudaErrors(
					cudaMalloc((void**)&k_ring_neighbor_indices[k][i], sizeof(int) * n_adj_vertex));
				k_ring_neighbor_sizes[k][i] = n_adj_vertex;

				checkCudaErrors(
					cudaMemcpy(k_ring_neighbors[k][i], neighbor_B_vals.data(), sizeof(SimScalar) * n_adj_vertex, cudaMemcpyHostToDevice));
				checkCudaErrors(
					cudaMemcpy(k_ring_neighbor_indices[k][i], neighbor_indices.data(), sizeof(int) * n_adj_vertex, cudaMemcpyHostToDevice));
			}

			// copy memory
			checkCudaErrors(
				cudaMemcpy(d_k_ring_neighbors[k], k_ring_neighbors[k], sizeof(SimScalar*) * n_vertex, cudaMemcpyHostToDevice));
			checkCudaErrors(
				cudaMemcpy(d_k_ring_neighbor_indices[k], k_ring_neighbor_indices[k], sizeof(int*) * n_vertex, cudaMemcpyHostToDevice));
			checkCudaErrors(
				cudaMemcpy(d_k_ring_neighbor_sizes[k], k_ring_neighbor_sizes[k].data(), sizeof(int) * n_vertex, cudaMemcpyHostToDevice));
		}
	}

	SimVectorX AJacobi::solve(const SimVectorX& b)
	{
		SimVectorX ret;
		ret.resizeLike(b);

		checkCudaErrors(cudaMemcpy(d_b, b.data(), sizeof(SimScalar) * n, cudaMemcpyHostToDevice));
		for (int i = 0; i < order; i++)
		{
			cudaMemset(d_prev_x[i], 0, sizeof(SimScalar) * n);
			cudaMemset(d_x[i], 0, sizeof(SimScalar) * n);
			cudaMemset(d_next_x[i], 0, sizeof(SimScalar) * n);
		}

		const int n_vertex = n / 3;
		const int n_blocks = util::get_n_blocks(n_vertex);

		SimScalar omega = 1;
		const auto get_omega = [&](int cur_n_itr) -> SimScalar
		{
			constexpr auto chebyshev_chmod_threshold = 11;

			if (cur_n_itr < chebyshev_chmod_threshold)
			{
				return 1;
			}
			else if (cur_n_itr == chebyshev_chmod_threshold)
			{
				return (SimScalar)2 / (2 - rho * rho);
			}
			return (SimScalar)4 / (4 - rho * rho * omega);
		};

		if (order == 1)
		{
			// precompute b term 1
			precompute_b_term_1<<<n_blocks, WARP_SIZE>>>(d_b_term, d_b, d_diagonals, n_vertex);

			for (int i = 0; i < n_itr; i++)
			{
				omega = get_omega(n_itr);

				// perform triple buffer to avoid swapping overhead
				if (i % 3 == 0)
				{
					itr_order_1 << <n_blocks, WARP_SIZE >> > (
						d_next_x[0],
						d_x[0],
						d_prev_x[0],
						d_k_ring_neighbors[0],
						d_k_ring_neighbor_indices[0],
						d_k_ring_neighbor_sizes[0],
						d_diagonals,
						d_b_term,
						n_vertex,
						omega,
						under_relaxation
						);
				}
				else if (i % 3 == 1)
				{
					itr_order_1 << <n_blocks, WARP_SIZE >> > (
						d_prev_x[0],
						d_next_x[0],
						d_x[0],
						d_k_ring_neighbors[0],
						d_k_ring_neighbor_indices[0],
						d_k_ring_neighbor_sizes[0],
						d_diagonals,
						d_b_term,
						n_vertex,
						omega,
						under_relaxation
						);
				}
				else // i % 3 == 2
				{
					itr_order_1 << <n_blocks, WARP_SIZE >> > (
						d_x[0],
						d_prev_x[0],
						d_next_x[0],
						d_k_ring_neighbors[0],
						d_k_ring_neighbor_indices[0],
						d_k_ring_neighbor_sizes[0],
						d_diagonals,
						d_b_term,
						n_vertex,
						omega,
						under_relaxation
						);
				}
			}
		}
		else if (order == 2)
		{
			// precompute b term 2
			precompute_b_term_2<<<n_blocks, WARP_SIZE>>>(
				d_b_term,
				d_b,
				d_k_ring_neighbors[0],
				d_k_ring_neighbor_indices[0],
				d_k_ring_neighbor_sizes[0],
				d_diagonals,
				n_vertex
			);

			for (int i = 0; i < n_itr; i++)
			{
				omega = get_omega(n_itr);

				// perform triple buffer to avoid swapping overhead
				if (i % 3 == 0)
				{
					itr_order_2 << <n_blocks, WARP_SIZE >> > (
						d_next_x[0],
						d_next_x[1],
						d_x[0],
						d_x[1],
						d_prev_x[0],
						d_prev_x[1],
						d_k_ring_neighbors[1],
						d_k_ring_neighbor_indices[1],
						d_k_ring_neighbor_sizes[1],
						d_diagonals,
						d_b_term,
						n_vertex,
						omega,
						under_relaxation
						);
				}
				else if (i % 3 == 1)
				{
					itr_order_2 << <n_blocks, WARP_SIZE >> > (
						d_prev_x[0],
						d_prev_x[1],
						d_next_x[0],
						d_next_x[1],
						d_x[0],
						d_x[1],
						d_k_ring_neighbors[1],
						d_k_ring_neighbor_indices[1],
						d_k_ring_neighbor_sizes[1],
						d_diagonals,
						d_b_term,
						n_vertex,
						omega,
						under_relaxation
						);
				}
				else // i % 3 == 2
				{
					itr_order_2 << <n_blocks, WARP_SIZE >> > (
						d_x[0],
						d_x[1],
						d_prev_x[0],
						d_prev_x[1],
						d_next_x[0],
						d_next_x[1],
						d_k_ring_neighbors[1],
						d_k_ring_neighbor_indices[1],
						d_k_ring_neighbor_sizes[1],
						d_diagonals,
						d_b_term,
						n_vertex,
						omega,
						under_relaxation
						);
				}
			}
		}
		else if (order == 3)
		{
			// precompute b term 3
			precompute_b_term_3<<<n_blocks, WARP_SIZE>>>(
				d_b_term,
				d_b,
				d_k_ring_neighbors[0],
				d_k_ring_neighbor_indices[0],
				d_k_ring_neighbor_sizes[0],
				d_k_ring_neighbors[1],
				d_k_ring_neighbor_indices[1],
				d_k_ring_neighbor_sizes[1],
				d_diagonals,
				n_vertex
			);

			for (int i = 0; i < n_itr; i++)
			{
				omega = get_omega(n_itr);

				// perform triple buffer to avoid swapping overhead
				if (i % 3 == 0)
				{
					itr_order_3 << <n_blocks, WARP_SIZE >> > (
						d_next_x[0],
						d_next_x[1],
						d_next_x[2],
						d_x[0],
						d_x[1],
						d_x[2],
						d_prev_x[0],
						d_prev_x[1],
						d_prev_x[2],
						d_k_ring_neighbors[2],
						d_k_ring_neighbor_indices[2],
						d_k_ring_neighbor_sizes[2],
						d_diagonals,
						d_b_term,
						n_vertex,
						omega,
						under_relaxation
						);
				}
				else if (i % 3 == 1)
				{
					itr_order_3 << <n_blocks, WARP_SIZE >> > (
						d_prev_x[0],
						d_prev_x[1],
						d_prev_x[2],
						d_next_x[0],
						d_next_x[1],
						d_next_x[2],
						d_x[0],
						d_x[1],
						d_x[2],
						d_k_ring_neighbors[2],
						d_k_ring_neighbor_indices[2],
						d_k_ring_neighbor_sizes[2],
						d_diagonals,
						d_b_term,
						n_vertex,
						omega,
						under_relaxation
						);
				}
				else // i % 3 == 2
				{
					itr_order_3 << <n_blocks, WARP_SIZE >> > (
						d_x[0],
						d_x[1],
						d_x[2],
						d_prev_x[0],
						d_prev_x[1],
						d_prev_x[2],
						d_next_x[0],
						d_next_x[1],
						d_next_x[2],
						d_k_ring_neighbors[2],
						d_k_ring_neighbor_indices[2],
						d_k_ring_neighbor_sizes[2],
						d_diagonals,
						d_b_term,
						n_vertex,
						omega,
						under_relaxation
						);
				}
			}
		}

		checkCudaErrors(cudaMemcpy(ret.data(), d_x[order - 1], sizeof(SimScalar) * n, cudaMemcpyDeviceToHost));
		// check if the error is OK
		SimVectorX err_checker;
		err_checker.resizeLike(ret);
		checkCudaErrors(cudaMemcpy(err_checker.data(), d_next_x[order - 1], sizeof(SimScalar) * n, cudaMemcpyDeviceToHost));
		constexpr SimScalar eps = 1e-3;
		for (int i = 0; i < n; i++)
		{
			if (std::abs(err_checker[i] - ret[i]) > eps)
			{
				printf("Warning: A-Jacobi iteration incomplete: At index %d, values are %f, %f.\n", i, err_checker[i], ret[i]);
				break;
			}
			if (std::abs(err_checker[i]) > 1e10)
			{
				printf("Warning: A-Jacobi divergence issue occured: At index %d, value is %f.\n", i, err_checker[i]);
			}
		}
		if (err_checker.hasNaN())
		{
			printf("Warning: A-Jacobi divergence issue occured: NaN detected!\n");
		}
		// if (true)
			// std::cout << "err checker[32] = " << err_checker[32] << "\n" << "ret[32] = " << ret[32] << "\n";

		return ret;
	}
}