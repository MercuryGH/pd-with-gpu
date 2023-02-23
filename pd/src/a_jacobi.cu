#include <pd/a_jacobi.h>
#include <pd/types.h>
#include <util/helper_cuda.h>
#include <util/adj_list_graph.h>
#include <array>

namespace pd
{
	void AJacobi::clear()
	{
		if (is_allocated)
		{
			// free memory
			B.clear();
			D.clear();
			checkCudaErrors(cudaFree(d_b));
			for (int k = 0; k < order; k++)
			{
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

	void AJacobi::set_A(const Eigen::SparseMatrix<float>& A, const pd::Constraints& constraints)
	{
		n = A.rows();
		assert((n / 3) * 3 == n);
		if (order > A_JACOBI_MAX_ORDER || order < 1)
		{
			printf("Error: Invalid A-Jacobi order!\n");
			assert(false);
		}
		// set precomputation values

		checkCudaErrors(cudaMalloc((void**)&d_b, sizeof(float) * n));
		for (int i = 0; i < order; i++)
		{
			checkCudaErrors(cudaMalloc((void**)&d_x[i], sizeof(float) * n));
			checkCudaErrors(cudaMalloc((void**)&d_next_x[i], sizeof(float) * n));
		}
		precompute_A_jacobi(A, constraints);

		is_allocated = true;
	}

	void AJacobi::precompute_A_jacobi(const Eigen::SparseMatrix<float>& A, const pd::Constraints& constraints)
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
		for (const auto& constraint : constraints)
		{
			int* vertices = nullptr;
			const int n_vertices = constraint->get_involved_vertices(&vertices);
			if (n_vertices == 2) // edge length constraint
			{
				int vi = vertices[0];
				int vj = vertices[1];

				edges.emplace_back(vi, vj);
			}
		}
		util::AdjListGraph adj_list_graph(edges, n_vertex);

		const std::vector<std::unordered_set<int>>& adj_list = adj_list_graph.get_adj_list();
		D.resize(n_vertex);
		B.resize(n_vertex);

		// B and D precomputation
		for (int i = 0; i < n_vertex; i++)
		{
			for (int j : adj_list[i])
			{
				// Note: negative sign!!!
				B[i][j] = -get_B_ij(i, j);
			}
			D[i] = 1.0f / get_D_ii(i);
			if (std::abs(D[i]) < 1e-5f)
			{
				printf("Warning: i = %d, D[i] = %f\n", i, D[i]);
			}
		}

		// --- Compute d_diagonals
		checkCudaErrors(cudaMalloc((void**)&d_diagonals, sizeof(float) * n_vertex));
		checkCudaErrors(cudaMemcpy(d_diagonals, D.data(), sizeof(float) * n_vertex, cudaMemcpyHostToDevice));

		const auto get_vertex_k_ring_neighbors = [&](
			int i,
			int ring_width,
			std::vector<int>& neighbor_indices,
			std::vector<float>& neighbor_B_vals
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
					std::unordered_map<int, float> a_products_on_vertex;
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
					std::unordered_map<int, float> a_products_on_vertex;
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
			checkCudaErrors(cudaMalloc((void***)&d_k_ring_neighbors[k], sizeof(float*) * n_vertex));
			checkCudaErrors(cudaMalloc((void***)&d_k_ring_neighbor_indices[k], sizeof(int*) * n_vertex));
			checkCudaErrors(cudaMalloc((void**)&d_k_ring_neighbor_sizes[k], sizeof(int) * n_vertex));
			// to avoid indexing device memory on the host, use tmp host memory
			k_ring_neighbors[k] = (float**)malloc(sizeof(float*) * n_vertex);
			k_ring_neighbor_indices[k] = (int**)malloc(sizeof(int*) * n_vertex);
			k_ring_neighbor_sizes[k].resize(n_vertex);

			for (int i = 0; i < n_vertex; i++)
			{
				std::vector<int> neighbor_indices;
				std::vector<float> neighbor_B_vals;
				get_vertex_k_ring_neighbors(i, k + 1, neighbor_indices, neighbor_B_vals);
				const int n_adj_vertex = neighbor_indices.size();
				checkCudaErrors(
					cudaMalloc((void**)&k_ring_neighbors[k][i], sizeof(float) * n_adj_vertex));
				checkCudaErrors(
					cudaMalloc((void**)&k_ring_neighbor_indices[k][i], sizeof(int) * n_adj_vertex));
				k_ring_neighbor_sizes[k][i] = n_adj_vertex;

				checkCudaErrors(
					cudaMemcpy(k_ring_neighbors[k][i], neighbor_B_vals.data(), sizeof(float) * n_adj_vertex, cudaMemcpyHostToDevice));
				checkCudaErrors(
					cudaMemcpy(k_ring_neighbor_indices[k][i], neighbor_indices.data(), sizeof(int) * n_adj_vertex, cudaMemcpyHostToDevice));
			}

			// copy memory
			checkCudaErrors(
				cudaMemcpy(d_k_ring_neighbors[k], k_ring_neighbors[k], sizeof(float*) * n_vertex, cudaMemcpyHostToDevice));
			checkCudaErrors(
				cudaMemcpy(d_k_ring_neighbor_indices[k], k_ring_neighbor_indices[k], sizeof(int*) * n_vertex, cudaMemcpyHostToDevice));
			checkCudaErrors(
				cudaMemcpy(d_k_ring_neighbor_sizes[k], k_ring_neighbor_sizes[k].data(), sizeof(int) * n_vertex, cudaMemcpyHostToDevice));
		}
	}

	Eigen::VectorXf AJacobi::solve(const Eigen::VectorXf& b)
	{
		Eigen::VectorXf ret;
		ret.resizeLike(b);

		checkCudaErrors(cudaMemcpy(d_b, b.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
		for (int i = 0; i < order; i++)
		{
			cudaMemset(d_x[i], 0, sizeof(float) * n);
			cudaMemset(d_next_x[i], 0, sizeof(float) * n);
		}

		const int n_vertex = n / 3;
		if (order == 1)
		{
			const int n_blocks = n_vertex / WARP_SIZE + (n_vertex % WARP_SIZE == 0 ? 0 : 1);
			for (int i = 0; i < n_itr; i++)
			{
				if (i % 2 == 1)
				{
					itr_order_1 << <n_blocks, WARP_SIZE >> > (
						d_x[0],
						d_next_x[0],
						d_k_ring_neighbors[0],
						d_k_ring_neighbor_indices[0],
						d_k_ring_neighbor_sizes[0],
						d_diagonals,
						d_b,
						n
						);
				}
				else
				{
					itr_order_1 << <n_blocks, WARP_SIZE >> > (
						d_next_x[0],
						d_x[0],
						d_k_ring_neighbors[0],
						d_k_ring_neighbor_indices[0],
						d_k_ring_neighbor_sizes[0],
						d_diagonals,
						d_b,
						n
						);
				}
			}
		}
		else if (order == 2)
		{
			const int n_blocks = n_vertex / WARP_SIZE + (n_vertex % WARP_SIZE == 0 ? 0 : 1);
			for (int i = 0; i < n_itr; i++)
			{
				if (i % 2 == 1)
				{
					itr_order_2 << <n_blocks, WARP_SIZE >> > (
						d_x[0],
						d_x[1],
						d_next_x[0],
						d_next_x[1],
						d_k_ring_neighbors[0],
						d_k_ring_neighbor_indices[0],
						d_k_ring_neighbor_sizes[0],
						d_k_ring_neighbors[1],
						d_k_ring_neighbor_indices[1],
						d_k_ring_neighbor_sizes[1],
						d_diagonals,
						d_b,
						n
						);
				}
				else
				{
					itr_order_2 << <n_blocks, WARP_SIZE >> > (
						d_next_x[0],
						d_next_x[1],
						d_x[0],
						d_x[1],
						d_k_ring_neighbors[0],
						d_k_ring_neighbor_indices[0],
						d_k_ring_neighbor_sizes[0],
						d_k_ring_neighbors[1],
						d_k_ring_neighbor_indices[1],
						d_k_ring_neighbor_sizes[1],
						d_diagonals,
						d_b,
						n
						);
				}
			}
		}
		else if (order == 3)
		{
			const int n_blocks = n_vertex / WARP_SIZE + (n_vertex % WARP_SIZE == 0 ? 0 : 1);
			for (int i = 0; i < n_itr; i++)
			{
				if (i % 2 == 1)
				{
					itr_order_3 << <n_blocks, WARP_SIZE >> > (
						d_x[0],
						d_x[1],
						d_x[2],
						d_next_x[0],
						d_next_x[1],
						d_next_x[2],
						d_k_ring_neighbors[0],
						d_k_ring_neighbor_indices[0],
						d_k_ring_neighbor_sizes[0],
						d_k_ring_neighbors[1],
						d_k_ring_neighbor_indices[1],
						d_k_ring_neighbor_sizes[1],
						d_k_ring_neighbors[2],
						d_k_ring_neighbor_indices[2],
						d_k_ring_neighbor_sizes[2],
						d_diagonals,
						d_b,
						n
						);
				}
				else
				{
					itr_order_3 << <n_blocks, WARP_SIZE >> > (
						d_next_x[0],
						d_next_x[1],
						d_next_x[2],
						d_x[0],
						d_x[1],
						d_x[2],
						d_k_ring_neighbors[0],
						d_k_ring_neighbor_indices[0],
						d_k_ring_neighbor_sizes[0],
						d_k_ring_neighbors[1],
						d_k_ring_neighbor_indices[1],
						d_k_ring_neighbor_sizes[1],
						d_k_ring_neighbors[2],
						d_k_ring_neighbor_indices[2],
						d_k_ring_neighbor_sizes[2],
						d_diagonals,
						d_b,
						n
						);
				}
			}
		}

		checkCudaErrors(cudaMemcpy(ret.data(), d_x[order - 1], sizeof(float) * n, cudaMemcpyDeviceToHost));
		// check if the error is OK
		Eigen::VectorXf err_checker;
		err_checker.resizeLike(ret);
		checkCudaErrors(cudaMemcpy(err_checker.data(), d_next_x[order - 1], sizeof(float) * n, cudaMemcpyDeviceToHost));
		constexpr float eps = 1e-3f;
		for (int i = 0; i < n; i++)
		{
			if (std::abs(err_checker[i] - ret[i]) > eps)
			{
				printf("Warning: A-Jacobi Iteration Incomplete. At index %d, values are %f, %f.\n", i, err_checker[i], ret[i]);
				break;
			}
		}
		if (true)
			std::cout << "err checker[32] = " << err_checker[32] << "\n" << "ret[32] = " << ret[32] << "\n";

		return ret;
	}

	__global__ void itr_order_1(
		float* __restrict__ next_x,
		const float* __restrict__ x,
		float** __restrict__ d_1_ring_neighbors,
		int** __restrict__ d_1_ring_neighbor_indices,
		const int* __restrict__ d_1_ring_neighbor_sizes,
		const float* __restrict__ d_diagonals,
		const float* __restrict__ b,
		int n  // #Vertex, parallelism is n but not 3n
	)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < n)
		{
			const float* B_ijs = d_1_ring_neighbors[idx];
			const int* js = d_1_ring_neighbor_indices[idx];

			float sum_0 = 0.0f;
			float sum_1 = 0.0f;
			float sum_2 = 0.0f;

			int j_cnt = d_1_ring_neighbor_sizes[idx];
			for (int k = 0; k < j_cnt; k++)
			{
				float B_ij = B_ijs[k];
				int j = js[k];

				sum_0 += B_ij * x[3 * j];
				sum_1 += B_ij * x[3 * j + 1];
				sum_2 += B_ij * x[3 * j + 2];
			}

			float D_ii_inv = d_diagonals[idx];

			if (idx <= 20)
			{
				printf("In GPU: idx = %d, %f %f %f\n", idx, sum_2, D_ii_inv, b[3 * idx + 2]);
				// TODO: Fix the bug that D_ii_inv suddenly becomes zero 
				// (maybe a memory access UB in previous GPU code)
			}

			next_x[3 * idx] = (sum_0 + b[3 * idx]) * D_ii_inv;
			next_x[3 * idx + 1] = (sum_1 + b[3 * idx + 1]) * D_ii_inv;
			next_x[3 * idx + 2] = (sum_2 + b[3 * idx + 2]) * D_ii_inv;
		}
	}

	__global__ void itr_order_2(
		float* __restrict__ next_x_1,
		float* __restrict__ next_x_2,
		const float* __restrict__ x_1,
		const float* __restrict__ x_2,

		float** __restrict__ d_1_ring_neighbors,
		int** __restrict__ d_1_ring_neighbor_indices,
		const int* __restrict__ d_1_ring_neighbor_sizes,

		float** __restrict__ d_2_ring_neighbors,
		int** __restrict__ d_2_ring_neighbor_indices,
		const int* __restrict__ d_2_ring_neighbor_sizes,

		const float* __restrict__ d_diagonals, // D_ii

		const float* __restrict__ b,
		int n  // #Vertex
	)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < n)
		{
			const float* B_issjs = d_2_ring_neighbors[idx];
			const int* js = d_2_ring_neighbor_indices[idx];

			float sum_0_1 = 0.0f;
			float sum_1_1 = 0.0f;
			float sum_2_1 = 0.0f;

			float sum_0_2 = 0.0f;
			float sum_1_2 = 0.0f;
			float sum_2_2 = 0.0f;

			int j_cnt = d_2_ring_neighbor_sizes[idx];
			for (int k = 0; k < j_cnt; k++)
			{
				float B_issj = B_issjs[k];
				int j = js[k];

				sum_0_1 += B_issj * x_1[3 * j];
				sum_1_1 += B_issj * x_1[3 * j + 1];
				sum_2_1 += B_issj * x_1[3 * j + 2];

				sum_0_2 += B_issj * x_2[3 * j];
				sum_1_2 += B_issj * x_2[3 * j + 1];
				sum_2_2 += B_issj * x_2[3 * j + 2];
			}

			// b_terms:
			const float* B_ijs_b = d_1_ring_neighbors[idx];
			const int* js_b = d_1_ring_neighbor_indices[idx];

			float b_term_0 = 0.0f;
			float b_term_1 = 0.0f;
			float b_term_2 = 0.0f;

			int j_cnt_b = d_1_ring_neighbor_sizes[idx];
			for (int k = 0; k < j_cnt_b; k++)
			{
				float B_ij = B_ijs_b[k];
				int j = js_b[k];

				b_term_0 += B_ij * b[3 * j];
				b_term_1 += B_ij * b[3 * j + 1];
				b_term_2 += B_ij * b[3 * j + 2];
			}

			float D_ii_inv = d_diagonals[idx];

			// may contains floating point precision problem
			b_term_0 = (b[3 * idx] + b_term_0) * D_ii_inv;
			b_term_1 = (b[3 * idx + 1] + b_term_1) * D_ii_inv;
			b_term_2 = (b[3 * idx + 2] + b_term_2) * D_ii_inv;

			next_x_1[3 * idx] = sum_0_1 * D_ii_inv + b_term_0;
			next_x_1[3 * idx + 1] = sum_1_1 * D_ii_inv + b_term_1;
			next_x_1[3 * idx + 2] = sum_2_1 * D_ii_inv + b_term_2;
			//if (idx == 1)
				//printf("%f %f %f\n", next_x_1[3 * idx], next_x_1[3 * idx + 1], next_x_1[3 * idx + 2]);

			next_x_2[3 * idx] = sum_0_2 * D_ii_inv + b_term_0;
			next_x_2[3 * idx + 1] = sum_1_2 * D_ii_inv + b_term_1;
			next_x_2[3 * idx + 2] = sum_2_2 * D_ii_inv + b_term_2;
		}
	}

	__global__ void itr_order_3(
		float* __restrict__ next_x_1,
		float* __restrict__ next_x_2,
		float* __restrict__ next_x_3,
		const float* __restrict__ x_1,
		const float* __restrict__ x_2,
		const float* __restrict__ x_3,

		float** __restrict__ d_1_ring_neighbors,
		int** __restrict__ d_1_ring_neighbor_indices,
		const int* __restrict__ d_1_ring_neighbor_sizes,

		float** __restrict__ d_2_ring_neighbors,
		int** __restrict__ d_2_ring_neighbor_indices,
		const int* __restrict__ d_2_ring_neighbor_sizes,

		float** __restrict__ d_3_ring_neighbors,
		int** __restrict__ d_3_ring_neighbor_indices,
		const int* __restrict__ d_3_ring_neighbor_sizes,

		const float* __restrict__ d_diagonals, // D_ii
		const float* __restrict__ b,
		int n  // #Vertex
	)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < n)
		{
			const float* B_ittssjs = d_3_ring_neighbors[idx];
			const int* js = d_3_ring_neighbor_indices[idx];

			float sum_0_1 = 0.0f;
			float sum_1_1 = 0.0f;
			float sum_2_1 = 0.0f;

			float sum_0_2 = 0.0f;
			float sum_1_2 = 0.0f;
			float sum_2_2 = 0.0f;

			float sum_0_3 = 0.0f;
			float sum_1_3 = 0.0f;
			float sum_2_3 = 0.0f;

			int j_cnt = d_3_ring_neighbor_sizes[idx];
			for (int k = 0; k < j_cnt; k++)
			{
				float B_ittssj = B_ittssjs[k];
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

			// b_terms_1:
			const float* B_ijs_b = d_1_ring_neighbors[idx];
			const int* js_b_1 = d_1_ring_neighbor_indices[idx];

			float b_term_0_1 = 0.0f;
			float b_term_1_1 = 0.0f;
			float b_term_2_1 = 0.0f;

			int j_cnt_b_1 = d_1_ring_neighbor_sizes[idx];
			for (int k = 0; k < j_cnt_b_1; k++)
			{
				float B_ij = B_ijs_b[k];
				int j = js_b_1[k];

				b_term_0_1 += B_ij * b[3 * j];
				b_term_1_1 += B_ij * b[3 * j + 1];
				b_term_2_1 += B_ij * b[3 * j + 2];
			}

			// b_terms_2:
			const float* B_issjs_b = d_2_ring_neighbors[idx];
			const int* js_b_2 = d_2_ring_neighbor_indices[idx];

			float b_term_0_2 = 0.0f;
			float b_term_1_2 = 0.0f;
			float b_term_2_2 = 0.0f;

			int j_cnt_b_2 = d_2_ring_neighbor_sizes[idx];
			for (int k = 0; k < j_cnt_b_2; k++)
			{
				float B_issj = B_issjs_b[k];
				int j = js_b_2[k];

				b_term_0_2 += B_issj * b[3 * j];
				b_term_1_2 += B_issj * b[3 * j + 1];
				b_term_2_2 += B_issj * b[3 * j + 2];
			}

			float D_ii_inv = d_diagonals[idx];

			// may contains floating point precision problem
			b_term_0_1 = (b[3 * idx] + b_term_0_1 + b_term_0_2) * D_ii_inv;
			b_term_1_1 = (b[3 * idx + 1] + b_term_1_1 + b_term_1_2) * D_ii_inv;
			b_term_2_1 = (b[3 * idx + 2] + b_term_2_1 + b_term_2_2) * D_ii_inv;

			next_x_1[3 * idx] = sum_0_1 * D_ii_inv + b_term_0_1;
			next_x_1[3 * idx + 1] = sum_1_1 * D_ii_inv + b_term_1_1;
			next_x_1[3 * idx + 2] = sum_2_1 * D_ii_inv + b_term_2_1;
			//if (idx == 1)
				//printf("%f %f %f\n", next_x_1[3 * idx], next_x_1[3 * idx + 1], next_x_1[3 * idx + 2]);

			next_x_2[3 * idx] = sum_0_2 * D_ii_inv + b_term_0_1;
			next_x_2[3 * idx + 1] = sum_1_2 * D_ii_inv + b_term_1_1;
			next_x_2[3 * idx + 2] = sum_2_2 * D_ii_inv + b_term_2_1;

			next_x_3[3 * idx] = sum_0_3 * D_ii_inv + b_term_0_1;
			next_x_3[3 * idx + 1] = sum_1_3 * D_ii_inv + b_term_1_1;
			next_x_3[3 * idx + 2] = sum_2_3 * D_ii_inv + b_term_2_1;
		}
	}
}