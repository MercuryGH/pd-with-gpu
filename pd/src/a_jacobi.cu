#include <pd/a_jacobi.h>
#include <pd/types.h>
#include <util/helper_cuda.h>
#include <util/adj_list_graph.h>

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
			for (int i = 0; i < order; i++)
			{
				checkCudaErrors(cudaFree(d_x[i]));
				checkCudaErrors(cudaFree(d_next_x[i]));
			}
			checkCudaErrors(cudaFree(d_diagonals));
			const int n_vertex = n / 3;
			for (int i = 0; i < n_vertex; i++)
			{
				checkCudaErrors(cudaFree(b_a_products[i]));
				checkCudaErrors(cudaFree(b_a_products_idx[i]));
			}
			checkCudaErrors(cudaFree(d_a_products));
			checkCudaErrors(cudaFree(d_a_products_idx));
			checkCudaErrors(cudaFree(d_a_product_sizes));
			free(b_a_products);
			free(b_a_products_idx);

			is_allocated = false;
		}
	}

	void AJacobi::set_A(const Eigen::SparseMatrix<float>& A, const pd::Constraints& constraints)
	{
		n = A.rows();
		assert((n / 3) * 3 == n);
		// set precomputation values

		checkCudaErrors(cudaMalloc((void**)&d_b, sizeof(float) * n));
		d_x.resize(order);
		d_next_x.resize(order);
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
			const std::vector<int>& vertices = constraint->get_involved_vertices();
			if (vertices.size() == 2) // edge length constraint
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
		}

		//// check if B is ok
		//for (int i = 0; i < n_vertex; i++)
		//{
		//	for (int j = 0; j < n_vertex; j++)
		//	{
		//		float a_v = A.coeff(3 * i, 3 * j);
		//		if (std::abs(a_v) > 0.1f)
		//		{
		//			if (a_v != B[i][j])
		//			{
		//				printf("%d, %d\n", i, j);
		//			}
		//		}
		//	}
		//}

		// --- Compute d_diagonals

		// compute D_{ii} for l = 1
		checkCudaErrors(cudaMalloc((void**)&d_diagonals, sizeof(float) * n_vertex));
		checkCudaErrors(cudaMemcpy(d_diagonals, D.data(), sizeof(float) * n_vertex, cudaMemcpyHostToDevice));

		// l = 2


		// --- Compute d_a_products, d_a_products_idx and d_a_products_sizes
		checkCudaErrors(cudaMalloc((void***)&d_a_products, sizeof(float*) * n_vertex));
		checkCudaErrors(cudaMalloc((void***)&d_a_products_idx, sizeof(int*) * n_vertex));
		checkCudaErrors(cudaMalloc((void**)&d_a_product_sizes, sizeof(int) * n_vertex));
		// to avoid indexing device memory on the host, use tmp host memory
		b_a_products = (float**)malloc(sizeof(float*) * n_vertex);
		b_a_products_idx = (int**)malloc(sizeof(int*) * n_vertex);
		std::vector<int> a_product_sizes(n_vertex);
		
		// compute B_{ij} for l = 1
		for (int i = 0; i < n_vertex; i++)
		{
			const int n_adj_vertex = B[i].size();
			a_product_sizes[i] = n_adj_vertex;
			checkCudaErrors(
				cudaMalloc((void**)&b_a_products[i], sizeof(float) * n_adj_vertex));
			checkCudaErrors(
				cudaMalloc((void**)&b_a_products_idx[i], sizeof(int) * n_adj_vertex));

			std::vector<int> keys(n_adj_vertex);
			std::vector<float> values(n_adj_vertex);
			int cnt = 0;
			for (const auto& [k, v] : B[i])
			{
				keys[cnt] = k;
				values[cnt] = v;
				cnt++;
			}
			checkCudaErrors(
				cudaMemcpy(b_a_products[i], values.data(), sizeof(float) * n_adj_vertex, cudaMemcpyHostToDevice));
			checkCudaErrors(
				cudaMemcpy(b_a_products_idx[i], keys.data(), sizeof(int) * n_adj_vertex, cudaMemcpyHostToDevice));
		}
		// copy memory
		checkCudaErrors(
			cudaMemcpy(d_a_products, b_a_products, sizeof(float*) * n_vertex, cudaMemcpyHostToDevice));
		checkCudaErrors(
			cudaMemcpy(d_a_products_idx, b_a_products_idx, sizeof(int*) * n_vertex, cudaMemcpyHostToDevice));
		checkCudaErrors(
			cudaMemcpy(d_a_product_sizes, a_product_sizes.data(), sizeof(int) * n_vertex, cudaMemcpyHostToDevice));

		// compute B_{is}B_{sj} for l = 2

		// compute B_{is}B_{st}B_{tj} for l = 3

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
						d_a_products,
						d_a_products_idx,
						d_a_product_sizes,
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
						d_a_products,
						d_a_products_idx,
						d_a_product_sizes,
						d_diagonals,
						d_b,
						n
					);
				}
			}
		}
		else if (order == 2)
		{
		}
		else if (order == 3)
		{
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
		//if (true)
			//std::cout << "err checker[33] = " << err_checker[33] << "\n" << "ret[33] = " << ret[33] << "\n";

		return ret;
	}

	__global__  void itr_order_1(
		float* next_x,
		const float* __restrict__ x,
		float** __restrict__ d_a_products,
		int** __restrict__ d_a_products_idx,
		const int* __restrict__ d_a_product_sizes,
		const float* __restrict__ d_diagonals,
		const float* __restrict__ b,
		int n  // #Vertex, parallelism is n but not 3n
	)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < n)
		{
			float D_ii_inv = d_diagonals[idx];
			const float* B_ijs = d_a_products[idx];
			const int* js = d_a_products_idx[idx];

			float sum_0 = 0.0f;
			float sum_1 = 0.0f;
			float sum_2 = 0.0f;

			int j_cnt = d_a_product_sizes[idx];
			for (int k = 0; k < j_cnt; k++)
			{
				float neg_B_ij = B_ijs[k];
				int j = js[k];

				sum_0 += neg_B_ij * x[3 * j];
				sum_1 += neg_B_ij * x[3 * j + 1];
				sum_2 += neg_B_ij * x[3 * j + 2];
			}

			next_x[3 * idx] = D_ii_inv * sum_0 + D_ii_inv * b[3 * idx];
			next_x[3 * idx + 1] = D_ii_inv * sum_1 + D_ii_inv * b[3 * idx + 1];
			next_x[3 * idx + 2] = D_ii_inv * sum_2 + D_ii_inv * b[3 * idx + 2];
		}
	}

	__global__ void itr_order_2(
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
	)
	{
	}
}