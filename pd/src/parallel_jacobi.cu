#include <pd/parallel_jacobi.h>
#include <util/cpu_timer.h>
#include <util/gpu_helper.h>
#include <iostream>

namespace pd
{
	__global__ void itr_shfl_down(SimScalar* __restrict__ next_x, const SimScalar* __restrict__ A, const SimScalar* __restrict__ x, const SimScalar* __restrict__ b, int n_row, int n_col)
	{
		int col_start = threadIdx.x; // indicates i-th thread in a warp, 0 <= i <= 31
		int row = blockIdx.x;
		int offset = row * n_col;
		SimScalar sum = 0.0f;

		if (row < n_row)
		{
			for (int i = col_start; i < n_col; i += blockDim.x) // blockDim.x == 32 == WARP_SIZE
			{
				sum += A[offset + i] * x[i];
			}

			sum += __shfl_down_sync(__activemask(), sum, 16);
			sum += __shfl_down_sync(__activemask(), sum, 8);
			sum += __shfl_down_sync(__activemask(), sum, 4);
			sum += __shfl_down_sync(__activemask(), sum, 2);
			sum += __shfl_down_sync(__activemask(), sum, 1);
			if (threadIdx.x == 0)
			{
				// let the first thread of a warp write next_x
				next_x[row] = (b[row] - (sum - A[offset + row] * x[row])) / A[offset + row];
			}
		}
	}

	__global__ void itr_normal(SimScalar* __restrict__ next_x, const SimScalar* __restrict__ A, const SimScalar* __restrict__ x, const SimScalar* __restrict__ b, int n_row, int n_col)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < n_row)
		{
			SimScalar sum = 0.0f;
			int row_offset = idx * n_col;
			for (int j = 0; j < n_col; j++)
			{
				sum += A[row_offset + j] * x[j];
			}
			sum -= A[row_offset + idx] * x[idx];
			next_x[idx] = (b[idx] - sum) / A[row_offset + idx];
		}
	}

	void ParallelJacobi::clear()
	{
		if (is_allocated)
		{
			checkCudaErrors(cudaFree(d_A));
			checkCudaErrors(cudaFree(d_b));
			checkCudaErrors(cudaFree(d_x));
			checkCudaErrors(cudaFree(d_next_x));
			is_allocated = false;
		}
	}

	// Make sure A is compressed
	void ParallelJacobi::set_A(const Eigen::SparseMatrix<SimScalar>& A, const std::unordered_map<MeshIDType, DeformableMesh>& models)
	{
		SimMatrixX _A = SimMatrixX(A);
		n = _A.rows(); // n = 3 * #Vertex

		checkCudaErrors(cudaMalloc((void**)&d_A, sizeof(SimScalar) * _A.size()));
		checkCudaErrors(cudaMalloc((void**)&d_b, sizeof(SimScalar) * n));
		checkCudaErrors(cudaMalloc((void**)&d_x, sizeof(SimScalar) * n));
		checkCudaErrors(cudaMalloc((void**)&d_next_x, sizeof(SimScalar) * n));
		is_allocated = true;

		checkCudaErrors(cudaMemcpy(d_A, _A.data(), sizeof(SimScalar) * _A.size(), cudaMemcpyHostToDevice));
	}

	SimVectorX ParallelJacobi::solve(const SimVectorX& b)
	{
		SimVectorX ret;
		ret.resizeLike(b);

		assert(b.size() == n);

		checkCudaErrors(cudaMemcpy(d_b, b.data(), sizeof(SimScalar) * n, cudaMemcpyHostToDevice));
		// set to IEEE-754 zero as iteration initial value
		cudaMemset(d_x, 0, sizeof(SimScalar) * n);
		cudaMemset(d_next_x, 0, sizeof(SimScalar) * n);

		// The solver iterates for a constant number, not checking error
		//SimScalar diff = 0.0f;
		//SimScalar eps = 1e-4f;
		if (false)
		{
			const int n_blocks = util::get_n_blocks(n);
			for (int i = 0; i < n_itr; i++)
			{
				// double buffer
				if (i % 2 == 1)
				{
					itr_normal << <n_blocks, WARP_SIZE >> > (d_x, d_A, d_next_x, d_b, n, n);
				}
				else
				{
					itr_normal << <n_blocks, WARP_SIZE >> > (d_next_x, d_A, d_x, d_b, n, n);
				}
				//cudaDeviceSynchronize(); no need to call since kernel execution in GPU is sequential
			}
		}
		else
		{
			for (int i = 0; i < n_itr; i++)
			{
				// double buffer
				if (i % 2 == 1)
				{
					itr_shfl_down << <n, WARP_SIZE >> > (d_x, d_A, d_next_x, d_b, n, n);
				}
				else
				{
					itr_shfl_down << <n, WARP_SIZE >> > (d_next_x, d_A, d_x, d_b, n, n);
				}
			}
		}
		//cudaDeviceSynchronize();  // no need to call since cudaMemcpy is synchronized

		checkCudaErrors(cudaMemcpy(ret.data(), d_x, sizeof(SimScalar) * n, cudaMemcpyDeviceToHost));

		// check if the error is OK
		SimVectorX err_checker;
		err_checker.resizeLike(ret);
		checkCudaErrors(cudaMemcpy(err_checker.data(), d_next_x, sizeof(SimScalar) * n, cudaMemcpyDeviceToHost));
		constexpr SimScalar eps = 1e-3f;
		for (int i = 0; i < n; i++)
		{
			if (std::abs(err_checker[i] - ret[i]) > eps)
			{
				printf("Warning: Jacobi Iteration Incomplete. At index %d, values are %f, %f.\n", i, err_checker[i], ret[i]);
				break;
			}
		}
		// if (true)
		// 	std::cout << "err checker[32] = " << err_checker[32] << "\n" << "ret[32] = " << ret[32] << "\n";

		return ret;
	}
}