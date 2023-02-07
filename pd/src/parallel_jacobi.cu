#include <pd/parallel_jacobi.h>
#include <util/cpu_timer.h>

namespace pd
{
	// Make sure A is compressed
	// Eigen::SparseMatrix can be converted to CUDA sparse matrix but it's quite tricky
	void ParallelJacobi::set_A(const Eigen::SparseMatrix<float>& A)
	{
		Eigen::MatrixXf _A = Eigen::MatrixXf(A);

		assert(cudaSuccess == cudaMalloc((void**)&d_A, sizeof(float) * _A.size()));

		cudaMemcpy(d_A, _A.data(), sizeof(float) * _A.size(), cudaMemcpyHostToDevice);
	}

	Eigen::VectorXf ParallelJacobi::solve(const Eigen::VectorXf& b)
	{
		Eigen::VectorXf ret;
		ret.resizeLike(b);

		n = b.size();
		assert(cudaSuccess == cudaMalloc((void**)&d_b, sizeof(float) * n));
		assert(cudaSuccess == cudaMalloc((void**)&d_x, sizeof(float) * n));
		assert(cudaSuccess == cudaMalloc((void**)&d_next_x, sizeof(float) * n));

		cudaMemcpy(d_b, b.data(), sizeof(float) * n, cudaMemcpyHostToDevice);
		// set to IEEE-754 zero as iteration initial value
		cudaMemset(d_x, 0, sizeof(float) * n);
		cudaMemset(d_next_x, 0, sizeof(float) * n);

		// The solver iterates for a constant number, not checking error
		//float diff = 0.0f;
		//float eps = 1e-4f;
		if (false)
		{
			int n_blocks = n / WARP_SIZE + (n % WARP_SIZE == 0 ? 0 : 1);
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

		cudaMemcpy(ret.data(), d_x, sizeof(float) * n, cudaMemcpyDeviceToHost);
		return ret;
	}

	__global__ void itr_shfl_down(float* next_x, const float* __restrict__ A, const float* __restrict__  x, const float* __restrict__ b, int n_row, int n_col)
	{
		int col_start = threadIdx.x; // indicates i-th thread in a warp, 0 <= i <= 31
		int row = blockIdx.x;
		int offset = row * n_col;
		float sum = 0.0f;

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

	__global__ void itr_normal(float* next_x, const float* __restrict__ A, const float* __restrict__  x, const float* __restrict__ b, int n_row, int n_col)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < n_row)
		{
			float sum = 0.0f;
			int row_offset = idx * n_col;
			for (int j = 0; j < n_col; j++)
			{
				sum += A[row_offset + j] * x[j];
			}
			sum = sum - A[row_offset + idx] * x[idx];
			next_x[idx] = (b[idx] - sum) / A[row_offset + idx];
		}
	}
}