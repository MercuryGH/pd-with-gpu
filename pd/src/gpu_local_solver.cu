#include <pd/gpu_local_solver.h>
#include <util/helper_cuda.h>

namespace pd
{
	__global__ void free_constraints(
		Constraint** __restrict__ d_local_constraints)
	{
		int idx = threadIdx.x;

		delete d_local_constraints[idx];
	}

	void GpuLocalSolver::free_local_gpu_memory_entry(int n_constraints)
	{
		free_constraints<<<1, n_constraints>>>(d_local_constraints);
		
		checkCudaErrors(cudaFree(d_local_constraints));
		checkCudaErrors(cudaFree(d_local_cnt));
		checkCudaErrors(cudaFree(d_b));
		checkCudaErrors(cudaFree(d_q_nplus1));
	}

	void GpuLocalSolver::gpu_local_step_solver_malloc(int n)
	{
		checkCudaErrors(cudaMalloc((void**)&d_b, sizeof(float) * n));
		checkCudaErrors(cudaMalloc((void**)&d_q_nplus1, sizeof(float) * n));
	}

	void GpuLocalSolver::gpu_object_creation(const Constraints& constraints)
	{
		// initialize memory on gpu
		//d_local_constraints = new Constraint * [n];
		//d_local_cnt = new int(0);
		checkCudaErrors(cudaMalloc((void**)&d_local_constraints, sizeof(Constraint*) * constraints.size()));
		checkCudaErrors(cudaMalloc((void**)&d_local_cnt, sizeof(int*)));
		cudaMemset(d_local_cnt, 0, sizeof(int));

		// copy all constraints to gpu (serial code)
		for (const auto& constraint : constraints)
		{
			if (auto p = dynamic_cast<const PositionalConstraint*>(constraint.get()))
			{
				create_local_gpu_positional_constraint << <1, 1 >> > (p->wi, p->vi, p->n, p->x0, p->y0, p->z0, d_local_constraints, d_local_cnt);
			}
			else if (auto p = dynamic_cast<const EdgeLengthConstraint*>(constraint.get()))
			{
				create_local_gpu_edge_length_constraint << <1, 1 >> > (p->wi, p->vi, p->vj, p->n, p->rest_length, d_local_constraints, d_local_cnt);
			}
			else
			{
				printf("Type Error while creating object on the GPU!\n");
				assert(false);
			}
		}
	}

	void GpuLocalSolver::gpu_local_step_entry(const Eigen::VectorXf& q_nplus1, Eigen::VectorXf& b, int n_constraints)
	{
		const int n = b.size();
		checkCudaErrors(cudaMemcpy(d_q_nplus1, q_nplus1.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_b, b.data(), sizeof(float) * n, cudaMemcpyHostToDevice));

		const int n_blocks = n_constraints / WARP_SIZE + (n_constraints % WARP_SIZE == 0 ? 0 : 1);
		gpu_local_step << <n_blocks, WARP_SIZE >> > (d_b, d_q_nplus1, d_local_constraints, d_local_cnt, n_constraints);

		checkCudaErrors(cudaMemcpy(b.data(), d_b, sizeof(float) * n, cudaMemcpyDeviceToHost));
	}

	__global__ void create_local_gpu_positional_constraint(
		float wi, int vi, int n, float x0, float y0, float z0,
		Constraint** __restrict__ d_local_constraints, int* __restrict__ d_local_cnt)
	{
		d_local_constraints[*d_local_cnt] = new PositionalConstraint(wi, vi, n, x0, y0, z0);
		(*d_local_cnt)++;
	}

	__global__ void create_local_gpu_edge_length_constraint(
		float wi, int vi, int vj, int n, float rest_length,
		Constraint** __restrict__ d_local_constraints, int* __restrict__ d_local_cnt)
	{
		d_local_constraints[*d_local_cnt] = new EdgeLengthConstraint(wi, vi, vj, n, rest_length);
		(*d_local_cnt)++;
	}

	__global__ void gpu_local_step(
		float* __restrict__ d_b, const float* __restrict__ d_q_nplus1,
		Constraint** __restrict__ d_local_constraints, int* __restrict__ d_local_cnt, int n_constraints)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		//int n_constraints = *d_local_cnt;

		assert(*d_local_cnt == n_constraints);

		if (idx < n_constraints)
		{
			Constraint* constraint = d_local_constraints[idx];

			constraint->project_i_wiSiTAiTBipi(d_b, d_q_nplus1);
		}
	}
}