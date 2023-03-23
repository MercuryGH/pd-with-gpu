#include <pd/gpu_local_solver.h>
#include <util/helper_cuda.h>

namespace pd
{
	// call destructor for all device constraints
	__global__ void free_constraints(
		Constraint **__restrict__ d_local_constraints, int n_constraints)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < n_constraints)
		{
			delete[] d_local_constraints[idx]->vertices;
		}
	}

	void GpuLocalSolver::free_local_gpu_memory_entry()
	{
		if (is_allocated)
		{
			const int n_blocks = n_constraints / WARP_SIZE + (n_constraints % WARP_SIZE == 0 ? 0 : 1);

			if (n_constraints != 0)
				free_constraints<<<n_blocks, WARP_SIZE>>>(d_local_constraints, n_constraints);

			for (int i = 0; i < n_constraints; i++)
			{
				checkCudaErrors(cudaFree(local_constraints[i]));
			}
			free(local_constraints);

			checkCudaErrors(cudaDeviceSynchronize());

			checkCudaErrors(cudaFree(d_local_constraints));
			checkCudaErrors(cudaFree(d_local_cnt));
			checkCudaErrors(cudaFree(d_b));
			checkCudaErrors(cudaFree(d_q_nplus1));

			is_allocated = false;
		}
	}

	void GpuLocalSolver::gpu_local_step_solver_init(int n)
	{
		this->n_constraints = 0;
		checkCudaErrors(cudaMalloc((void **)&d_b, sizeof(float) * n));
		checkCudaErrors(cudaMalloc((void **)&d_q_nplus1, sizeof(float) * n));
	}

	void GpuLocalSolver::gpu_object_creation_serial(const Constraints &constraints)
	{
		// initialize memory on gpu
		// d_local_constraints = new Constraint * [n];
		// d_local_cnt = new int(0);
		this->n_constraints += constraints.size();
		checkCudaErrors(cudaMalloc((void **)&d_local_constraints, sizeof(Constraint *) * constraints.size()));
		checkCudaErrors(cudaMalloc((void **)&d_local_cnt, sizeof(int *)));
		cudaMemset(d_local_cnt, 0, sizeof(int));

		// copy all constraints to gpu (serial code)
		for (const auto &constraint : constraints)
		{
			if (auto p = dynamic_cast<const PositionalConstraint *>(constraint.get()))
			{
				create_local_gpu_positional_constraint<<<1, 1>>>(p->wc, p->vi, p->n, p->x0, p->y0, p->z0, d_local_constraints, d_local_cnt);
			}
			else if (auto p = dynamic_cast<const EdgeLengthConstraint *>(constraint.get()))
			{
				create_local_gpu_edge_length_constraint<<<1, 1>>>(p->wc, p->vi, p->vj, p->n, p->rest_length, d_local_constraints, d_local_cnt);
			}
			else
			{
				printf("Type Error while creating object on the GPU!\n");
				assert(false);
			}
		}

		is_allocated = true;
	}

	void GpuLocalSolver::gpu_object_creation_parallel(const std::unordered_map<int, DeformableMesh>& models)
	{
		std::vector<float> t1_pcs;
		std::vector<int> t2_pcs;
		std::vector<int> t3_pcs;
		std::vector<float> t4_pcs;
		std::vector<float> t5_pcs;
		std::vector<float> t6_pcs;

		std::vector<float> t1_elcs;
		std::vector<int> t2_elcs;
		std::vector<int> t3_elcs;
		std::vector<int> t4_elcs;
		std::vector<float> t5_elcs;

		// copy all constraints (serial code but fast)
		int acc = 0;
		for (const auto& [id, model] : models)
		{
			const Constraints& constraints = model.get_all_constraints();
			int n = model.positions().rows();
			this->n_constraints += constraints.size();

			for (const auto& constraint : constraints)
			{
				if (auto p = dynamic_cast<const PositionalConstraint *>(constraint.get()))
				{
					t1_pcs.push_back(p->wc);
					t2_pcs.push_back(acc + p->vi);
					t3_pcs.push_back(p->n);
					t4_pcs.push_back(p->x0);
					t5_pcs.push_back(p->y0);
					t6_pcs.push_back(p->z0);
				}
				else if (auto p = dynamic_cast<const EdgeLengthConstraint *>(constraint.get()))
				{
					t1_elcs.push_back(p->wc);
					t2_elcs.push_back(acc + p->vi);
					t3_elcs.push_back(acc + p->vj);
					t4_elcs.push_back(p->n);
					t5_elcs.push_back(p->rest_length);
				}
				else
				{
					printf("Type Error while creating object on the GPU!\n");
					assert(false);
				}
			}
			acc += n;
		}

		// initialize memory on gpu
		// d_local_constraints = new Constraint * [n];
		// d_local_cnt = new int(0);
		checkCudaErrors(cudaMalloc((void **)&d_local_constraints, sizeof(Constraint *) * n_constraints));
		checkCudaErrors(cudaMalloc((void **)&d_local_cnt, sizeof(int *)));
		cudaMemset(d_local_cnt, 0, sizeof(int));

		// dev mem
		int n1 = t1_pcs.size();
		float *wi_pcs;
		int *vi_pcs;
		int *n_pcs;
		float *x0_pcs;
		float *y0_pcs;
		float *z0_pcs;

		int n2 = t1_elcs.size();
		float *wi_elcs;
		int *vi_elcs;
		int *vj_elcs;
		int *n_elcs;
		float *rest_length_elcs;

		// copy dev mem
		checkCudaErrors(cudaMalloc((void **)&wi_pcs, sizeof(float) * n1));
		checkCudaErrors(cudaMalloc((void **)&vi_pcs, sizeof(int) * n1));
		checkCudaErrors(cudaMalloc((void **)&n_pcs, sizeof(int) * n1));
		checkCudaErrors(cudaMalloc((void **)&x0_pcs, sizeof(float) * n1));
		checkCudaErrors(cudaMalloc((void **)&y0_pcs, sizeof(float) * n1));
		checkCudaErrors(cudaMalloc((void **)&z0_pcs, sizeof(float) * n1));

		checkCudaErrors(cudaMalloc((void **)&wi_elcs, sizeof(float) * n2));
		checkCudaErrors(cudaMalloc((void **)&vi_elcs, sizeof(int) * n2));
		checkCudaErrors(cudaMalloc((void **)&vj_elcs, sizeof(int) * n2));
		checkCudaErrors(cudaMalloc((void **)&n_elcs, sizeof(int) * n2));
		checkCudaErrors(cudaMalloc((void **)&rest_length_elcs, sizeof(float) * n2));

		checkCudaErrors(cudaMemcpy(wi_pcs, t1_pcs.data(), sizeof(float) * n1, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(vi_pcs, t2_pcs.data(), sizeof(int) * n1, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(n_pcs, t3_pcs.data(), sizeof(int) * n1, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(x0_pcs, t4_pcs.data(), sizeof(float) * n1, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(y0_pcs, t5_pcs.data(), sizeof(float) * n1, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(z0_pcs, t6_pcs.data(), sizeof(float) * n1, cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMemcpy(wi_elcs, t1_elcs.data(), sizeof(float) * n2, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(vi_elcs, t2_elcs.data(), sizeof(int) * n2, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(vj_elcs, t3_elcs.data(), sizeof(int) * n2, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(n_elcs, t4_elcs.data(), sizeof(int) * n2, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(rest_length_elcs, t5_elcs.data(), sizeof(float) * n2, cudaMemcpyHostToDevice));

		local_constraints = (Constraint **)malloc(sizeof(Constraint *) * n_constraints);

		for (int i = 0; i < n_constraints; i++)
		{
			if (i < n1)
			{
				checkCudaErrors(cudaMalloc(
					(void **)&local_constraints[i],
					sizeof(PositionalConstraint)));
			}
			else if (i < n1 + n2)
			{
				checkCudaErrors(cudaMalloc(
					(void **)&local_constraints[i],
					sizeof(EdgeLengthConstraint)));
			}
		}

		// allocation
		checkCudaErrors(
			cudaMemcpy(
				d_local_constraints,
				local_constraints,
				sizeof(Constraint*) * n_constraints,
				cudaMemcpyHostToDevice));

		// Note: when the b in <<<a, b>>> is too large, cuda will refuse to call the kernel function
		// but leaving no warning or error!
		const int n_blocks = n_constraints / WARP_SIZE + (n_constraints % WARP_SIZE == 0 ? 0 : 1);

		if (n_constraints != 0)
			create_local_gpu_constraints<<<n_blocks, WARP_SIZE>>>(
				n1,
				wi_pcs,
				vi_pcs,
				n_pcs,
				x0_pcs,
				y0_pcs,
				z0_pcs,

				n2,
				wi_elcs,
				vi_elcs,
				vj_elcs,
				n_elcs,
				rest_length_elcs,

				d_local_cnt,
				d_local_constraints);

		checkCudaErrors(cudaDeviceSynchronize());

		// free
		checkCudaErrors(cudaFree(wi_pcs));
		checkCudaErrors(cudaFree(vi_pcs));
		checkCudaErrors(cudaFree(n_pcs));
		checkCudaErrors(cudaFree(x0_pcs));
		checkCudaErrors(cudaFree(y0_pcs));
		checkCudaErrors(cudaFree(z0_pcs));

		checkCudaErrors(cudaFree(wi_elcs));
		checkCudaErrors(cudaFree(vi_elcs));
		checkCudaErrors(cudaFree(vj_elcs));
		checkCudaErrors(cudaFree(n_elcs));
		checkCudaErrors(cudaFree(rest_length_elcs));

		is_allocated = true;
	}

	void GpuLocalSolver::gpu_local_step_entry(const Eigen::VectorXf &q_nplus1, Eigen::VectorXf &b)
	{
		const int n = b.size();
		checkCudaErrors(cudaMemcpy(d_q_nplus1, q_nplus1.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_b, b.data(), sizeof(float) * n, cudaMemcpyHostToDevice));

		const int n_blocks = n_constraints / WARP_SIZE + (n_constraints % WARP_SIZE == 0 ? 0 : 1);

		if (n_constraints != 0)
			gpu_local_step<<<n_blocks, WARP_SIZE>>>(d_b, d_q_nplus1, d_local_constraints, d_local_cnt, n_constraints);

		checkCudaErrors(cudaMemcpy(b.data(), d_b, sizeof(float) * n, cudaMemcpyDeviceToHost));
	}

    template<typename T>
	__device__ void fix_vtable_pointer(T* obj)
	{
		T temp = T(*obj);
		memcpy(obj, &temp, sizeof(T));
	}

	__global__ void create_local_gpu_constraints(
		int n1,
		float *__restrict__ wi_pcs,
		int *__restrict__ vi_pcs,
		int *__restrict__ n_pcs,
		float *__restrict__ x0_pcs,
		float *__restrict__ y0_pcs,
		float *__restrict__ z0_pcs,

		int n2,
		float *__restrict__ wi_elcs,
		int *__restrict__ vi_elcs,
		int *__restrict__ vj_elcs,
		int *__restrict__ n_elcs,
		float *__restrict__ rest_length_elcs,

		int *__restrict__ d_local_cnt,
		Constraint **__restrict__ d_local_constraints)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < n1)
		{
			// Warning: DO NOT new objects in device code directly. Heap memory may be run out.
			// Do CudaMalloc in host code instead.

			PositionalConstraint *pc = (PositionalConstraint *)d_local_constraints[idx];
			// assert(pc != nullptr);
			fix_vtable_pointer<PositionalConstraint>(pc);
			pc->wc = wi_pcs[idx];
			pc->n = n_pcs[idx];
			pc->vi = vi_pcs[idx];
			pc->x0 = x0_pcs[idx];
			pc->y0 = y0_pcs[idx];
			pc->z0 = z0_pcs[idx];
			pc->n_vertices = 1;
			pc->vertices = new int[1]{pc->vi};
			// pc->print_name();
			// atomicAdd(d_local_cnt, *d_local_cnt + 1);
		}
		else if (idx < n1 + n2)
		{
			idx -= n1;
			EdgeLengthConstraint *elc = (EdgeLengthConstraint *)d_local_constraints[idx + n1];
			// assert(elc != nullptr);
			fix_vtable_pointer<EdgeLengthConstraint>(elc);

			elc->wc = wi_elcs[idx];
			elc->n = n_elcs[idx];
			elc->vi = vi_elcs[idx];
			elc->vj = vj_elcs[idx];
			elc->rest_length = rest_length_elcs[idx];
			elc->n_vertices = 2;
			elc->vertices = new int[2]{elc->vi, elc->vj};
			// elc->print_name();
			// atomicAdd(d_local_cnt, *d_local_cnt + 1);
		}
	}

	__global__ void create_local_gpu_positional_constraint(
		float wi, int vi, int n, float x0, float y0, float z0,
		Constraint **__restrict__ d_local_constraints, int *__restrict__ d_local_cnt)
	{
		d_local_constraints[*d_local_cnt] = new PositionalConstraint(wi, vi, n, x0, y0, z0);
		(*d_local_cnt)++;
	}

	__global__ void create_local_gpu_edge_length_constraint(
		float wi, int vi, int vj, int n, float rest_length,
		Constraint **__restrict__ d_local_constraints, int *__restrict__ d_local_cnt)
	{
		d_local_constraints[*d_local_cnt] = new EdgeLengthConstraint(wi, vi, vj, n, rest_length);
		(*d_local_cnt)++;
	}

	__global__ void gpu_local_step(
		float *__restrict__ d_b, const float *__restrict__ d_q_nplus1,
		Constraint **__restrict__ d_local_constraints, int *__restrict__ d_local_cnt, int n_constraints)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		// if (idx == 0)
		//{
		//	printf("%d %d\n", *d_local_cnt, n_constraints);
		// }
		//  possible memory access error when idx is large. Can debug it using NSight later
		// if (idx > 42500)
		//	return;
		// assert(*d_local_cnt == n_constraints);

		if (idx < n_constraints)
		{
			Constraint *constraint = d_local_constraints[idx];

			constraint->project_c_AcTAchpc(d_b, d_q_nplus1);
		}
	}

	__global__ void test_kernel()
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;

		printf("idx = %d\n", idx);
	}
}