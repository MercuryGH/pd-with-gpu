#include <pd/gpu_local_solver.h>
#include <util/gpu_helper.h>
#include <util/cuda_managed.h>

namespace pd
{
	__global__ void free_constraints(
		Constraint **__restrict__ d_cloned_constraints, int n_constraints)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < n_constraints)
		{
			delete d_cloned_constraints[idx];
		}
	}

	void GpuLocalSolver::free_local_gpu_memory_entry()
	{
		if (is_allocated)
		{
			printf("Clean up GPU local constraint memory...\n");

			// fix vtable
			restore_host_vtable_entry(pcs);
			restore_host_vtable_entry(escs);
			restore_host_vtable_entry(bcs);
			restore_host_vtable_entry(tscs);

			// delete all object using host side vtable
			for (auto ptr : cloned_constraints)
			{
				delete ptr;
			}

			cloned_constraints.clear();
			pcs.clear();
			escs.clear();
			bcs.clear();
			tscs.clear();

			checkCudaErrors(cudaDeviceSynchronize());

			checkCudaErrors(cudaFree(d_local_cnt));
			checkCudaErrors(cudaFree(d_b));
			checkCudaErrors(cudaFree(d_q_nplus1));

			is_allocated = false;
		}
	}

	void GpuLocalSolver::gpu_local_step_solver_alloc(int n)
	{
		checkCudaErrors(cudaMalloc((void **)&d_b, sizeof(float) * n));
		checkCudaErrors(cudaMalloc((void **)&d_q_nplus1, sizeof(float) * n));
	}

	void GpuLocalSolver::gpu_object_creation_serial(const thrust::host_vector<pd::Constraint*>& constraints)
	{
		// initialize memory on gpu
		// d_cloned_constraints = new Constraint * [n];
		// d_local_cnt = new int(0);
		this->n_constraints += constraints.size();
		checkCudaErrors(cudaMalloc((void **)&d_cloned_constraints, sizeof(Constraint *) * constraints.size()));
		checkCudaErrors(cudaMalloc((void **)&d_local_cnt, sizeof(int *)));
		cudaMemset(d_local_cnt, 0, sizeof(int));

		// copy all constraints to gpu (serial code)
		for (const auto &constraint : constraints)
		{
			if (auto p = dynamic_cast<const PositionalConstraint *>(constraint))
			{
				// create_local_gpu_positional_constraint<<<1, 1>>>(p->wc, p->vi, p->x0, p->y0, p->z0, d_cloned_constraints, d_local_cnt);
			}
			else if (auto p = dynamic_cast<const EdgeStrainConstraint *>(constraint))
			{
				// create_local_gpu_edge_length_constraint<<<1, 1>>>(p->wc, p->vi, p->vj, p->rest_length, d_cloned_constraints, d_local_cnt);
			}
			else
			{
				printf("Type Error while creating object on the GPU!\n");
				assert(false);
			}
		}

		is_allocated = true;
	}

	template<typename T>
	static void restore_dev_vtable_entry(const thrust::host_vector<T*>& cs)
	{
		const int n_blocks = util::get_n_blocks(cs.size());
		// copy to device to fetch device-side vtable
		thrust::device_vector<const T*> d_cs = cs;
		if (n_blocks > 0)
			restore_dev_vtables<T><<<n_blocks, WARP_SIZE>>>(thrust::raw_pointer_cast(d_cs.data()), d_cs.size());
	}

	template<typename T>
	static void restore_host_vtable_entry(const thrust::host_vector<T*>& cs)
	{
		for (auto ptr : cs)
		{
			// fetch host-side vtable
			fix_vtable_pointer(ptr);
		}
	}

	void GpuLocalSolver::gpu_object_creation_parallel(const std::unordered_map<int, DeformableMesh>& models)
	{
		this->n_constraints = 0;

		int acc = 0;
		for (const auto& [id, model] : models)
		{
			int n = model.positions().rows();
			const auto& constraints = model.get_all_constraints();
			this->n_constraints += constraints.size();

			for (const auto& constraint : constraints)
			{
				Constraint* c = constraint->clone();
				if (auto p = dynamic_cast<const PositionalConstraint*>(c))
				{
					pcs.push_back(p);
				}
				else if (auto p = dynamic_cast<const EdgeStrainConstraint*>(c))
				{
					escs.push_back(p);
				}
				else if (auto p = dynamic_cast<const BendingConstraint*>(c))
				{
					bcs.push_back(p);
				}
				else if (auto p = dynamic_cast<const TetStrainConstraint*>(c))
				{
					tscs.push_back(p);
				}
				else 
				{
					printf("Error occured during dynamic_cast!\n");
				}

				c->set_vertex_offset(acc);
				this->cloned_constraints.push_back(c);
			}

			acc += n;
		}

		// printf("Fix vtable\n");

		// fix vtable
		restore_dev_vtable_entry(pcs);
		restore_dev_vtable_entry(escs);
		restore_dev_vtable_entry(bcs);
		restore_dev_vtable_entry(tscs);

		// printf("#Constraints = %d\n", cloned_constraints.size());

		// copy to device
		d_cloned_constraints = cloned_constraints;
		// d_cloned_constraints = thrust::raw_pointer_cast(cloned_constraints.data());
		is_allocated = true;
	}

	void GpuLocalSolver::gpu_local_step_entry(const Eigen::VectorXf &q_nplus1, Eigen::VectorXf &b)
	{
		const int n = b.size();
		checkCudaErrors(cudaMemcpy(d_q_nplus1, q_nplus1.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_b, b.data(), sizeof(float) * n, cudaMemcpyHostToDevice));

		const int n_blocks = util::get_n_blocks(n_constraints);

		if (n_constraints != 0)
			gpu_local_step<<<n_blocks, WARP_SIZE>>>(d_b, d_q_nplus1, thrust::raw_pointer_cast(d_cloned_constraints.data()), n_constraints);

		checkCudaErrors(cudaMemcpy(b.data(), d_b, sizeof(float) * n, cudaMemcpyDeviceToHost));
	}

	__host__ __device__ void print_ptr_content(void* ptr)
	{
		printf("Addr = %p, Content = %p\n", ptr, *(u_int64_t*)ptr);
	}

    template<typename T>
	__host__ __device__ void fix_vtable_pointer(const T* obj)
	{
		T temp = T(); // default constructor does nothing but allocation on device (hopefully)
		// memcpy((void*)obj, &temp, sizeof(T)); // This causes memory leak

		// peak vtable and its content
		// print_ptr_content((void*)&temp);
		// print_ptr_content((void*)obj);
		memcpy((void*)obj, &temp, sizeof(void*)); // Can this only copy vtable
	}

	template<typename T>
	__global__ void restore_dev_vtables(const T* const* __restrict__ constraints, int n)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < n)
		{
			const T* constraint = constraints[idx];
			fix_vtable_pointer<T>(constraint);
		}
	}

	__global__ void gpu_local_step(
		float *__restrict__ d_b, const float *__restrict__ d_q_nplus1,
		Constraint **__restrict__ d_cloned_constraints, int n_constraints)
	{
		for (int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < n_constraints; idx += blockDim.x * gridDim.x)
		{
			Constraint *constraint = d_cloned_constraints[idx];

			constraint->project_c_AcTAchpc(d_b, d_q_nplus1);
		}
	}

	__global__ void test_kernel()
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;

		printf("idx = %d\n", idx);
	}
}