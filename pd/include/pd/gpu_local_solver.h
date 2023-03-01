#pragma once
#include <cuda_runtime.h>
#include <pd/constraint.h>
#include <pd/positional_constraint.h>
#include <pd/edge_length_constraint.h>
#include <pd/types.h>

namespace pd
{
	class GpuLocalSolver
	{
	public:
		void free_local_gpu_memory_entry();
		void gpu_local_step_solver_malloc(int n);
		void gpu_object_creation_serial(const Constraints& constraints);
		void gpu_object_creation_parallel(const Constraints& constraints);
		void gpu_local_step_entry(const Eigen::VectorXf& q_nplus1, Eigen::VectorXf& b);

	private:
		int n_constraints{ 0 };
		bool is_allocated{ false };

		// host mem
		Constraint** local_constraints; // aux

		// dev mem
		Constraint** d_local_constraints; // array of constraints*
		int* d_local_cnt; // GPU counter for objects (use in serial creation only)
		float* d_q_nplus1;
		float* d_b;
	};

	__global__ void create_local_gpu_constraints(
		int n1,
		float* __restrict__ wi_pcs,
		int* __restrict__ vi_pcs,
		int* __restrict__ n_pcs,
		float* __restrict__ x0_pcs,
		float* __restrict__ y0_pcs,
		float* __restrict__ z0_pcs,

		int n2,
		float* __restrict__ wi_elcs,
		int* __restrict__ vi_elcs,
		int* __restrict__ vj_elcs,
		int* __restrict__ n_elcs,
		float* __restrict__ rest_length_elcs,

		int* __restrict__ d_local_cnt,
		Constraint** __restrict__ d_local_constraints);

	__global__ void test_kernel();

	__global__ void create_local_gpu_positional_constraint(
		float wi, int vi, int n, float x0, float y0, float z0, 
		Constraint** __restrict__ d_local_constraints, int* __restrict__ d_local_cnt);

	__global__ void create_local_gpu_edge_length_constraint(
		float wi, int vi, int vj, int n, float rest_length,
		Constraint** __restrict__ d_local_constraints, int* __restrict__ d_local_cnt);

	__global__ void free_constraints(
		Constraint** __restrict__ d_local_constraints, int n_constraints);

	__global__ void gpu_local_step(
		float* __restrict__ d_b, const float* __restrict__ d_q_nplus1,
		Constraint** __restrict__ d_local_constraints, int* __restrict__ d_local_cnt, int n_constraints);
}