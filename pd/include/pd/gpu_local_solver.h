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
		void free_local_gpu_memory_entry(int n_constraints);
		void gpu_local_step_solver_malloc(int n);
		void gpu_object_creation(const Constraints& constraints);
		void gpu_local_step_entry(const Eigen::VectorXf& q_nplus1, Eigen::VectorXf& b, int n_constraints);


	private:
		bool is_allocated{ false };

		// dev mem
		Constraint** d_local_constraints; // array of constraints*
		int* d_local_cnt; // GPU counter for objects
		float* d_q_nplus1;
		float* d_b;
	};

	__global__ void create_local_gpu_positional_constraint(
		float wi, int vi, int n, float x0, float y0, float z0, 
		Constraint** __restrict__ d_local_constraints, int* __restrict__ d_local_cnt);

	__global__ void create_local_gpu_edge_length_constraint(
		float wi, int vi, int vj, int n, float rest_length,
		Constraint** __restrict__ d_local_constraints, int* __restrict__ d_local_cnt);

	__global__ void free_constraints(
		Constraint** __restrict__ d_local_constraints);

	__global__ void gpu_local_step(
		float* __restrict__ d_b, const float* __restrict__ d_q_nplus1,
		Constraint** __restrict__ d_local_constraints, int* __restrict__ d_local_cnt, int n_constraints);
}