#pragma once

#include <pd/constraint.h>
#include <pd/positional_constraint.h>
#include <pd/edge_strain_constraint.h>
#include <pd/bending_constraint.h>
#include <pd/tet_strain_constraint.h>
#include <pd/types.h>

namespace pd
{
	class GpuLocalSolver
	{
	public:
		GpuLocalSolver();
		~GpuLocalSolver();

		/**
		 * @brief Solver free local step GPU memory.
		 * Call when dirty
		 */
		void free_local_gpu_memory_entry();

		void gpu_local_step_solver_alloc(int n);

		/**
		 * @brief Enable objects on GPU serially
		 * @deprecated Very slow, and not compatible with unified memory,
		 * not implemented anymore
		 * @warning Currently not use it
		 */
		void gpu_object_creation_serial(const std::vector<pd::Constraint*>& constraints);

		/**
		 * @brief Enable objects on GPU parallelly
		 * @remark First clone all objects serially, then fix their vtables parallelly
		 * @param models all deformable meshes
		 */
		void gpu_object_creation_parallel(const std::unordered_map<MeshIDType, DeformableMesh>& models);
		void gpu_local_step_entry(const SimPositions& q_nplus1, SimVectorX& b);

	private:
		struct Impl;
		Impl* impl;
	};

	// virtual table handling
	template<typename T>
	static void restore_dev_vtable_entry(const std::vector<T*>& cs);

	template<typename T>
	static void restore_host_vtable_entry(const std::vector<T*>& cs);

	template<typename T>
	static __host__ __device__ void fix_vtable_pointer(const T* obj);

	/**
	 * @brief This kernel function fixes all cloned constraints' vtable to device memory,
	 * also create GPU (copy) cache on the constraints.
	 *
	 * @tparam T Constraint type
	 * @param constraints cloned constraints with type T
	 * @param n number of constraints
	 */
	template<typename T>
	__global__ void restore_dev_vtables(const T* const* __restrict__ constraints, int n);

	/**
	 * @brief Call destructor for all device constraints
	 * @bug This function fails when n_constraints is large, unknown reason.
	 * Calling new/delete in device code with some newer devices presents this problem.
	 * @warning Currently not use it
	 *
	 * @param d_cloned_constraints Constraints to be free
	 * @param n_constraints #d_cloned_constraints
	 */
	__global__ void free_constraints(
		Constraint** __restrict__ d_cloned_constraints, int n_constraints);

	/**
	 * @brief GPU implemented local step
	 *
	 * @param d_b @see formula
	 * @param d_q_nplus1 @see formula
	 * @param d_cloned_constraints all constraints
	 * @param n_constraints #constraints
	 */
	__global__ void gpu_local_step(
		SimScalar* __restrict__ d_b, const SimScalar* __restrict__ d_q_nplus1,
		Constraint** __restrict__ d_cloned_constraints, int n_constraints);

	__global__ void test_kernel();
}
