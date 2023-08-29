#pragma once

#include <vector>
#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <pd/types.h>

#include <util/cuda_managed.h>

namespace pd {
	/**
	 * @brief Abstract Constraint
	 */
	class Constraint: public util::CudaManaged
	{
	public:
		Constraint() = default;

		__host__ __device__ Constraint(SimScalar wc, int n_vertices) : wc(wc), n_vertices(n_vertices) {}

		/**
		 * @brief Copy constructor
		 */
		__host__ __device__ Constraint(const Constraint& rhs);

		/**
		 * @brief Move constructor
		 */
		Constraint(Constraint&& rhs) noexcept;

		/**
		 * @brief Assignment operator
		 */
		Constraint& operator=(const Constraint& rhs);

		/**
		 * @brief Move assignment
		 */
		Constraint& operator=(Constraint&& rhs) noexcept;

		/**
		 * @brief Deep copy self
		 */
		virtual Constraint* clone() const = 0;

		// Local solve for A_c'p_c
		// q: 3n * 1 vector indicating positions
		// return: A_c'p_c
		// deprecated

		// For global solve linear system A precomputing (prefactoring)
		// return: triplets indicate several entry value in linear system A
		virtual std::vector<Eigen::Triplet<SimScalar>> get_c_AcTAc(int n_vertex_offset) const = 0;

		// project_c_AcTAchpc. Local step optimization
		__host__ __device__ virtual void project_c_AcTAchpc(SimScalar* __restrict__ b, const SimScalar* __restrict__ q) const = 0;

		__host__ __device__ virtual void print_name() const = 0;

        // Called by host but not actually called by device
		__host__ __device__ virtual ~Constraint()
		{
			cudaFree(vertices);
		}

		int get_involved_vertices(VertexIndexType** vertices) const;

		void set_vertex_offset(int n_vertex_offset);

	private:
		void realloc_vertices(int size);

	protected:
		SimScalar wc{ 0.0f };

		int n_vertices{ 0 };
		VertexIndexType* vertices{ nullptr };
	};
}
