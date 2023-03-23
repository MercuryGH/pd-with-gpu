#pragma once

#include <vector>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <cuda_runtime.h>

namespace pd {
	class Constraint
	{
	public:
		__host__ __device__ Constraint(float wi, int n) : wc(wi), n(n) {}

		// Local solve for A_c'p_c
		// q: 3n * 1 vector indicating positions
		// return: A_c'p_c
		virtual Eigen::VectorXf local_solve(const Eigen::VectorXf& q) const = 0;

		// For global solve computation in one iteration
		// pi: 3k * 1 vector indicating local solution, k depends on the dimension of Ac matrix
		// return: one term in the summation of linear system b
		virtual Eigen::VectorXf get_c_AcTAchpc(const Eigen::VectorXf& pc) const = 0;

		// For global solve linear system A precomputing (prefactoring)
		// return: triplets indicate several entry value in linear system A
		virtual std::vector<Eigen::Triplet<float>> get_c_AcTAc(int n_vertex_offset) const = 0;

		// project_c_AcTAchpc. Local step optimization
		__host__ __device__ virtual void project_c_AcTAchpc(float* __restrict__ b, const float* __restrict__ q) const = 0;

		__host__ __device__ virtual void print_name() const = 0;

        // Called by host but not actually called by device
		__host__ __device__ virtual ~Constraint()
		{
			delete[] vertices;
		}

		int get_involved_vertices(int** vertices) const
		{
			*vertices = this->vertices;
			return n_vertices;
		}

	public:
		int n; // #vertex in the model
		float wc;

		int n_vertices;
		int* vertices;
	};
}