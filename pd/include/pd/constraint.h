#pragma once

#include <vector>
#include <Eigen/Core>
#include <Eigen/SparseCore>
//#include <cuda_runtime.h>

namespace pd {
	class Constraint
	{
	public:
		Constraint(float wi, int n): wi(wi), n(n) {}

		// Local solve for pi
		// q: 3n * 1 vector indicating positions
		// return: pi
		virtual Eigen::VectorXf local_solve(const Eigen::VectorXf &q) const = 0;

		// For global solve computation in one iteration
		// pi: 3k * 1 vector indicating local solution, k depends on the dimension of Si matrix
		// return: one term in the summation of linear system b
		virtual Eigen::VectorXf get_i_wiSiTAiTBipi(const Eigen::VectorXf &pi) const = 0;

		// For global solve linear system A precomputing (prefactoring)
		// return: triplets indicate several entry value in linear system A
		virtual std::vector<Eigen::Triplet<float>> get_A_wiSiTAiTAiSi() const = 0; 

	protected:
		int n; // #vertex
		float wi;
	};
}