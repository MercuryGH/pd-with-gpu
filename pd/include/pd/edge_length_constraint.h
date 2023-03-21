#pragma once

#include <pd/constraint.h>

#include <pd/types.h>

namespace pd {
	class EdgeLengthConstraint : public Constraint
	{
	public:
		__host__ __device__ EdgeLengthConstraint(float wi, int vi, int vj, int n, float rest_length);

		EdgeLengthConstraint(float wi, int vi, int vj, const Positions& p) :
			Constraint(wi, p.rows()),
			vi(vi),
			vj(vj),
			rest_length((p.row(vi) - p.row(vj)).norm())
		{
			assert(vi != vj);
			n_vertices = 2;
			vertices = new int[n_vertices] {vi, vj};
		}

		Eigen::VectorXf local_solve(const Eigen::VectorXf& q) const override;
		Eigen::VectorXf get_c_AcTAchpc(const Eigen::VectorXf& pi) const override;
		std::vector<Eigen::Triplet<float>> get_c_AcTAc(int n_vertex_offset) const override;

		__host__ __device__ void project_c_AcTAchpc(float* __restrict__ b, const float* __restrict__ q) const override;

		__host__ __device__ void print_name() const override
		{
			printf("EdgeLengthConstraint\n");
		}


	public:
		int vi, vj;

		float rest_length;
	};
}