#pragma once

#include <pd/constraint.h>
#include <pd/types.h>

namespace pd {
	class EdgeStrainConstraint : public Constraint
	{
	public:
		__host__ __device__ EdgeStrainConstraint(float wc, int vi, int vj, float rest_length);

		EdgeStrainConstraint(float wc, int vi, int vj, const Positions& p) :
			Constraint(wc, 2, new int[2] {vi, vj}),
			vi(vi),
			vj(vj),
			rest_length((p.row(vi) - p.row(vj)).norm())
		{
			assert(vi != vj);
		}

		Eigen::VectorXf local_solve(const Eigen::VectorXf& q) const override;
		std::vector<Eigen::Triplet<float>> get_c_AcTAc(int n_vertex_offset) const override;

		__host__ __device__ void project_c_AcTAchpc(float* __restrict__ b, const float* __restrict__ q) const override;

		__host__ __device__ void print_name() const override
		{
			printf("EdgeStrainConstraint\n");
		}


	public:
		int vi, vj;

		float rest_length;
	};
}