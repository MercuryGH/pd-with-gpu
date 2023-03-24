#pragma once

#include <pd/constraint.h>
#include <pd/types.h>

namespace pd
{
    class BendingConstraint: public Constraint
    {
    public:
		__host__ __device__ BendingConstraint(float wc, int n);

		BendingConstraint(float wc, const Positions& p) : Constraint(wc, 0, nullptr)
		{
		}

		Eigen::VectorXf local_solve(const Eigen::VectorXf& q) const override;
		std::vector<Eigen::Triplet<float>> get_c_AcTAc(int n_vertex_offset) const override;

		__host__ __device__ void project_c_AcTAchpc(float* __restrict__ b, const float* __restrict__ q) const override;

		__host__ __device__ void print_name() const override
		{
			printf("BendingConstraint\n");
		}

	public:

    };
}