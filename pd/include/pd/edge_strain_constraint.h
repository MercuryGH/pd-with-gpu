#pragma once

#include <pd/constraint.h>
#include <pd/types.h>

namespace pd {
	class EdgeStrainConstraint : public Constraint
	{
	public:
		// __host__ __device__ EdgeStrainConstraint(float wc, int vi, int vj, float rest_length);
		EdgeStrainConstraint() = default;

		EdgeStrainConstraint(float wc, int vi, int vj, const Positions& p):
			Constraint(wc, 2),
			rest_length((p.row(vi) - p.row(vj)).norm())
		{
			assert(vi != vj);

            cudaMallocManaged(&vertices, sizeof(int) * 2);
			vertices[0] = vi;
			vertices[1] = vj;
		}

		Constraint* clone() const override
		{
			return new EdgeStrainConstraint(*this);
		}

		Eigen::VectorXf local_solve(const Eigen::VectorXf& q) const override;
		std::vector<Eigen::Triplet<float>> get_c_AcTAc(int n_vertex_offset) const override;

		__host__ __device__ void project_c_AcTAchpc(float* __restrict__ b, const float* __restrict__ q) const override;

		__host__ __device__ void print_name() const override
		{
			printf("EdgeStrainConstraint\n");
		}

		__host__ __device__ ~EdgeStrainConstraint() override {}

		// getters
		__host__ __device__ int vi() const { return vertices[0]; }
		__host__ __device__ int vj() const { return vertices[1]; }

	private:
		float rest_length;
	};
}