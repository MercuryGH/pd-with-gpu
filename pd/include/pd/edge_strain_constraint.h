#pragma once

#include <pd/constraint.h>
#include <pd/types.h>

namespace pd {
	class EdgeStrainConstraint : public Constraint
	{
	public:
		// __host__ __device__ EdgeStrainConstraint(SimScalar wc, int vi, int vj, float rest_length);
		EdgeStrainConstraint() = default;

		EdgeStrainConstraint(SimScalar wc, VertexIndexType vi, VertexIndexType vj, const PositionData& p):
			Constraint(wc, 2),
			rest_length((p.row(vi) - p.row(vj)).norm())
		{
			assert(vi != vj);

            cudaMallocManaged(&vertices, sizeof(VertexIndexType) * 2);
			vertices[0] = vi;
			vertices[1] = vj;
		}

		Constraint* clone() const override
		{
			return new EdgeStrainConstraint(*this);
		}

		std::vector<Eigen::Triplet<SimScalar>> get_c_AcTAc(int n_vertex_offset) const override;

		__host__ __device__ void project_c_AcTAchpc(SimScalar* __restrict__ b, const SimScalar* __restrict__ q) const override;

		__host__ __device__ void print_name() const override
		{
			printf("EdgeStrainConstraint\n");
		}

		__host__ __device__ ~EdgeStrainConstraint() override {}

		// getters
		__host__ __device__ VertexIndexType vi() const { return vertices[0]; }
		__host__ __device__ VertexIndexType vj() const { return vertices[1]; }

	private:
		SimScalar rest_length;
	};
}