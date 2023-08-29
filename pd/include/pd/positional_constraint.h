#pragma once

#include <pd/constraint.h>
#include <pd/deformable_mesh.h>

namespace pd {
	class PositionalConstraint : public Constraint
	{
	public:
		// __host__ __device__ PositionalConstraint(SimScalar wc, int vi, float x0, float y0, float z0);
		PositionalConstraint() = default;

		PositionalConstraint(SimScalar wc, VertexIndexType vi, const PositionData& p) :
			Constraint(wc, 1),
			p0(p.row(vi).transpose().cast<SimScalar>())
		{
			cudaMallocManaged(&vertices, sizeof(VertexIndexType) * 1);
			vertices[0] = vi;
		}

		PositionalConstraint(const PositionalConstraint& rhs) = default;
		PositionalConstraint(PositionalConstraint&& rhs) noexcept = default;
		PositionalConstraint& operator=(const PositionalConstraint& rhs) = default;
		PositionalConstraint& operator=(PositionalConstraint&& rhs) noexcept = default;

		Constraint* clone() const override
		{
			return new PositionalConstraint(*this);
		}

		std::vector<Eigen::Triplet<SimScalar>> get_c_AcTAc(int n_vertex_offset) const override;

		__host__ __device__ void project_c_AcTAchpc(SimScalar* __restrict__ b, const SimScalar* __restrict__ q) const override;

		__host__ __device__ void print_name() const override
		{
			printf("PositionalConstraint\n");
		}

		__host__ __device__ ~PositionalConstraint() override { /* printf("Delete PC\n"); */ }

		__host__ __device__ VertexIndexType vi() const { return vertices[0]; }

	private:
		// fixed vertex position
		SimVector3 p0;
	};
}
