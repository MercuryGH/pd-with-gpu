#pragma once

#include <pd/constraint.h>
#include <pd/deformable_mesh.h>

namespace pd {
	class PositionalConstraint : public Constraint
	{
	public:
		__host__ __device__ PositionalConstraint(float wc, int vi, float x0, float y0, float z0);

		PositionalConstraint(float wc, int vi, const Positions& p) :
			Constraint(wc, 1, new int[1] {vi}),
			vi(vi)
		{
			Eigen::VectorXf p0 = p.row(vi).transpose().cast<float>();
			x0 = p0.x();
			y0 = p0.y();
			z0 = p0.z();
		}

		Eigen::VectorXf local_solve(const Eigen::VectorXf& q) const override;
		std::vector<Eigen::Triplet<float>> get_c_AcTAc(int n_vertex_offset) const override;

		__host__ __device__ void project_c_AcTAchpc(float* __restrict__ b, const float* __restrict__ q) const override;

		__host__ __device__ void print_name() const override
		{
			printf("PositionalConstraint\n");
		}

	public:
		int vi;

		// fixed vertex position
		float x0, y0, z0;
		//Eigen::Vector3f p0;
	};
}