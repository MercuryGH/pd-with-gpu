#pragma once

#include <pd/constraint.h>
#include <pd/deformable_mesh.h>

namespace pd {
	class PositionalConstraint : public Constraint
	{
	public:
		__host__ __device__ PositionalConstraint(float wi, int vi, int n, float x0, float y0, float z0);

		PositionalConstraint(float wi, int vi, const Positions& p) :
			Constraint(wi, p.rows()),
			vi(vi)
		{
			Eigen::VectorXf p0 = p.row(vi).transpose().cast<float>();
			x0 = p0.x();
			y0 = p0.y();
			z0 = p0.z();
			n_vertices = 1;
			vertices = new int[n_vertices] {vi};
		}

		Eigen::VectorXf local_solve(const Eigen::VectorXf& q) const override;
		Eigen::VectorXf get_i_wiSiTAiTBipi(const Eigen::VectorXf& pi) const override;
		std::vector<Eigen::Triplet<float>> get_A_wiSiTAiTAiSi() const override;

		__host__ __device__ void project_i_wiSiTAiTBipi(float* __restrict__ b, const float* __restrict__ q) const override;

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