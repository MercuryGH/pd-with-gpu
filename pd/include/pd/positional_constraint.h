#pragma once

#include <pd/constraint.h>
#include <pd/deformable_mesh.h>

namespace pd {
	class PositionalConstraint : public Constraint
	{
	public:
		PositionalConstraint(float wi, int vi, const Positions &p): 
			Constraint(wi, p.rows()),
			n(p.rows()),
			vi(vi),
			p0(p.row(vi).transpose().cast<float>()) {}

		Eigen::VectorXf local_solve(const Eigen::VectorXf& q) const override;
		Eigen::VectorXf get_i_wiSiTAiTBipi(const Eigen::VectorXf& pi) const override;
		std::vector<Eigen::Triplet<float>> get_A_wiSiTAiTAiSi() const override;

	private:
		int n;
		int vi;

	    // fixed vertex position
		Eigen::Vector3f p0;
	};
}