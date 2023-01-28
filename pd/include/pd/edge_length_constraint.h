#pragma once

#include <pd/constraint.h>

#include <pd/types.h>

namespace pd {
	class EdgeLengthConstraint : public Constraint
	{
	public:
		EdgeLengthConstraint(float wi, int vi, int vj, const Positions& p) :
			Constraint(wi, p.rows()),
			vi(vi),
			vj(vj),
			rest_length((p.row(vi) - p.row(vj)).norm())
		{
			assert(vi != vj);
		}

		Eigen::VectorXf local_solve(const Eigen::VectorXf& q) const override;
		Eigen::VectorXf get_i_wiSiTAiTBipi(const Eigen::VectorXf& pi) const override;
		std::vector<Eigen::Triplet<float>> get_A_wiSiTAiTAiSi() const override;

	private:
		int vi, vj;

		float rest_length;
	};
}