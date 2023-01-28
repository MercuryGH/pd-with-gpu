#include <array>
#include <pd/positional_constraint.h>

namespace pd {
	Eigen::VectorXf PositionalConstraint::local_solve(const Eigen::VectorXf& q) const
	{
		return p0;
	}

	Eigen::VectorXf PositionalConstraint::get_i_wiSiTAiTBipi(const Eigen::VectorXf& pi) const
	{
		Eigen::VectorXf ret;
		ret.resize(3 * n);
		ret.setZero();

		// pi = p0 in positional constraint
		ret.block(3 * vi, 0, 3, 1) = wi * p0;
		//std::cout << "PC test = " << ret << "\n";
		return ret;
	}

	std::vector<Eigen::Triplet<float>> PositionalConstraint::get_A_wiSiTAiTAiSi() const
	{
		std::array<Eigen::Triplet<float>, 3u> triplets;
		for (int i = 0; i < 3; i++)
		{
			triplets[i] = { 3 * vi + i, 3 * vi + i, wi };
		}
		return std::vector<Eigen::Triplet<float>>{ triplets.begin(), triplets.end() };
	}
}