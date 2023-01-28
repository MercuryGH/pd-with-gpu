#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <vector>
#include <array>
#include <iostream>

#include <pd/edge_length_constraint.h>

namespace pd {
	Eigen::VectorXf EdgeLengthConstraint::local_solve(const Eigen::VectorXf& q) const
	{
		Eigen::VectorXf ret;
		ret.resize(6);

		Eigen::Vector3f vi_pos = q.block(3 * vi, 0, 3, 1);
		Eigen::Vector3f vj_pos = q.block(3 * vj, 0, 3, 1);
		Eigen::Vector3f j2i = vj_pos - vi_pos;

		float delta_x = j2i.norm() - rest_length;
		Eigen::Vector3f j2i_normalized = j2i.normalized();
		
		Eigen::Vector3f pi1 = vi_pos + 0.5 * delta_x * j2i_normalized;
		Eigen::Vector3f pi2 = vj_pos - 0.5 * delta_x * j2i_normalized;

		ret.block(0, 0, 3, 1) = pi1;
		ret.block(3, 0, 3, 1) = pi2;
		//std::cout << "ret = " << ret << "\n";

		return ret;
	}

	Eigen::VectorXf EdgeLengthConstraint::get_i_wiSiTAiTBipi(const Eigen::VectorXf& pi) const
	{
		assert(pi.rows() == 6);
		Eigen::VectorXf ret;
		ret.resize(3 * n);
		ret.setZero();

		Eigen::Vector3f pi1 = pi.block(0, 0, 3, 1);
		Eigen::Vector3f pi2 = pi.block(3, 0, 3, 1);

		for (int i = 0; i < 3; i++)
		{
			ret(3 * vi + i) = wi * 0.5 * (pi1(i) - pi2(i));
		}
		for (int i = 0; i < 3; i++)
		{
			ret(3 * vj + i) = wi * 0.5 * (pi2(i) - pi1(i));
		}

		return ret;
	}

	std::vector<Eigen::Triplet<float>> EdgeLengthConstraint::get_A_wiSiTAiTAiSi() const
	{
		std::vector<Eigen::Triplet<float>> triplets(12);

		for (int i = 0; i < 3; i++)
		{
			triplets.emplace_back(3 * vi + i, 3 * vi + i, 0.5 * wi);
		}
		for (int i = 0; i < 3; i++)
		{
			triplets.emplace_back(3 * vj + i, 3 * vj + i, 0.5 * wi);
		}
		for (int i = 0; i < 3; i++)
		{
			triplets.emplace_back(3 * vi + i, 3 * vj + i, -0.5 * wi);
		}
		for (int i = 0; i < 3; i++)
		{
			triplets.emplace_back(3 * vj + i, 3 * vi + i, -0.5 * wi);
		}

		return std::vector<Eigen::Triplet<float>>{ triplets.begin(), triplets.end() };
	}
}