#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <vector>
#include <array>
#include <iostream>

#include <pd/edge_length_constraint.h>

namespace pd {
	__host__ __device__ EdgeLengthConstraint::EdgeLengthConstraint(float wi, int vi, int vj, int n, float rest_length) :
		Constraint(wi, n),
		vi(vi),
		vj(vj),
		rest_length(rest_length)
	{
		assert(vi != vj);
		n_vertices = 2;
		vertices = new int[n_vertices] {vi, vj};
	}

	Eigen::VectorXf EdgeLengthConstraint::local_solve(const Eigen::VectorXf& q) const
	{
		Eigen::VectorXf ret;
		ret.resize(6);

		Eigen::Vector3f vi_pos = q.block(3 * vi, 0, 3, 1);
		Eigen::Vector3f vj_pos = q.block(3 * vj, 0, 3, 1);
		Eigen::Vector3f j2i = vj_pos - vi_pos;

		float delta_x = j2i.norm() - rest_length; // This is the constraint set C_i (edge length)
		Eigen::Vector3f j2i_normalized = j2i.normalized();
		
		Eigen::Vector3f pi1 = vi_pos + 0.5f * delta_x * j2i_normalized;
		Eigen::Vector3f pi2 = vj_pos - 0.5f * delta_x * j2i_normalized;

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

	__host__ __device__ void EdgeLengthConstraint::project_i_wiSiTAiTBipi(float* __restrict__ b, const float* __restrict__ q) const
	{
		Eigen::Vector3f vi_pos{ q[3 * vi], q[3 * vi + 1], q[3 * vi + 2] };
		Eigen::Vector3f vj_pos{ q[3 * vj], q[3 * vj + 1], q[3 * vj + 2] };

		Eigen::Vector3f j2i = vj_pos - vi_pos;
		float delta_x = j2i.norm() - rest_length; // This is the constraint set C_i (edge length)
		Eigen::Vector3f j2i_normalized = j2i.normalized();

		Eigen::Vector3f pi1 = vi_pos + 0.5f * delta_x * j2i_normalized;
		Eigen::Vector3f pi2 = vj_pos - 0.5f * delta_x * j2i_normalized;

		for (int i = 0; i < 3; i++)
		{
		#ifdef __CUDA_ARCH__
			atomicAdd(&b[3 * vi + i], 0.5f * wi * (pi1[i] - pi2[i]));
			atomicAdd(&b[3 * vj + i], 0.5f * wi * (pi2[i] - pi1[i]));
		#else
			b[3 * vi + i] += 0.5f * wi * (pi1[i] - pi2[i]);
			b[3 * vj + i] += 0.5f * wi * (pi2[i] - pi1[i]);
		#endif
		}
	}
}