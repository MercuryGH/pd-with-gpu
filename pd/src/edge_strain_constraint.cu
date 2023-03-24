#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <vector>
#include <array>
#include <iostream>

#include <pd/edge_strain_constraint.h>

namespace pd {
	__host__ __device__ EdgeStrainConstraint::EdgeStrainConstraint(float wc, int vi, int vj, float rest_length) :
		Constraint(wc, 2, new int[2] {vi, vj}),
		vi(vi),
		vj(vj),
		rest_length(rest_length)
	{
		assert(vi != vj);
	}

	Eigen::VectorXf EdgeStrainConstraint::local_solve(const Eigen::VectorXf& q) const
	{
		Eigen::VectorXf ret;
		ret.resize(3);

		// printf("%d %d\n", vi, vj);

		Eigen::Vector3f vi_pos = q.block(3 * vi, 0, 3, 1);
		Eigen::Vector3f vj_pos = q.block(3 * vj, 0, 3, 1);
		Eigen::Vector3f j2i = vj_pos - vi_pos;

		float delta_x = j2i.norm() - rest_length; // This is the constraint set C_i (edge length)
		Eigen::Vector3f j2i_normalized = j2i.normalized();
		
		Eigen::Vector3f pc1 = vi_pos + 0.5f * delta_x * j2i_normalized;
		Eigen::Vector3f pc2 = vj_pos - 0.5f * delta_x * j2i_normalized;

		ret = pc1 - pc2;

		//std::cout << "ret = " << ret << "\n";
		return ret;
	}

	std::vector<Eigen::Triplet<float>> EdgeStrainConstraint::get_c_AcTAc(int n_vertex_offset) const
	{
		std::vector<Eigen::Triplet<float>> triplets(12);

		for (int i = 0; i < 3; i++)
		{
			triplets.emplace_back(3 * n_vertex_offset + 3 * vi + i, 3 * n_vertex_offset + 3 * vi + i, wc);
		}
		for (int i = 0; i < 3; i++)
		{
			triplets.emplace_back(3 * n_vertex_offset + 3 * vj + i, 3 * n_vertex_offset + 3 * vj + i, wc);
		}
		for (int i = 0; i < 3; i++)
		{
			triplets.emplace_back(3 * n_vertex_offset + 3 * vi + i, 3 * n_vertex_offset + 3 * vj + i, -wc);
		}
		for (int i = 0; i < 3; i++)
		{
			triplets.emplace_back(3 * n_vertex_offset + 3 * vj + i, 3 * n_vertex_offset + 3 * vi + i, -wc);
		}

		return std::vector<Eigen::Triplet<float>>{ triplets.begin(), triplets.end() };
	}

	__host__ __device__ void EdgeStrainConstraint::project_c_AcTAchpc(float* __restrict__ b, const float* __restrict__ q) const
	{
		// #vertex offset is already included
		Eigen::Vector3f vi_pos{ q[3 * vi], q[3 * vi + 1], q[3 * vi + 2] };
		Eigen::Vector3f vj_pos{ q[3 * vj], q[3 * vj + 1], q[3 * vj + 2] };

		Eigen::Vector3f j2i = vi_pos - vj_pos;
		Eigen::Vector3f Achpc = j2i / j2i.norm() * rest_length;
		// float delta_x = j2i.norm() - rest_length; // This is the constraint set C_i (edge length)
		// Eigen::Vector3f j2i_normalized = j2i.normalized();

		// Eigen::Vector3f pc1 = vi_pos + 0.5f * delta_x * j2i_normalized;
		// Eigen::Vector3f pc2 = vj_pos - 0.5f * delta_x * j2i_normalized;
		// Eigen::Vector3f Acpc = pc1 - pc2;

		for (int i = 0; i < 3; i++)
		{
		#ifdef __CUDA_ARCH__
			atomicAdd(&b[3 * vi + i], wc * Achpc[i]);
			atomicAdd(&b[3 * vj + i], -wc * Achpc[i]);
		#else
			b[3 * vi + i] += wc * Achpc[i];
			b[3 * vj + i] += -wc * Achpc[i];
		#endif
		}
	}
}