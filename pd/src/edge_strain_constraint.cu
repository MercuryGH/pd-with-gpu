#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <vector>
#include <array>
#include <iostream>

#include <pd/edge_strain_constraint.h>

namespace pd {
	std::vector<Eigen::Triplet<SimScalar>> EdgeStrainConstraint::get_c_AcTAc(int n_vertex_offset) const
	{
		std::vector<Eigen::Triplet<SimScalar>> triplets(12);

		for (int i = 0; i < 3; i++)
		{
			triplets.emplace_back(3 * n_vertex_offset + 3 * vi() + i, 3 * n_vertex_offset + 3 * vi() + i, wc);
		}
		for (int i = 0; i < 3; i++)
		{
			triplets.emplace_back(3 * n_vertex_offset + 3 * vj() + i, 3 * n_vertex_offset + 3 * vj() + i, wc);
		}
		for (int i = 0; i < 3; i++)
		{
			triplets.emplace_back(3 * n_vertex_offset + 3 * vi() + i, 3 * n_vertex_offset + 3 * vj() + i, -wc);
		}
		for (int i = 0; i < 3; i++)
		{
			triplets.emplace_back(3 * n_vertex_offset + 3 * vj() + i, 3 * n_vertex_offset + 3 * vi() + i, -wc);
		}

		return triplets;
	}

	__host__ __device__ void EdgeStrainConstraint::project_c_AcTAchpc(SimScalar* __restrict__ b, const SimScalar* __restrict__ q) const
	{
		// #vertex offset is not included
		SimVector3 vi_pos{ q[3 * vi()], q[3 * vi() + 1], q[3 * vi() + 2] };
		SimVector3 vj_pos{ q[3 * vj()], q[3 * vj() + 1], q[3 * vj() + 2] };

		SimVector3 j2i = vi_pos - vj_pos;
		SimVector3 Achpc = j2i / j2i.norm() * rest_length;

		for (int i = 0; i < 3; i++)
		{
		#ifdef __CUDA_ARCH__
			atomicAdd(&b[3 * vi() + i], wc * Achpc[i]);
			atomicAdd(&b[3 * vj() + i], -wc * Achpc[i]);
		#else
			b[3 * vi() + i] += wc * Achpc[i];
			b[3 * vj() + i] += -wc * Achpc[i];
		#endif
		}
	}
}