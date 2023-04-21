#include <array>
#include <pd/positional_constraint.h>

namespace pd {
	std::vector<Eigen::Triplet<SimScalar>> PositionalConstraint::get_c_AcTAc(int n_vertex_offset) const
	{
		std::vector<Eigen::Triplet<SimScalar>> triplets(3);
		for (int i = 0; i < 3; i++)
		{
			triplets.emplace_back(3 * n_vertex_offset + 3 * vi() + i, 3 * n_vertex_offset + 3 * vi() + i, wc);
		}
		return triplets;
	}

	__host__ __device__ void PositionalConstraint::project_c_AcTAchpc(SimScalar* __restrict__ b, const SimScalar* __restrict__ q) const
	{
#ifdef __CUDA_ARCH__
		atomicAdd(&b[3 * vi()], wc * p0.x());
		atomicAdd(&b[3 * vi() + 1], wc * p0.y());
		atomicAdd(&b[3 * vi() + 2], wc * p0.z());
#else
		b[3 * vi()] += wc * p0.x();
		b[3 * vi() + 1] += wc * p0.y();
		b[3 * vi() + 2] += wc * p0.z();
#endif
	}
}