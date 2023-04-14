#include <array>
#include <pd/positional_constraint.h>

namespace pd {
	// __host__ __device__ PositionalConstraint::PositionalConstraint(float wc, int vi, float x0, float y0, float z0) :
	// 	Constraint(wc, 1, new int[1] {vi}),
	// 	vi(vi),
	// 	x0(x0), y0(y0), z0(z0)
	// {
	// }

	Eigen::VectorXf PositionalConstraint::local_solve(const Eigen::VectorXf& q) const
	{
		return p0;
	}

	std::vector<Eigen::Triplet<float>> PositionalConstraint::get_c_AcTAc(int n_vertex_offset) const
	{
		std::vector<Eigen::Triplet<float>> triplets(3);
		for (int i = 0; i < 3; i++)
		{
			triplets.emplace_back(3 * n_vertex_offset + 3 * vi() + i, 3 * n_vertex_offset + 3 * vi() + i, wc);
		}
		return triplets;
	}

	__host__ __device__ void PositionalConstraint::project_c_AcTAchpc(float* __restrict__ b, const float* __restrict__ q) const
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