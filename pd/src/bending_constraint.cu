#include <pd/bending_constraint.h>

namespace pd
{
    __host__ __device__ BendingConstraint::BendingConstraint(float wc, int n): Constraint(wc, 0, nullptr)
	{

	}

	Eigen::VectorXf BendingConstraint::local_solve(const Eigen::VectorXf& q) const
	{
		Eigen::VectorXf ret;
		ret.resize(3);


		return ret;
	}

	std::vector<Eigen::Triplet<float>> BendingConstraint::get_c_AcTAc(int n_vertex_offset) const
	{
		std::vector<Eigen::Triplet<float>> triplets(12);

		// for (int i = 0; i < 3; i++)
		// {
		// 	triplets.emplace_back(3 * n_vertex_offset + 3 * vi + i, 3 * n_vertex_offset + 3 * vi + i, wc);
		// }
		// for (int i = 0; i < 3; i++)
		// {
		// 	triplets.emplace_back(3 * n_vertex_offset + 3 * vj + i, 3 * n_vertex_offset + 3 * vj + i, wc);
		// }
		// for (int i = 0; i < 3; i++)
		// {
		// 	triplets.emplace_back(3 * n_vertex_offset + 3 * vi + i, 3 * n_vertex_offset + 3 * vj + i, -wc);
		// }
		// for (int i = 0; i < 3; i++)
		// {
		// 	triplets.emplace_back(3 * n_vertex_offset + 3 * vj + i, 3 * n_vertex_offset + 3 * vi + i, -wc);
		// }

		return std::vector<Eigen::Triplet<float>>{ triplets.begin(), triplets.end() };
	}

	__host__ __device__ void BendingConstraint::project_c_AcTAchpc(float* __restrict__ b, const float* __restrict__ q) const
	{
		// #vertex offset is already included



        /*
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
        */
	}
}