#include <array>
#include <pd/positional_constraint.h>

namespace pd {
	__host__ __device__ PositionalConstraint::PositionalConstraint(float wi, int vi, int n, float x0, float y0, float z0) :
		Constraint(wi, n),
		vi(vi),
		x0(x0), y0(y0), z0(z0)
	{
		n_vertices = 1;
		vertices = new int[n_vertices] {vi};
	}

	Eigen::VectorXf PositionalConstraint::local_solve(const Eigen::VectorXf& q) const
	{
		return Eigen::Vector3f(x0, y0, z0);
	}

	Eigen::VectorXf PositionalConstraint::get_c_AcTAchpc(const Eigen::VectorXf& pc) const
	{
		Eigen::VectorXf ret;
		ret.resize(3 * n);
		ret.setZero();

		// pc = p0 in positional constraint
		ret.block(3 * vi, 0, 3, 1) = wc * Eigen::Vector3f(x0, y0, z0);
		return ret;
	}

	std::vector<Eigen::Triplet<float>> PositionalConstraint::get_c_AcTAc(int n_vertex_offset) const
	{
		std::array<Eigen::Triplet<float>, 3u> triplets;
		for (int i = 0; i < 3; i++)
		{
			triplets[i] = { 3 * n_vertex_offset + 3 * vi + i, 3 * n_vertex_offset + 3 * vi + i, wc };
		}
		return std::vector<Eigen::Triplet<float>>{ triplets.begin(), triplets.end() };
	}

	__host__ __device__ void PositionalConstraint::project_c_AcTAchpc(float* __restrict__ b, const float* __restrict__ q) const
	{
#ifdef __CUDA_ARCH__
		atomicAdd(&b[3 * vi], wc * x0);
		atomicAdd(&b[3 * vi + 1], wc * y0);
		atomicAdd(&b[3 * vi + 2], wc * z0);
#else
		b[3 * vi] += wc * x0;
		b[3 * vi + 1] += wc * y0;
		b[3 * vi + 2] += wc * z0;
#endif
	}
}