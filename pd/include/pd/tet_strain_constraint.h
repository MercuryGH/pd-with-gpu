#pragma once

#include <pd/constraint.h>
#include <pd/types.h>

namespace pd
{
    class TetStrainConstraint: public Constraint
    {
    public:
		TetStrainConstraint() = default;

		TetStrainConstraint(float wc, const Positions& p, Eigen::RowVector4i vertices);
		TetStrainConstraint(float wc, const Positions& p, Eigen::RowVector4i vertices, Eigen::Vector3f min_strain_xyz, Eigen::Vector3f max_strain_xyz);

		void precompute_D_m_inv(const Positions& positions, float wc);

		Constraint* clone() const override
		{
			return new TetStrainConstraint(*this);
		}

		Eigen::VectorXf local_solve(const Eigen::VectorXf& q) const override;
		std::vector<Eigen::Triplet<float>> get_c_AcTAc(int n_vertex_offset) const override;

		__host__ __device__ void project_c_AcTAchpc(float* __restrict__ b, const float* __restrict__ q) const override;

		__host__ __device__ void print_name() const override
		{
			printf("TetStrainConstraint\n");
		}

		__host__ __device__ ~TetStrainConstraint() override {}

	public:
		// Eigen::SparseMatrix<float, Eigen::ColMajor> A_c;
		// Eigen::SparseMatrix<float, Eigen::RowMajor> A_c_transpose;

		Eigen::Vector3f min_strain_xyz{ Eigen::Vector3f::Ones() };
		Eigen::Vector3f max_strain_xyz{ Eigen::Vector3f::Ones() };
		Eigen::Matrix3f D_m_inv{ Eigen::Matrix3f::Identity() };
    };
}