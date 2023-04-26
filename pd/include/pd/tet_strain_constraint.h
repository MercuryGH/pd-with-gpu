#pragma once

#include <pd/constraint.h>
#include <pd/types.h>

#include <util/svd3_cuda.h>

namespace pd
{
    class TetStrainConstraint: public Constraint
    {
    public:
		TetStrainConstraint() = default;

		TetStrainConstraint(SimScalar wc, const PositionData& p, IndexRowVector4 vertices);
		TetStrainConstraint(SimScalar wc, const PositionData& p, IndexRowVector4 vertices, SimVector3 min_strain_xyz, SimVector3 max_strain_xyz);

		Constraint* clone() const override
		{
			return new TetStrainConstraint(*this);
		}

		std::vector<Eigen::Triplet<SimScalar>> get_c_AcTAc(int n_vertex_offset) const override;

		__host__ __device__ static SimScalar determinant3(const SimMatrix3& mat);
		__host__ __device__ static SimMatrix3 multiply3x3(const SimMatrix3& A, const SimMatrix3& B);
		__host__ __device__ static SimMatrix3 multiply_diagx3(const SimVector3& A, const SimMatrix3& B);
		__device__ static void gpu_svd3(const SimMatrix3& mat, SimMatrix3& U, SimVector3& sigma, SimMatrix3& V);

		__host__ __device__ void project_c_AcTAchpc(SimScalar* __restrict__ b, const SimScalar* __restrict__ q) const override;

		__host__ __device__ void print_name() const override
		{
			printf("TetStrainConstraint\n");
		}

		__host__ __device__ ~TetStrainConstraint() override {}

	private:
		void precompute_D_m_inv(const PositionData& positions);

		// Eigen::SparseMatrix<float, Eigen::ColMajor> A_c;
		// Eigen::SparseMatrix<float, Eigen::RowMajor> A_c_transpose;

		SimVector3 min_strain_xyz{ SimVector3::Ones() };
		SimVector3 max_strain_xyz{ SimVector3::Ones() };
		SimMatrix3 D_m_inv{ SimMatrix3::Identity() };
    };
}