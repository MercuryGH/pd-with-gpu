#pragma once

#include <pd/constraint.h>
#include <pd/types.h>

#include <Eigen/Geometry>

namespace pd
{
    class BendingConstraint: public Constraint
    {
    public:
		BendingConstraint() = default;

		/**
		 * The neighbor_vertices param must be given in sequential coutner-clockwise order.
		*/
		BendingConstraint(SimScalar wc, int center_vertex, const std::vector<VertexIndexType>& neighbor_vertices, const PositionData& q);

		BendingConstraint(const BendingConstraint& rhs);
		BendingConstraint(BendingConstraint&& rhs) noexcept;
		BendingConstraint& operator=(const BendingConstraint& rhs);
		BendingConstraint& operator=(BendingConstraint&& rhs) noexcept;

		Constraint* clone() const override
		{
			return new BendingConstraint(*this);
		}

		std::vector<Eigen::Triplet<SimScalar>> get_c_AcTAc(int n_vertex_offset) const override;

		__host__ __device__ void project_c_AcTAchpc(SimScalar* __restrict__ b, const SimScalar* __restrict__ q) const override;

		__host__ __device__ void print_name() const override
		{
			printf("BendingConstraint\n");
		}

		__host__ __device__ ~BendingConstraint() override
		{
			cudaFree(laplacian_weights);
		}

	private:
		void realloc_laplacian_weights();
		// precomputing using mean value formula
		__host__ void precompute_laplacian_weights(const std::vector<VertexIndexType>& neighbor_vertices, const PositionData& q);

		__host__ DataVector3 apply_laplacian(const PositionData& positions) const;
		__host__ __device__ SimVector3 apply_laplacian(const SimScalar* __restrict__ q) const;

		// determine if 3 points are in the same line
		template<typename T>
		static bool is_collinear(Eigen::Matrix<T, 3, 1> p1, Eigen::Matrix<T, 3, 1> p2, Eigen::Matrix<T, 3, 1> p3)
		{
			const T EPS = 1e-6;
			const T area = (p1 - p2).cross(p3 - p2).norm();
			return area < EPS;
		}

		// get cosine angle 123 (p2 is the center vertex position)
		template<typename T>
		static double get_cos(Eigen::Matrix<T, 3, 1> p12, Eigen::Matrix<T, 3, 1> p32)
		{
			return p12.dot(p32) / (p12.norm() * p32.norm());
		}

		// get sin angle 123
		template<typename T>
		static double get_sin(Eigen::Matrix<T, 3, 1> p12, Eigen::Matrix<T, 3, 1> p32)
		{
			return p12.cross(p32).norm() / (p12.norm() * p32.norm());
		}

		// tan(x / 2) = (1 - cos x) / sin x
		template<typename T>
		static double get_half_tan(Eigen::Matrix<T, 3, 1> p1, Eigen::Matrix<T, 3, 1> p2, Eigen::Matrix<T, 3, 1> p3)
		{
			const Eigen::Matrix<T, 3, 1> p12 = p1 - p2;
			const Eigen::Matrix<T, 3, 1> p32 = p3 - p2;
			return (1 - get_cos(p12, p32)) / get_sin(p12, p32);
		}

		__host__ __device__ SimVector3 get_center_vertex_normal(const SimScalar* __restrict__ q) const;

		// p1, p2 and p3 are given in counterclockwise order
		__host__ __device__ static SimVector3 get_triangle_normal(SimVector3 p21, SimVector3 p31);

		SimScalar* laplacian_weights{ nullptr };

		SimScalar rest_mean_curvature{ 0 };
    };
}