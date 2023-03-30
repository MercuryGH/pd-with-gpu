#pragma once

#include <pd/constraint.h>
#include <pd/types.h>

#include <Eigen/Geometry>

namespace pd
{
    class BendingConstraint: public Constraint
    {
    public:
		__host__ __device__ BendingConstraint(
			float wc, 
			int n_vertices, 
			int center_vertex, 
			const int* const neighbor_vertices,
			const float* const laplacian_weights
		);

		/**
		 * The neighbor_vertices param must be given in sequential clockwise order.
		*/
		BendingConstraint(float wc, int center_vertex, const std::vector<int>& neighbor_vertices, const Positions& q);

		Eigen::VectorXf local_solve(const Eigen::VectorXf& q) const override;
		std::vector<Eigen::Triplet<float>> get_c_AcTAc(int n_vertex_offset) const override;

		__host__ __device__ void project_c_AcTAchpc(float* __restrict__ b, const float* __restrict__ q) const override;

		__host__ __device__ void print_name() const override
		{
			printf("BendingConstraint\n");
		}

		__host__ __device__ ~BendingConstraint() override
		{
			delete[] laplacian_weights;
		}

	private:
		// precomputing using mean value formula
		__host__ void precompute_laplacian_weights(const std::vector<int>& neighbor_vertices, const Positions& q);

		__host__ __device__ Eigen::Vector3f apply_laplacian(Eigen::Vector3f pos, const float* __restrict__ q) const;

		// determine if 3 points are in the same line
		static bool is_collinear(Eigen::Vector3d p1, Eigen::Vector3d p2, Eigen::Vector3d p3)
		{
			const double EPS = 1e-6;
			const double area = (p1 - p2).cross(p3 - p2).norm();
			return area < EPS;
		}

		// get cosine angle 123 (p2 is the center vertex position)
		static double get_cos(Eigen::Vector3d p12, Eigen::Vector3d p32)
		{
			return p12.dot(p32) / (p12.norm() * p32.norm());
		}

		// get sin angle 123
		static double get_sin(Eigen::Vector3d p12, Eigen::Vector3d p32)
		{
			return p12.cross(p32).norm() / (p12.norm() * p32.norm());
		}

		// tan(x / 2) = (1 - cos x) / sin x
		static double get_half_tan(Eigen::Vector3d p1, Eigen::Vector3d p2, Eigen::Vector3d p3)
		{
			const Eigen::Vector3d p12 = p1 - p2;
			const Eigen::Vector3d p32 = p3 - p2;
			return (1 - get_cos(p12, p32)) / get_sin(p12, p32);
		}

		__host__ __device__ Eigen::Vector3f get_center_vertex_normal(const float* __restrict__ q) const;

		// p1, p2 and p3 are given in counterclockwise order
		__host__ __device__ static Eigen::Vector3f get_triangle_normal(Eigen::Vector3f p21, Eigen::Vector3f p31);

	public:
		int center_vertex;

		float* laplacian_weights;

		float rest_mean_curvature{ 0 };
    };
}