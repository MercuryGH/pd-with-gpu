#include <pd/bending_constraint.h>

namespace pd
{
    __host__ __device__ BendingConstraint::BendingConstraint(
		float wc, 
		int n_vertices, 
		int center_vertex, 
		const int* const neighbor_vertices,
		const float* const laplacian_weights
	): Constraint(wc, n_vertices)
	{
		this->vertices = new int[n_vertices];
	}

	BendingConstraint::BendingConstraint(float wc, int center_vertex, const std::vector<int>& neighbor_vertices, const Positions& q) : Constraint(wc, 1 + neighbor_vertices.size())
	{
		this->vertices = new int[n_vertices];
		vertices[0] = center_vertex;
		for (int i = 0; i < n_vertices; i++)
		{
			vertices[i] = neighbor_vertices[i];
		}

		precompute_laplacian_weights(neighbor_vertices, q);
	}

	Eigen::VectorXf BendingConstraint::local_solve(const Eigen::VectorXf& q) const
	{
		Eigen::VectorXf ret;
		ret.resize(3);

		// for unit test only


		return ret;
	}

	std::vector<Eigen::Triplet<float>> BendingConstraint::get_c_AcTAc(int n_vertex_offset) const
	{
		std::vector<Eigen::Triplet<float>> triplets(3 * n_vertices - 1);

		for (int i = 0; i < n_vertices; i++)
		{
			const int v = vertices[i];

			// discard terms between adjacent vertices since they are rather too small
			for (int j = 0; j < 3; j++)
			{
				triplets.emplace_back(
					3 * n_vertex_offset + 3 * center_vertex + j, 
					3 * n_vertex_offset + 3 * v + j, 
					wc * laplacian_weights[i]
				);
				if (v != center_vertex)
				{
					triplets.emplace_back(
						3 * n_vertex_offset + 3 * v + j, 
						3 * n_vertex_offset + 3 * center_vertex + j, 
						wc * laplacian_weights[i]
					);
				}
			}
		}

		return std::vector<Eigen::Triplet<float>>{ triplets.begin(), triplets.end() };
	}

	__host__ void BendingConstraint::precompute_laplacian_weights(const std::vector<int>& neighbor_vertices, const Positions& q)
	{
		const int neighbor_size = neighbor_vertices.size();
		assert(neighbor_size >= 3); // avoid singular vertex

		laplacian_weights = new float[n_vertices];

		const Eigen::Vector3d center_pos = q.row(center_vertex).transpose();
		for (int i = 0; i < neighbor_size; i++)
		{
			const int cur_pos_idx = neighbor_vertices[i];
			const int clockwise_next_pos_idx = neighbor_vertices[(i + 1) % neighbor_size];
			const int counter_clockwise_next_pos_idx = neighbor_vertices[(i + neighbor_size - 1) % neighbor_size];
			const Eigen::Vector3d cur_pos = q.row(cur_pos_idx).transpose();
			const Eigen::Vector3d clockwise_next_pos = q.row(clockwise_next_pos_idx).transpose();
			const Eigen::Vector3d counter_clockwise_next_pos = q.row(counter_clockwise_next_pos_idx).transpose();

			if (is_collinear(cur_pos, center_pos, clockwise_next_pos))
			{
				printf("Warning: Vertex %d, %d, %d are collinear, the triangulation of the mesh may be wrong!\n", cur_pos_idx, clockwise_next_pos_idx, center_vertex);
			}
			if (is_collinear(cur_pos, center_pos, counter_clockwise_next_pos))
			{
				printf("Warning: Vertex %d, %d, %d are collinear, the triangulation of the mesh may be wrong!\n", cur_pos, counter_clockwise_next_pos_idx, center_vertex);
			}

			const double dis = (cur_pos - center_pos).norm();

			const double tan_alpha = get_half_tan(cur_pos, center_pos, clockwise_next_pos);
			const double tan_beta = get_half_tan(cur_pos, center_pos, counter_clockwise_next_pos);

			const double coefficient = (tan_alpha + tan_beta) / dis;

			laplacian_weights[i + 1] = static_cast<float>(-coefficient);
			laplacian_weights[0] += static_cast<float>(coefficient);
		}

		Eigen::Vector3f rest_mean_curvature_vector;
		rest_mean_curvature_vector.setZero();
		for (int i = 0; i < neighbor_size; i++)
		{
			const Eigen::Vector3f e = (q.row(center_vertex) - q.row(neighbor_vertices[i])).transpose().cast<float>();
			rest_mean_curvature_vector += e * laplacian_weights[i];
		}
		rest_mean_curvature = rest_mean_curvature_vector.norm();
	}

	__host__ __device__ Eigen::Vector3f BendingConstraint::apply_laplacian(Eigen::Vector3f pos, const float* __restrict__ q) const
	{
		const int neighbor_size = n_vertices - 1;
		
		Eigen::Vector3f ret;
		for (int i = 0; i < neighbor_size; i++)
		{
			const int cur_pos_idx = vertices[i + 1];
			const Eigen::Vector3f cur_pos = { q[3 * cur_pos_idx], q[3 * cur_pos_idx + 1], q[3 * cur_pos_idx + 2] };

			ret += (pos - cur_pos) * laplacian_weights[i];
		}
		return ret;
	}

	__host__ __device__ void BendingConstraint::project_c_AcTAchpc(float* __restrict__ b, const float* __restrict__ q) const
	{
		const float EPS = 1e-6;
		if (rest_mean_curvature < EPS)
		{
			return; // no constraint indeed
		}

		const Eigen::Vector3f center_pos = { q[3 * center_vertex], q[3 * center_vertex + 1], q[3 * center_vertex + 2] };
		
		Eigen::Vector3f deformed_laplace = apply_laplacian(center_pos, q);
		const float deformed_laplace_norm = deformed_laplace.norm();

		Eigen::Vector3f Achpc;
		if (deformed_laplace_norm < EPS)
		{
			// if norm too small, don't divide by it, instead just use current normal
			Achpc = get_center_vertex_normal(q) * rest_mean_curvature; // mean curvature vector
		}
		else 
		{
			Achpc = deformed_laplace * rest_mean_curvature / deformed_laplace_norm; // Ac'pc = v_f * |v_g| / |v_f|
		}

		for (int i = 0; i < n_vertices; i++)
		{
			const int v = vertices[i];
			for (int j = 0; j < 3; j++)
			{
			#ifdef __CUDA_ARCH__
				atomicAdd(&b[3 * v + j], wc * laplacian_weights[i] * Achpc[j]);
			#else			
				b[3 * v + j] += wc * laplacian_weights[i] * Achpc[j];
			#endif
			}
		}
	}

	__host__ __device__ Eigen::Vector3f BendingConstraint::get_center_vertex_normal(const float* __restrict__ q) const
	{
		const int neighbor_size = n_vertices - 1;

		const Eigen::Vector3f center_pos = { q[3 * center_vertex], q[3 * center_vertex + 1], q[3 * center_vertex + 2] };

		Eigen::Vector3f normal;
		normal.setZero();
		for (int i = 0; i < neighbor_size; i++)
		{
			const int cur_pos_idx = vertices[i + 1];
			const int clockwise_next_pos_idx = vertices[((i + 1) % neighbor_size) + 1];
			const Eigen::Vector3f cur_pos = { q[3 * cur_pos_idx], q[3 * cur_pos_idx + 1], q[3 * cur_pos_idx + 2] };
			const Eigen::Vector3f clockwise_next_pos = { q[3 * clockwise_next_pos_idx], q[3 * clockwise_next_pos_idx + 1], q[3 * clockwise_next_pos_idx + 2] };

			normal += get_triangle_normal(clockwise_next_pos - center_pos, cur_pos - center_pos);
		}
		normal.normalize(); // take the average

		return normal;
	}

	__host__ __device__ Eigen::Vector3f BendingConstraint::get_triangle_normal(Eigen::Vector3f p21, Eigen::Vector3f p31)
	{
		return p21.cross(p31).normalized();
	}
}