#include <iostream>
#include <pd/bending_constraint.h>

namespace pd
{
    __host__ __device__ BendingConstraint::BendingConstraint(
		float wc, 
		int n_vertices, 
		float rest_mean_curvature,
		int* const vertices,
		float* const laplacian_weights
	): Constraint(wc, n_vertices), rest_mean_curvature(rest_mean_curvature)
	{
		this->vertices = vertices;
		this->laplacian_weights = laplacian_weights;
	}

	BendingConstraint::BendingConstraint(float wc, int center_vertex, const std::vector<int>& neighbor_vertices, const Positions& positions) : Constraint(wc, 1 + neighbor_vertices.size())
	{
		cudaMallocManaged(&vertices, sizeof(int) * n_vertices);

		vertices[0] = center_vertex;
		for (int i = 0; i < n_vertices - 1; i++)
		{
			vertices[i + 1] = neighbor_vertices[i];
		}

		precompute_laplacian_weights(neighbor_vertices, positions);
	}

	// perform deep copy
	BendingConstraint::BendingConstraint(const BendingConstraint& rhs): 
	Constraint(rhs), rest_mean_curvature(rhs.rest_mean_curvature)
	{
		realloc_laplacian_weights();
        memcpy(laplacian_weights, rhs.laplacian_weights, sizeof(float) * n_vertices);
	}

	BendingConstraint::BendingConstraint(BendingConstraint&& rhs) noexcept: 
	Constraint(rhs), rest_mean_curvature(rhs.rest_mean_curvature)
	{
		laplacian_weights = rhs.laplacian_weights;
		rhs.laplacian_weights = nullptr;
	}

	// perform deep copy
	BendingConstraint& BendingConstraint::operator=(const BendingConstraint& rhs)
	{
		// __super__
		Constraint::operator=(rhs);
		if (this != &rhs)
		{
			realloc_laplacian_weights();
			memcpy(laplacian_weights, rhs.laplacian_weights, sizeof(float) * n_vertices);
			rest_mean_curvature = rhs.rest_mean_curvature;
		}
		return *this;
	}

	BendingConstraint& BendingConstraint::operator=(BendingConstraint&& rhs) noexcept
	{
		// __super__
		Constraint::operator=(rhs);
		if (this != &rhs)
		{
			cudaFree(laplacian_weights);

			laplacian_weights = rhs.laplacian_weights;
			rhs.laplacian_weights = nullptr;

			rest_mean_curvature = rhs.rest_mean_curvature;
		}
		return *this;
	}

	void BendingConstraint::realloc_laplacian_weights()
	{
		cudaFree(laplacian_weights);
        cudaMallocManaged(&laplacian_weights, sizeof(float) * n_vertices);
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

		const int center_vertex = vertices[0];

		for (int i = 0; i < n_vertices; i++)
		{
			const int v = vertices[i];

			// discard terms between adjacent vertices since they are rather too small
			for (int j = 0; j < 3; j++)
			{
				float val = laplacian_weights[0] * laplacian_weights[i] * wc;
				// printf("%f, %f, %f\n", laplacian_weights[0], laplacian_weights[i], wc);
				// printf("%d-%d adds %f\n", center_vertex, v, val);
				triplets.emplace_back(
					3 * n_vertex_offset + 3 * center_vertex + j, 
					3 * n_vertex_offset + 3 * v + j, 
					val
				);
				if (v != center_vertex)
				{
					triplets.emplace_back(
						3 * n_vertex_offset + 3 * v + j, 
						3 * n_vertex_offset + 3 * center_vertex + j, 
						val
					);
				}
			}
		}

		return triplets;
	}

	__host__ void BendingConstraint::precompute_laplacian_weights(const std::vector<int>& neighbor_vertices, const Positions& positions)
	{
		const int neighbor_size = neighbor_vertices.size();
		assert(neighbor_size >= 3); // avoid singular vertex

		cudaMallocManaged(&laplacian_weights, sizeof(float) * n_vertices);

		laplacian_weights[0] = 0.0f; // init value

		const int center_vertex = vertices[0];

		const Eigen::Vector3d center_pos = positions.row(center_vertex).transpose();
		// traverse in counter-clockwise order
		for (int i = 0; i < neighbor_size; i++)
		{
			const int cur_pos_idx = neighbor_vertices[i];
			const int counter_clockwise_next_pos_idx = neighbor_vertices[(i + 1) % neighbor_size];
			const int clockwise_next_pos_idx = neighbor_vertices[(i + neighbor_size - 1) % neighbor_size];
			const Eigen::Vector3d cur_pos = positions.row(cur_pos_idx).transpose();
			const Eigen::Vector3d counter_clockwise_next_pos = positions.row(counter_clockwise_next_pos_idx).transpose();
			const Eigen::Vector3d clockwise_next_pos = positions.row(clockwise_next_pos_idx).transpose();

			if (is_collinear(cur_pos, center_pos, counter_clockwise_next_pos))
			{
				printf("Warning: Vertex %d, %d, %d are collinear, the triangulation of the mesh may be wrong!\n", cur_pos_idx, counter_clockwise_next_pos_idx, center_vertex);
				assert(false);
			}
			if (is_collinear(cur_pos, center_pos, clockwise_next_pos))
			{
				printf("Warning: Vertex %d, %d, %d are collinear, the triangulation of the mesh may be wrong!\n", cur_pos_idx, clockwise_next_pos_idx, center_vertex);
				assert(false);
			}

			const double dis = (cur_pos - center_pos).norm();

			const double tan_alpha = get_half_tan(cur_pos, center_pos, counter_clockwise_next_pos);
			const double tan_beta = get_half_tan(cur_pos, center_pos, clockwise_next_pos);

			const double coefficient = (tan_alpha + tan_beta) / dis;

			laplacian_weights[i + 1] = static_cast<float>(-coefficient);
			laplacian_weights[0] += static_cast<float>(coefficient);
		}

		// Debug only
		// for (int i = 0; i < neighbor_size; i++)
		// {
		// 	printf("vertices %d, weights = %f\n", vertices[0], laplacian_weights[i]);
		// }

		Eigen::Vector3f rest_mean_curvature_vector = apply_laplacian(positions).cast<float>();
		// std::cout << rest_mean_curvature_vector << "\n";
		rest_mean_curvature = rest_mean_curvature_vector.norm();
	}

	__host__ Eigen::Vector3d BendingConstraint::apply_laplacian(const Positions& positions) const
	{
		const int center_vertex = vertices[0];
		const Eigen::Vector3d center_pos = positions.row(center_vertex).transpose();

		Eigen::Vector3d ret;
		ret.setZero();
		for (int i = 0; i < n_vertices; i++)
		{
			const int cur_pos_idx = vertices[i];
			const Eigen::Vector3d cur_pos = positions.row(cur_pos_idx).transpose();
			ret += cur_pos * laplacian_weights[i];
		}

		return ret;
	}

	__host__ __device__ Eigen::Vector3f BendingConstraint::apply_laplacian(const float* __restrict__ q) const
	{
		Eigen::Vector3f ret;
		ret.setZero();

		const int center_vertex = vertices[0];
		const Eigen::Vector3f center_pos = { q[3 * center_vertex], q[3 * center_vertex + 1], q[3 * center_vertex + 2] };

		for (int i = 0; i < n_vertices; i++)
		{
			const int cur_pos_idx = vertices[i];
			const Eigen::Vector3f cur_pos = { q[3 * cur_pos_idx], q[3 * cur_pos_idx + 1], q[3 * cur_pos_idx + 2] };
			ret += cur_pos * laplacian_weights[i];
		}
		return ret;
	}

	__host__ __device__ void BendingConstraint::project_c_AcTAchpc(float* __restrict__ b, const float* __restrict__ q) const
	{
		const float EPS = 1e-5;
	
		if (std::abs(rest_mean_curvature) < EPS)
		{
			return; // no constraint indeed
		}

		Eigen::Vector3f deformed_laplace = apply_laplacian(q);
		const float deformed_laplace_norm = deformed_laplace.norm();
		// printf("Vertex %d, Rest mean curvature = %f, deformed_laplace = %f, %f, %f\n", vertices[0], rest_mean_curvature, deformed_laplace.x(), deformed_laplace.y(), deformed_laplace.z());

		Eigen::Vector3f Achpc;
		if (deformed_laplace_norm < EPS)
		// if (true)
		{
			// if norm is too small, don't divide by it, instead just use current normal
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
				atomicAdd(&b[3 * v + j], laplacian_weights[i] * Achpc[j] * wc);
			#else			
				b[3 * v + j] += laplacian_weights[i] * Achpc[j] * wc;
			#endif
			}
		}
	}

	__host__ __device__ Eigen::Vector3f BendingConstraint::get_center_vertex_normal(const float* __restrict__ q) const
	{
		const int neighbor_size = n_vertices - 1;

		const int center_vertex = vertices[0];

		const Eigen::Vector3f center_pos = { q[3 * center_vertex], q[3 * center_vertex + 1], q[3 * center_vertex + 2] };

		Eigen::Vector3f normal;
		normal.setZero();
		for (int i = 0; i < neighbor_size; i++)
		{
			const int cur_pos_idx = vertices[i + 1];
			const int counter_clockwise_next_pos_idx = vertices[((i + 1) % neighbor_size) + 1];
			const Eigen::Vector3f cur_pos = { q[3 * cur_pos_idx], q[3 * cur_pos_idx + 1], q[3 * cur_pos_idx + 2] };
			const Eigen::Vector3f counter_clockwise_next_pos = { q[3 * counter_clockwise_next_pos_idx], q[3 * counter_clockwise_next_pos_idx + 1], q[3 * counter_clockwise_next_pos_idx + 2] };

			normal += get_triangle_normal(counter_clockwise_next_pos - center_pos, cur_pos - center_pos);
		}
		normal.normalize(); // take the average
		normal = -normal;  // Invert it (not quite sure if this fits all circumstances)

		// #ifndef __CUDA_ARCH__
		// std::cout << normal << "\n";
		// #endif

		return normal;
	}

	__host__ __device__ Eigen::Vector3f BendingConstraint::get_triangle_normal(Eigen::Vector3f p21, Eigen::Vector3f p31)
	{
		return p21.cross(p31).normalized();
	}
}