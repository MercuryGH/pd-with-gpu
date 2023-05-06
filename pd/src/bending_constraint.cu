#include <iostream>
#include <pd/bending_constraint.h>

namespace pd
{
	BendingConstraint::BendingConstraint(SimScalar wc, int center_vertex, const std::vector<VertexIndexType>& neighbor_vertices, const PositionData& positions, bool discard_quadratic_term) : Constraint(wc, 1 + neighbor_vertices.size()), discard_quadratic_term(discard_quadratic_term)
	{
		cudaMallocManaged(&vertices, sizeof(VertexIndexType) * n_vertices);

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
        memcpy(laplacian_weights, rhs.laplacian_weights, sizeof(SimScalar) * n_vertices);
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
			memcpy(laplacian_weights, rhs.laplacian_weights, sizeof(SimScalar) * n_vertices);
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
        cudaMallocManaged(&laplacian_weights, sizeof(SimScalar) * n_vertices);
	}

	std::vector<Eigen::Triplet<SimScalar>> BendingConstraint::get_c_AcTAc(int n_vertex_offset) const
	{
		std::vector<Eigen::Triplet<SimScalar>> triplets(3 * n_vertices - 1);

		VertexIndexType local_max_vertex_idx = 0;
		for (int i = 0; i < n_vertices; i++)
		{
			local_max_vertex_idx = std::max(local_max_vertex_idx, vertices[i]);
		}

		Eigen::SparseMatrix<SimScalar> A_c(1, local_max_vertex_idx + 1); // preallocate space is necessary
		A_c.setZero();

		for (int i = 0; i < n_vertices; i++)
		{
			const VertexIndexType v = vertices[i];
			A_c.insert(0, v) = laplacian_weights[i];
		}

		A_c.makeCompressed();

		Eigen::SparseMatrix<SimScalar> AcT_Ac = A_c.transpose() * A_c;
		AcT_Ac.makeCompressed();
		for (int i = 0; i < AcT_Ac.outerSize(); i++)
		{
			for (Eigen::SparseMatrix<SimScalar>::InnerIterator itr(AcT_Ac, i); itr; ++itr)
			{
				for (int j = 0; j < 3; j++)
				{
					// discard quadratic term
					if (discard_quadratic_term == true && itr.row() != vertices[0] && itr.col() != vertices[0])
						continue;
						
					triplets.emplace_back(3 * n_vertex_offset + 3 * itr.row() + j, 3 * n_vertex_offset + 3 * itr.col() + j, wc * itr.value());
				}
			}
		}

		return triplets;
	}

	__host__ void BendingConstraint::precompute_laplacian_weights(const std::vector<VertexIndexType>& neighbor_vertices, const PositionData& positions)
	{
		// TODO: problematic, should minus center of mass rather than use global coordinate
		const int neighbor_size = neighbor_vertices.size();
		assert(neighbor_size >= 3); // avoid singular vertex

		cudaMallocManaged(&laplacian_weights, sizeof(SimScalar) * n_vertices);

		laplacian_weights[0] = 0.0f; // init value

		const VertexIndexType center_vertex = vertices[0];

		const DataVector3 center_pos = positions.row(center_vertex).transpose();
		// traverse in counter-clockwise order
		for (int i = 0; i < neighbor_size; i++)
		{
			const VertexIndexType cur_pos_idx = neighbor_vertices[i];
			const VertexIndexType counter_clockwise_next_pos_idx = neighbor_vertices[(i + 1) % neighbor_size];
			const VertexIndexType clockwise_next_pos_idx = neighbor_vertices[(i + neighbor_size - 1) % neighbor_size];
			const DataVector3 cur_pos = positions.row(cur_pos_idx).transpose();
			const DataVector3 counter_clockwise_next_pos = positions.row(counter_clockwise_next_pos_idx).transpose();
			const DataVector3 clockwise_next_pos = positions.row(clockwise_next_pos_idx).transpose();

			bool cc_collinear_flag = false;
			if (is_collinear(cur_pos, center_pos, counter_clockwise_next_pos))
			{
				printf("Warning: Vertex %d, %d, %d are collinear, the triangulation of the mesh may be wrong!\n", cur_pos_idx, counter_clockwise_next_pos_idx, center_vertex);
				cc_collinear_flag = true;
			}
			bool c_collinear_flag = false;
			if (is_collinear(cur_pos, center_pos, clockwise_next_pos))
			{
				printf("Warning: Vertex %d, %d, %d are collinear, the triangulation of the mesh may be wrong!\n", cur_pos_idx, clockwise_next_pos_idx, center_vertex);
				c_collinear_flag = true;
			}

			const DataScalar dis = (cur_pos - center_pos).norm();

			const DataScalar tan_alpha = cc_collinear_flag ? 0 : get_half_tan(cur_pos, center_pos, counter_clockwise_next_pos);
			const DataScalar tan_beta = c_collinear_flag ? 0 : get_half_tan(cur_pos, center_pos, clockwise_next_pos);

			const DataScalar coefficient = (tan_alpha + tan_beta) / dis;

			laplacian_weights[i + 1] = static_cast<SimScalar>(-coefficient);
			laplacian_weights[0] += static_cast<SimScalar>(coefficient);
		}

		SimVector3 rest_mean_curvature_vector = apply_laplacian(positions).cast<SimScalar>();
		// std::cout << rest_mean_curvature_vector << "\n";
		rest_mean_curvature = rest_mean_curvature_vector.norm();

		// Debug
		// if (center_vertex == 412)
		// {
		// 	std::cout << "rmc = " << rest_mean_curvature << "\n";
		// 	std::cout << "lw0 = " << laplacian_weights[0] << "\n";
		// 	for (int i = 1; i < neighbor_size; i++)
		// 	{
		// 		printf("vertices %d, weights = %f\n", vertices[0], laplacian_weights[i]);
		// 	}
		// }
	}

	__host__ DataVector3 BendingConstraint::apply_laplacian(const PositionData& positions) const
	{
		const VertexIndexType center_vertex = vertices[0];
		const DataVector3 center_pos = positions.row(center_vertex).transpose();

		DataVector3 ret;
		ret.setZero();
		// This causes huge numerical error when vertex is far away from origin
		// for (int i = 0; i < n_vertices; i++)
		// {
		// 	const VertexIndexType cur_pos_idx = vertices[i];
		// 	const DataVector3 cur_pos = positions.row(cur_pos_idx).transpose();
		// 	ret += cur_pos * laplacian_weights[i];
		// }

		// This computes extra E times subtraction but is stable in numerics
		for (int i = 1; i < n_vertices; i++)
		{
			const VertexIndexType cur_pos_idx = vertices[i];
			const DataVector3 cur_pos = positions.row(cur_pos_idx).transpose();			
			ret += (center_pos - cur_pos) * laplacian_weights[i];
		}

		return ret;
	}

	__host__ __device__ SimVector3 BendingConstraint::apply_laplacian(const SimScalar* __restrict__ q) const
	{
		SimVector3 ret;
		ret.setZero();

		const VertexIndexType center_vertex = vertices[0];
		const SimVector3 center_pos = { q[3 * center_vertex], q[3 * center_vertex + 1], q[3 * center_vertex + 2] };

		for (int i = 1; i < n_vertices; i++)
		{
			const VertexIndexType cur_pos_idx = vertices[i];
			const SimVector3 cur_pos = { q[3 * cur_pos_idx], q[3 * cur_pos_idx + 1], q[3 * cur_pos_idx + 2] };
			ret += (center_pos - cur_pos) * laplacian_weights[i];
		}
		return ret;
	}

	__host__ __device__ void BendingConstraint::project_c_AcTAchpc(SimScalar* __restrict__ b, const SimScalar* __restrict__ q) const
	{
		const SimScalar EPS = 1e-5;
	
		if (std::abs(rest_mean_curvature) < EPS)
		{
			return; // no constraint indeed
		}

		SimVector3 deformed_laplace = apply_laplacian(q);
		const SimScalar deformed_laplace_norm = deformed_laplace.norm();
		// printf("Vertex %d, Rest mean curvature = %f, deformed_laplace = %f, %f, %f\n", vertices[0], rest_mean_curvature, deformed_laplace.x(), deformed_laplace.y(), deformed_laplace.z());

		SimVector3 Achpc;
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
			const VertexIndexType v = vertices[i];
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

	__host__ __device__ SimVector3 BendingConstraint::get_center_vertex_normal(const SimScalar* __restrict__ q) const
	{
		const int neighbor_size = n_vertices - 1;

		const int center_vertex = vertices[0];

		const SimVector3 center_pos = { q[3 * center_vertex], q[3 * center_vertex + 1], q[3 * center_vertex + 2] };

		SimVector3 normal;
		normal.setZero();
		for (int i = 0; i < neighbor_size; i++)
		{
			const VertexIndexType cur_pos_idx = vertices[i + 1];
			const VertexIndexType counter_clockwise_next_pos_idx = vertices[((i + 1) % neighbor_size) + 1];
			const SimVector3 cur_pos = { q[3 * cur_pos_idx], q[3 * cur_pos_idx + 1], q[3 * cur_pos_idx + 2] };
			const SimVector3 counter_clockwise_next_pos = { q[3 * counter_clockwise_next_pos_idx], q[3 * counter_clockwise_next_pos_idx + 1], q[3 * counter_clockwise_next_pos_idx + 2] };

			normal += get_triangle_normal(counter_clockwise_next_pos - center_pos, cur_pos - center_pos);
		}
		normal.normalize(); // take the average
		normal = -normal;  // Invert it (not quite sure if this fits all circumstances)

		// #ifndef __CUDA_ARCH__
		// std::cout << normal << "\n";
		// #endif

		return normal;
	}

	__host__ __device__ SimVector3 BendingConstraint::get_triangle_normal(SimVector3 p21, SimVector3 p31)
	{
		return p21.cross(p31).normalized();
	}
}