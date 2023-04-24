#include <iostream>

#include <pd/tet_strain_constraint.h>

#include <Eigen/SVD>
#include <Eigen/Geometry>

namespace pd
{
	TetStrainConstraint::TetStrainConstraint(SimScalar wc, const PositionData& p, IndexRowVector4 vertices): 
		Constraint(wc, vertices.size())
	{
		assert(n_vertices == 4);
		cudaMallocManaged(&this->vertices, sizeof(VertexIndexType) * 4);
		for (int i = 0; i < 4; i++)
		{
			this->vertices[i] = vertices[i];
		}

		precompute_D_m_inv(p);
	}

	TetStrainConstraint::TetStrainConstraint(SimScalar wc, const PositionData& p, IndexRowVector4 vertices, SimVector3 min_strain_xyz, SimVector3 max_strain_xyz):
		TetStrainConstraint(wc, p, vertices)
	{
		this->min_strain_xyz = min_strain_xyz;
		this->max_strain_xyz = max_strain_xyz;
	}

	void TetStrainConstraint::precompute_D_m_inv(const PositionData& positions)
	{
		DataMatrix3 D_m;

		// v[3] as pivot
		const DataVector3 pivot_vertex_pos = positions.row(vertices[3]).transpose();
		for (int i = 0; i < 3; i++)
		{
			const VertexIndexType v = vertices[i];
			const DataVector3 cur_vertex_pos = positions.row(v).transpose();
			D_m.col(i) = cur_vertex_pos - pivot_vertex_pos;
		}		

		// set weight related to rest volume
		// bug1: wc did not multiplied
		const SimScalar volume = static_cast<SimScalar>(std::abs(D_m.determinant()) / 6);
		wc *= volume;

		constexpr SimScalar EPS = 1e-5;
		if (volume < EPS)
		{
			printf("Tetrahedron %d, %d, %d, %d is too small, the tetrahedralization may be wrong!\n", vertices[0], vertices[1], vertices[2], vertices[3]);
			assert(false);
		}

		// precompute D_m_inv
		D_m_inv = D_m.inverse().cast<SimScalar>();
	}

	std::vector<Eigen::Triplet<SimScalar>> TetStrainConstraint::get_c_AcTAc(int n_vertex_offset) const
	{
		std::vector<Eigen::Triplet<SimScalar>> triplets(3 * 4 * 4); // 48

		VertexIndexType local_max_vertex_idx = 0;
		for (int i = 0; i < n_vertices; i++)
		{
			local_max_vertex_idx = std::max(local_max_vertex_idx, vertices[i]);
		}

		Eigen::SparseMatrix<SimScalar> A_c(3, local_max_vertex_idx + 1); // preallocate space is necessary
		A_c.setZero();

		for (int i = 0; i < 4; i++)
		{
			const VertexIndexType v = vertices[i];
			for (int j = 0; j < 3; j++)
			{
				// v[3] as pivot
				if (i == 3)
				{
					A_c.insert(j, v) = -(D_m_inv.coeff(0, j) + D_m_inv.coeff(1, j) + D_m_inv.coeff(2, j));
				}
				else
				{
					A_c.insert(j, v) = D_m_inv.coeff(i, j);
				}
			}
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
					triplets.emplace_back(3 * n_vertex_offset + 3 * itr.row() + j, 3 * n_vertex_offset + 3 * itr.col() + j, wc * itr.value());
				}
			}
		}

		return triplets;
	}

	__host__ __device__ void TetStrainConstraint::project_c_AcTAchpc(SimScalar* __restrict__ b, const SimScalar* __restrict__ q) const
	{
		SimVector3 cur_pos[4];
		for (int i = 0; i < 4; i++)
		{
			const VertexIndexType v = vertices[i];

			SimScalar x = (SimScalar)q[3 * v];
			SimScalar y = (SimScalar)q[3 * v + 1];
			SimScalar z = (SimScalar)q[3 * v + 2];

			cur_pos[i] = { x, y, z };
		}
		// v[3] as pivot
		SimMatrix3 D_s;
		for (int i = 0; i < 3; i++)
		{
			D_s.col(i) = cur_pos[i] - cur_pos[3];
		}

		const SimMatrix3 F = D_s * D_m_inv; // deformation gradient

	#ifdef __CUDA_ARCH__
		// GPU side determinant
		bool tet_inverted = false;

		// GPU side SVD
		SimMatrix3 U;
		SimMatrix3 V;
		SimVector3 sigma;
	#else
		const bool tet_inverted = F.determinant() < 0;

		// CPU side SVD
		Eigen::JacobiSVD<SimMatrix3> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
		const SimMatrix3& U = svd.matrixU();
		const SimMatrix3& V = svd.matrixV();
		SimVector3 sigma = svd.singularValues();
	#endif
		for (int i = 0; i < 3; i++)
		{
			sigma(i) = std::clamp(sigma(i), (SimScalar)min_strain_xyz(i), (SimScalar)max_strain_xyz(i));
		}

		// This code is necessary to preserve the orientation of the tetrahedron
		if (tet_inverted)
		{
			sigma(2) = -sigma(2);
		}

		// U * sigma * V^T is NEARLY the rotation part of deformation gradient assumes simga(i) is clamp to nearly 1
		// Thus we can transpose it to get the inversion of this matrix so as to restore the deformation
		SimMatrix3 Achpc = V * sigma.asDiagonal() * U.transpose(); // equivalent to (U * sigma * V^T)^{-1}
		// Note: if we don't transpose U * sigma * V^T, the simulation very become stiff and weird

		#ifndef __CUDA_ARCH__
			std::cout << "F = \n" << F << "\n";
			std::cout << "Achpc = \n" << Achpc << "\n";
		#endif

		// apply A_c^T
		for (int i = 0; i < 4; i++)
		{
			const VertexIndexType v = vertices[i];

			const SimRowVector3 neg_D_m_inv = -(D_m_inv.row(0) + D_m_inv.row(1) + D_m_inv.row(2));
			
			SimRowVector3 sum_of_products;
			sum_of_products.setZero();
			for (int j = 0; j < 3; j++)
			{
				// v3 as pivot vertex
				if (i == 3)
				{
					sum_of_products += Achpc.row(j) * neg_D_m_inv(j);
				}
				else
				{
					sum_of_products += Achpc.row(j) * D_m_inv.coeff(i, j);
				}
			}

			for (int j = 0; j < 3; j++)
			{
			#ifdef __CUDA_ARCH__
				atomicAdd(&b[3 * v + j], sum_of_products[j] * wc);
			#else
				b[3 * v + j] += sum_of_products[j] * wc;
			#endif
			}
		}
	}
}