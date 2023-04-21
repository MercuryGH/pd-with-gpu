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

		// v[0] as pivot
		// const Eigen::Vector3d pivot_vertex_pos = positions.row(vertices[0]).transpose();
		// for (int i = 0; i < 3; i++)
		// {
		// 	const int v = vertices[i + 1];
		// 	const Eigen::Vector3d cur_vertex_pos = positions.row(v).transpose();
		// 	D_m.col(i) = cur_vertex_pos - pivot_vertex_pos;
		// }

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
				// v[0] as pivot
				// if (i == 0)
				// {
				// 	A_c.insert(j, v) = -(D_m_inv.coeff(0, j) + D_m_inv.coeff(1, j) + D_m_inv.coeff(2, j));
				// }
				// else
				// {
				// 	A_c.insert(j, v) = D_m_inv.coeff(i - 1, j);
				// }

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
		// for (int i = 0; i < n_vertices; i++)
		// {
		// 	std::cout << vertices[i] << "\n";
		// }

		Eigen::SparseMatrix<SimScalar> AcT_Ac = wc * A_c.transpose() * A_c;
		AcT_Ac.makeCompressed();
		for (int i = 0; i < AcT_Ac.outerSize(); i++)
		{
			for (Eigen::SparseMatrix<SimScalar>::InnerIterator itr(AcT_Ac, i); itr; ++itr)
			{
				for (int j = 0; j < 3; j++)
				{
					triplets.emplace_back(3 * n_vertex_offset + 3 * itr.row() + j, 3 * n_vertex_offset + 3 * itr.col() + j, itr.value());
				}
			}
		}

		// for (int i = 0; i < triplets.size(); i++)
		// {
		// 	printf("%d, %d, %f\n", triplets[i].row(), triplets[i].col(), triplets[i].value());
		// }
		// printf("triplets.size() = %d\n", triplets.size());
		// assert(false);

		return triplets;
	}

	__host__ __device__ void TetStrainConstraint::test_local_step_tet_strain(SimScalar* __restrict__ b, const SimScalar* __restrict__ q) const
	{
		#ifdef __CUDA_ARCH__
			return;
		#else

		SimVector3 cur_pos[4];
		for (int i = 0; i < 4; i++)
		{
			SimScalar x = (SimScalar)q[3 * vertices[i]];
			SimScalar y = (SimScalar)q[3 * vertices[i] + 1];
			SimScalar z = (SimScalar)q[3 * vertices[i] + 2];

			cur_pos[i] = { x, y, z };
		}
		// v[3] as pivot
		SimMatrix3 D_s;
		for (int i = 0; i < 3; i++)
		{
			D_s.col(i) = cur_pos[i] - cur_pos[3];
		}

		SimMatrix3 D_m_inv = this->D_m_inv.cast<SimScalar>();

		SimMatrix3 F = D_s * D_m_inv; // deformation gradient

		if (vertices[0] == 0)
		{
			std::cout << "D_s = " << D_s << "\n";
		}
		// std::cout << "D_m_inv = " << D_m_inv << "\n";
		// std::cout << "deformation gradient = " << F << "\n";

		SimMatrix3 Achpc;
		const bool tet_inverted = F.determinant() < 0;
		Eigen::JacobiSVD<SimMatrix3> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
		SimMatrix3 U = svd.matrixU();
		SimMatrix3 V = svd.matrixV();
		SimVector3 sigma = svd.singularValues();
		for (int i = 0; i < 3; i++)
		{
			sigma(i) = std::clamp(sigma(i), (SimScalar)min_strain_xyz(i), (SimScalar)max_strain_xyz(i));
		}

		// This code is necessary to preserve the orientation of the tetrahedron
		if (tet_inverted)
		{
			sigma(2) = -sigma(2);
		}

		Achpc = U * sigma.asDiagonal() * V.transpose();
		// static int cnt = 0;
		// if (vertices[0] == 0)
		// {
		// 	std::cout << "Achpc = " << Achpc << "\n";
		// 	cnt++;
		// }
		// if (cnt == 100) while (1);

		// apply A_c^T
		for (int i = 0; i < 4; i++)
		{
			const int v = vertices[i];
			
			SimRowVector3 sum_of_products;
			sum_of_products.setZero();
			for (int j = 0; j < 3; j++)
			{
				// v3 as pivot vertex
				if (i == 3)
				{
					SimRowVector3 neg_D_m_inv = -(D_m_inv.row(0) + D_m_inv.row(1) + D_m_inv.row(2));
					sum_of_products += Achpc.row(j) * neg_D_m_inv(j);
				}
				else
				{
					sum_of_products += Achpc.row(j) * D_m_inv.coeff(i, j);
				}
			}

			for (int j = 0; j < 3; j++)
			{
				// static int cnt = 0;
				// if (v == 0 && j == 1)
				// {
				// 	std::cout << sum_of_products[j] * wc + b[1] << " = " << b[1] << " + "<< sum_of_products[j] << " * " << wc << "\n";
				// 	cnt++;
				// }
				// if (cnt == 300)
				// {
				// 	while (1);
				// }

				b[3 * v + j] += sum_of_products[j] * wc;
				// std::cout << "write val = " << sum_of_products[j] << ", " << wc << "\n";
			}
		}
		#endif	
	}

	__host__ __device__ void TetStrainConstraint::project_c_AcTAchpc(SimScalar* __restrict__ b, const SimScalar* __restrict__ q) const
	{
		test_local_step_tet_strain(b, q);
	// 	return;
	// 	#ifndef __CUDA_ARCH__
	// 	// for (int i = 0; i < 3 * 8; i += 3)
	// 	// {
	// 	// 	printf("vertex %d pos = %f %f %f\n", i / 3, q[i], q[i + 1], q[i + 2]);
	// 	// }
	// 	#endif			

	// 	Eigen::Vector3f cur_pos[4];
	// 	for (int i = 0; i < 4; i++)
	// 	{
	// 		cur_pos[i] = { q[3 * vertices[i]], q[3 * vertices[i] + 1], q[3 * vertices[i] + 2] };
	// 	#ifndef __CUDA_ARCH__
	// 		// std::cout << "cur_pos[i] = " << cur_pos[i] << "\n";
	// 	#endif			
	// 	}

	// 	// v[0] as pivot
	// 	// Eigen::Matrix3f D_s;
	// 	// for (int i = 0; i < 3; i++)
	// 	// {
	// 	// 	D_s.col(i) = cur_pos[i + 1] - cur_pos[0];
	// 	// }

	// 	// v[3] as pivot
	// 	Eigen::Matrix3f D_s;
	// 	for (int i = 0; i < 3; i++)
	// 	{
	// 		D_s.col(i) = cur_pos[i] - cur_pos[3];
	// 	}


	// 	Eigen::Matrix3f F = D_s * D_m_inv; // deformation gradient

	// #ifndef __CUDA_ARCH__
	// 	// std::cout << "D_s = " << D_s << "\n";
	// 	// std::cout << "D_m_inv = " << D_m_inv << "\n";
	// 	// std::cout << "deformation gradient = " << F << "\n";
	// #endif

	// 	Eigen::Matrix3f Achpc;
	// #ifdef __CUDA_ARCH__

	// #else
	// 	const bool tet_inverted = F.determinant() < 0;
	// 	Eigen::JacobiSVD<Eigen::Matrix3f> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
	// 	Eigen::Matrix3f U = svd.matrixU();
	// 	Eigen::Matrix3f V = svd.matrixV();
	// 	Eigen::Vector3f sigma = svd.singularValues();
	// 	for (int i = 0; i < 3; i++)
	// 	{
	// 		sigma(i) = std::clamp(sigma(i), min_strain_xyz(i), max_strain_xyz(i));
	// 	}

	// 	// This code is necessary to preserve the orientation of the tetrahedron
	// 	if (tet_inverted)
	// 	{
	// 		sigma(2) = -sigma(2);
	// 	}

	// 	Achpc = U * sigma.asDiagonal() * V.transpose();
	// 	// static int cnt = 0;
	// 	// if (vertices[0] == 0)
	// 	// {
	// 	// 	std::cout << "Achpc = " << Achpc << "\n";
	// 	// 	cnt++;
	// 	// }
	// 	// if (cnt == 100) while (1);
	// #endif

	// 	// apply A_c^T
	// 	for (int i = 0; i < 4; i++)
	// 	{
	// 		const int v = vertices[i];
			
	// 		Eigen::RowVector3f sum_of_products;
	// 		sum_of_products.setZero();
	// 		for (int j = 0; j < 3; j++)
	// 		{
	// 			// v0 as pivot vertex
	// 			// if (i == 0)
	// 			// {
	// 			// 	Eigen::RowVector3f neg_D_m_inv = -(D_m_inv.row(0) + D_m_inv.row(1) + D_m_inv.row(2));
	// 			// 	sum_of_products += Achpc.row(j) * neg_D_m_inv(j);
	// 			// }
	// 			// else
	// 			// {
	// 			// 	sum_of_products += Achpc.row(j) * D_m_inv.coeff(i - 1, j);
	// 			// }

	// 			// v3 as pivot vertex
	// 			if (i == 3)
	// 			{
	// 				Eigen::RowVector3f neg_D_m_inv = -(D_m_inv.row(0) + D_m_inv.row(1) + D_m_inv.row(2));
	// 				sum_of_products += Achpc.row(j) * neg_D_m_inv(j);
	// 			}
	// 			else
	// 			{
	// 				sum_of_products += Achpc.row(j) * D_m_inv.coeff(i, j);
	// 			}
	// 		}

	// 		for (int j = 0; j < 3; j++)
	// 		{
	// 		#ifdef __CUDA_ARCH__
	// 			atomicAdd(&b[3 * v + j], sum_of_products[j] * wc);
	// 		#else			
	// 			// static int cnt = 0;
	// 			// if (v == 0 && j == 1)
	// 			// {
	// 			// 	std::cout << sum_of_products[j] * wc + b[1] << " = " << b[1] << " + "<< sum_of_products[j] << " * " << wc << "\n";
	// 			// 	cnt++;
	// 			// }
	// 			// if (cnt == 300)
	// 			// {
	// 			// 	while (1);
	// 			// }

	// 			b[3 * v + j] += sum_of_products[j] * wc;
	// 			// std::cout << "write val = " << sum_of_products[j] << ", " << wc << "\n";
	// 		#endif
	// 		}
	// 	}
	}
}