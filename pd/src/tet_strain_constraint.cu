#include <iostream>

#include <pd/tet_strain_constraint.h>

#include <Eigen/SVD>
#include <Eigen/Geometry>

namespace pd
{
	Eigen::VectorXf TetStrainConstraint::local_solve(const Eigen::VectorXf& q) const
	{
		Eigen::VectorXf ret;
		ret.resize(3);


		return ret;
	}

	TetStrainConstraint::TetStrainConstraint(float wc, const Positions& p, Eigen::RowVector4i vertices) : 
		Constraint(wc, vertices.size())
	{
		assert(n_vertices == 4);
		cudaMallocManaged(&this->vertices, sizeof(int) * 4);
		for (int i = 0; i < 4; i++)
		{
			this->vertices[i] = vertices[i];
		}

		precompute_D_m_inv(p, wc);
	}

	TetStrainConstraint::TetStrainConstraint(float wc, const Positions& p, Eigen::RowVector4i vertices, Eigen::Vector3f min_strain_xyz, Eigen::Vector3f max_strain_xyz):
		TetStrainConstraint(wc, p, vertices)
	{
		this->min_strain_xyz = min_strain_xyz;
		this->max_strain_xyz = max_strain_xyz;
	}

	void TetStrainConstraint::precompute_D_m_inv(const Positions& positions, float wc)
	{
		Eigen::Matrix3d D_m;
		const Eigen::Vector3d pivot_vertex_pos = positions.row(vertices[0]).transpose();
		for (int i = 0; i < 3; i++)
		{
			const int v = vertices[i + 1];
			const Eigen::Vector3d cur_vertex_pos = positions.row(v).transpose();
			D_m.col(i) = cur_vertex_pos - pivot_vertex_pos;
		}

		// set weight related to rest volume
		const float volume = static_cast<float>(std::abs(D_m.determinant()) / 6);
		wc *= volume;

		constexpr float EPS = 1e-5;
		if (volume < EPS)
		{
			printf("Tetrahedron %d, %d, %d, %d is too small, the tetrahedralization may be wrong!\n", vertices[0], vertices[1], vertices[2], vertices[3]);
			assert(false);
		}

		// precompute D_m_inv
		D_m_inv = D_m.inverse().cast<float>();
	}

	std::vector<Eigen::Triplet<float>> TetStrainConstraint::get_c_AcTAc(int n_vertex_offset) const
	{
		std::vector<Eigen::Triplet<float>> triplets(3 * 4 * 4); // 48

		int local_max_vertex_idx = 0;
		for (int i = 0; i < n_vertices; i++)
		{
			local_max_vertex_idx = std::max(local_max_vertex_idx, vertices[i]);
		}

		Eigen::SparseMatrix<float> A_c(3, local_max_vertex_idx + 1); // preallocate space is necessary
		A_c.setZero();

		for (int i = 0; i < 4; i++)
		{
			const int v = vertices[i];
			for (int j = 0; j < 3; j++)
			{
				if (i == 0)
				{
					A_c.insert(j, v) = -(D_m_inv.coeff(0, j) + D_m_inv.coeff(1, j) + D_m_inv.coeff(2, j));
				}
				else
				{
					A_c.insert(j, v) = D_m_inv.coeff(i - 1, j);
				}
			}
		}

		A_c.makeCompressed();
		// for (int i = 0; i < n_vertices; i++)
		// {
		// 	std::cout << vertices[i] << "\n";
		// }

		Eigen::SparseMatrix<float> AcT_Ac = wc * A_c.transpose() * A_c;
		AcT_Ac.makeCompressed();
		for (int i = 0; i < AcT_Ac.outerSize(); i++)
		{
			for (Eigen::SparseMatrix<float>::InnerIterator itr(AcT_Ac, i); itr; ++itr)
			{
				for (int j = 0; j < 3; j++)
				{
					triplets.emplace_back(3 * n_vertex_offset + 3 * itr.row() + j, 3 * n_vertex_offset + 3 * itr.col() + j, itr.value());
				}
			}
		}

		// std::cout << "AcT_Ac = " << AcT_Ac << "\n";
		// for (int i = 0; i < triplets.size(); i++)
		// {
		// 	printf("%d, %d, %f\n", triplets[i].row(), triplets[i].col(), triplets[i].value());
		// }
		// printf("triplets.size() = %d\n", triplets.size());
		// assert(false);

		return triplets;
	}

	__host__ __device__ void TetStrainConstraint::project_c_AcTAchpc(float* __restrict__ b, const float* __restrict__ q) const
	{
		#ifndef __CUDA_ARCH__
		// for (int i = 0; i < 3 * 8; i += 3)
		// {
		// 	printf("vertex %d pos = %f %f %f\n", i / 3, q[i], q[i + 1], q[i + 2]);
		// }
		#endif			

		Eigen::Vector3f cur_pos[4];
		for (int i = 0; i < 4; i++)
		{
			cur_pos[i] = { q[3 * vertices[i]], q[3 * vertices[i] + 1], q[3 * vertices[i] + 2] };
		#ifndef __CUDA_ARCH__
			// std::cout << "cur_pos[i] = " << cur_pos[i] << "\n";
		#endif			
		}

		Eigen::Matrix3f D_s;
		for (int i = 0; i < 3; i++)
		{
			D_s.col(i) = cur_pos[i + 1] - cur_pos[0];
		}

		Eigen::Matrix3f F = D_s * D_m_inv; // deformation gradient

	#ifndef __CUDA_ARCH__
		// std::cout << "D_s = " << D_s << "\n";
		// std::cout << "D_m_inv = " << D_m_inv << "\n";
		// std::cout << "deformation gradient = " << F << "\n";
	#endif

		Eigen::Matrix3f Achpc;
	#ifdef __CUDA_ARCH__

	#else
		const bool tet_inverted = F.determinant() < 0;
		Eigen::JacobiSVD<Eigen::Matrix3f> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::Matrix3f U = svd.matrixU();
		Eigen::Matrix3f V = svd.matrixV();
		Eigen::Vector3f sigma = svd.singularValues();
		for (int i = 0; i < 3; i++)
		{
			sigma(i) = std::clamp(sigma(i), min_strain_xyz(i), max_strain_xyz(i));
		}

		// if (tet_inverted)
		// {
		// 	sigma(2) = -sigma(2);
		// }

		Achpc = U * sigma.asDiagonal() * V.transpose();
		// std::cout << "Achpc = " << Achpc << "\n";
	#endif

		// apply A_c^T
		for (int i = 0; i < 4; i++)
		{
			const int v = vertices[i];
			
			Eigen::RowVector3f sum_of_products;
			sum_of_products.setZero();
			for (int j = 0; j < 3; j++)
			{
				// pivot vertex
				if (i == 0)
				{
					Eigen::RowVector3f neg_D_m_inv = -(D_m_inv.row(0) + D_m_inv.row(1) + D_m_inv.row(2));
					sum_of_products += Achpc.row(j) * neg_D_m_inv(j);
				}
				else
				{
					sum_of_products += Achpc.row(j) * D_m_inv.coeff(i - 1, j);
				}
			}

			for (int j = 0; j < 3; j++)
			{
			#ifdef __CUDA_ARCH__
				atomicAdd(&b[3 * v + j], sum_of_products[j] * wc);
			#else			
				b[3 * v + j] += sum_of_products[j] * wc;
				// std::cout << "write val = " << sum_of_products[j] << ", " << wc << "\n";
			#endif
			}
		}
	}
}