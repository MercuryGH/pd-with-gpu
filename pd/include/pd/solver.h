#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <pd/deformable_mesh.h>

namespace pd
{
	class Solver
	{
	public:
		void set_model(DeformableMesh* model)
		{
			this->model = model;
			this->dirty = true;
		}

		void precompute(float dt);
		void step(const Eigen::MatrixXd& f_ext, int n_iterations);

		// dirty = true indicates the solver needs to recompute
		bool dirty{ false };

	private:
		// solver params
		float dt;

		// model
		DeformableMesh* model;
		Eigen::SimplicialCholesky<Eigen::SparseMatrix<float>> cholesky_decomp;
		Eigen::MatrixXf A;
	};
}