#include <iostream>
#include <pd/solver.h>
#include <pd/positional_constraint.h>
#include <pd/edge_length_constraint.h>

namespace pd {
	Solver::Solver()
	{
		solvers[0] = std::make_unique<CholeskyDirect>();
		solvers[1] = std::make_unique<ParallelJacobi>();
		solvers[2] = std::make_unique<AJacobi>(1);
		solvers[3] = std::make_unique<AJacobi>(2);
		solvers[4] = std::make_unique<AJacobi>(3);
		//linear_sys_solver = &solvers[0];
		linear_sys_solver = solvers.begin();
	}

	void Solver::precompute_A()
	{
		const float dtsqr_inv = 1.0f / (dt * dt);
		const int n = model->positions().rows(); // #vertex

		// Use triplets (i, j, val) to represent sparse matrix
		// Triplets will sum up for possible duplicates
		std::vector<Eigen::Triplet<float>> A_triplets;

		// preallocate enough space to avoid vector enlarge overhead
		const size_t vector_init_size = 3 * n;

		A_triplets.reserve(vector_init_size);

		for (const auto& constraint : model->constraints)
		{
			std::vector<Eigen::Triplet<float>> wiSiTAiTAiSi = constraint->get_A_wiSiTAiTAiSi();
			A_triplets.insert(A_triplets.end(), wiSiTAiTAiSi.begin(), wiSiTAiTAiSi.end());
		}

		for (int i = 0; i < n; i++)
		{
			A_triplets.emplace_back(3 * i, 3 * i, model->m(i) * dtsqr_inv);
			A_triplets.emplace_back(3 * i + 1, 3 * i + 1, model->m(i) * dtsqr_inv);
			A_triplets.emplace_back(3 * i + 2, 3 * i + 2, model->m(i) * dtsqr_inv);
		}

		A.resize(3 * n, 3 * n);
		A.setFromTriplets(A_triplets.begin(), A_triplets.end());
		A.makeCompressed();

		if (use_gpu_for_local_step)
		{
			if (gpu_local_solver == nullptr)
			{
				gpu_local_solver = std::make_unique<GpuLocalSolver>();
			}
			gpu_local_solver->gpu_local_step_solver_malloc(3 * n);
			//gpu_local_solver->gpu_object_creation_serial(model->constraints);
			gpu_local_solver->gpu_object_creation_parallel(model->constraints);
		}

		//std::vector<int> v_adj;
		//for (int i = 0; i < n; i++)
		//{
		//	if (std::abs(A.coeff(63, i)) > 1e-3f)
		//	{
		//		v_adj.push_back(i);
		//	}
		//}
		//for (auto v : v_adj) std::cout << v << " ";
		//int cnt = 3;
		//std::cout << A.coeff(0, 3) << "\n";
		//for (int i = 0; i < n / 3; i++)
		//{
		//	assert(A.coeff(3 * i, 3 * i) == A.coeff(3 * i + 1, 3 * i + 1));
		//	assert(A.coeff(3 * i + 2, 3 * i + 2) == A.coeff(3 * i + 1, 3 * i + 1));
		//}
		//for (int i = 0; i < n - cnt; i++)
		//{
		//	std::cout << A.block(3 * i, 3 * i + cnt, 3, 3) << "\n";
		//}
	}

	void Solver::precompute()
	{
		timer.start();

		precompute_A();
		(*linear_sys_solver)->set_A(A, model->constraints);

		timer.stop();
		last_precomputation_time = timer.elapsed_milliseconds();
	}

	void Solver::set_solver(ui::LinearSysSolver sel)
	{
		int idx = static_cast<int>(sel);
		linear_sys_solver = solvers.begin() + idx;
	}

	void Solver::clear_solver()
	{
		if (solvers.size() != 0) // not init
		{
			(*linear_sys_solver)->clear();
		}
		if (use_gpu_for_local_step)
		{
			if (gpu_local_solver != nullptr)
			{
				gpu_local_solver->free_local_gpu_memory_entry();
			}
		}
	}

	void Solver::step(const Eigen::MatrixXd& f_ext, int n_itr, int itr_solver_n_itr)
	{
		(*linear_sys_solver)->set_n_itr(itr_solver_n_itr);

		const float dtsqr = dt * dt;
		const float dt_inv = 1.0f / dt;
		const float dtsqr_inv = 1.0f / dtsqr;
		const int n = model->positions().rows(); // #vertex

		const Eigen::MatrixXd& q = model->positions();
		const Eigen::MatrixXd& v = model->v;

		// auto fails
		const Eigen::MatrixX3d a = f_ext.array().colwise() / (model->m).array(); // M^{-1} f_{ext}
		const Eigen::MatrixX3f q_explicit = (q + dt * v + dtsqr * a).cast<Float>(); // n * 3 matrix

		const auto flatten = [n](const Eigen::MatrixXf& q) {
			assert(q.cols() == 3);
			assert(q.rows() == n);

			Eigen::VectorXf ret;
			ret.resize(3 * n);
			for (int i = 0; i < n; i++)
			{
				ret.block(3 * i, 0, 3, 1) = q.row(i).transpose();
			}

			return ret;
		};

		const auto s_n = flatten(q_explicit); // 3n * 1 vector

		// Compute the value (M / dt^2) * s_n 
		// since M is diagonal we use vector product to optimize
		Eigen::VectorXf global_solve_b_mass_term; // (M / dt^2) * s_n 
		global_solve_b_mass_term.resize(3 * n);
		// can be parallelized
		for (int i = 0; i < n; i++)
		{
			Eigen::Matrix3f m_i;
			m_i.setZero();
			for (int j = 0; j < 3; j++)
				m_i(j, j) = static_cast<float>(model->m(i));

			const auto sn_i = s_n.block(3 * i, 0, 3, 1);
			global_solve_b_mass_term.block(3 * i, 0, 3, 1) = dtsqr_inv * m_i * sn_i;
		}

		// q_{n+1}
		Eigen::VectorXf q_nplus1 = s_n;

		//// Global solve b
		Eigen::VectorXf b;
		b.resize(3 * n);

		last_local_step_time = 0;
		last_global_step_time = 0;
		for (int k = 0; k < n_itr; k++)
		{
			b.setZero();
			b += global_solve_b_mass_term;

			timer.start();

			if (use_gpu_for_local_step)
			{
				local_step_gpu(q_nplus1, b);
			}
			else
			{
				local_step_cpu(q_nplus1, b);
			}

			timer.stop();
			last_local_step_time += timer.elapsed_milliseconds();

			//if (k == 0)
			//	std::cout << "b = " << b << "\n";

			// printf("%d PD itr\n", k);

			timer.start();

			q_nplus1 = (*linear_sys_solver)->solve(b);

			timer.stop();
			last_global_step_time += timer.elapsed_milliseconds();

			//if (k == 0)
				//std::cout << "q_nplus1 = " << q_nplus1 << "\n";
		}
		//last_local_step_time /= n_itr;
		//last_global_step_time /= n_itr;

		// 3n * 1 vector to n * 3 matrix
		const auto unflatten = [n](const Eigen::VectorXf& p) {
			assert(n * 3 == p.rows());

			Eigen::MatrixXf ret(n, 3);
			for (int i = 0; i < n; i++)
			{
				ret.row(i) = p.block(3 * i, 0, 3, 1).transpose();
			}
			return ret;
		};

		const Eigen::MatrixXd result = unflatten(q_nplus1).cast<double>();
		//{
		//	std::cout << "Test\n";
		//	std::cout << result.block(30, 0, 9, 1) << "\n";
		//}
		model->update_positions_and_velocities(result, (result - model->positions()) * static_cast<double>(dt_inv));
	}

	void Solver::local_step_cpu(const Eigen::VectorXf& q_nplus1, Eigen::VectorXf& b)
	{
		for (const auto& constraint : model->constraints)
		{
			//constraint->project_i_wiSiTAiTBipi(b.data(), q_nplus1.data(), b.size());
			//continue;
			const Eigen::VectorXf pi = constraint->local_solve(q_nplus1);

			//std::cout << pi << "\n";
			//const auto test = constraint->get_i_wiSiTAiTBipi(pi);
			//std::cout << "test(0) = " << test(0) << "\n";

			// This can be further optimized
			b += constraint->get_i_wiSiTAiTBipi(pi);
		}
	}

	void Solver::local_step_gpu(const Eigen::VectorXf& q_nplus1, Eigen::VectorXf& b)
	{
		gpu_local_solver->gpu_local_step_entry(q_nplus1, b);
	}
}