#include <iostream>
#include <pd/solver.h>
#include <pd/positional_constraint.h>
#include <pd/edge_strain_constraint.h>

namespace pd {

	Solver::Solver(
		std::unordered_map<MeshIDType, DeformableMesh>& models, 
		std::unordered_map<pd::MeshIDType, std::unique_ptr<primitive::Primitive>>& rigid_colliders
	):
	models(models), 
	rigid_colliders(rigid_colliders),
	dirty(true)
	{
		solvers[0] = std::make_unique<CholeskyDirect>();
		solvers[1] = std::make_unique<ParallelJacobi>();
		solvers[2] = std::make_unique<AJacobi>(1);
		solvers[3] = std::make_unique<AJacobi>(2);
		solvers[4] = std::make_unique<AJacobi>(3);
		linear_sys_solver = solvers.begin();
		gpu_local_solver = std::make_unique<GpuLocalSolver>();
	}

	void Solver::precompute_A()
	{
		const SimScalar dtsqr_inv = 1.0 / (dt * dt);
		int total_n = 0; // #Vertex
		for (const auto& [id, model] : models)
		{
			total_n += model.positions().rows();
		}

		// Use triplets (i, j, val) to represent sparse matrix
		// Triplets will sum up for possible duplicates
		std::vector<Eigen::Triplet<SimScalar>> A_triplets;

		// preallocate enough space to avoid vector enlarge overhead
		const int vector_init_size = 3 * total_n;

		A_triplets.reserve(vector_init_size);

		int acc = 0;
		for (const auto& [id, model] : models)
		{
			int n = model.positions().rows();

			for (const auto& constraint : model.constraints)
			{
				std::vector<Eigen::Triplet<SimScalar>> wiSiTAiTAiSi = constraint->get_c_AcTAc(acc);
				A_triplets.insert(A_triplets.end(), wiSiTAiTAiSi.begin(), wiSiTAiTAiSi.end());
			}

			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					A_triplets.emplace_back(3 * acc + 3 * i + j, 3 * acc + 3 * i + j, model.m(i) * dtsqr_inv);
				}
			}
			acc += n;
		}
		assert(acc == total_n);

		A.resize(3 * total_n, 3 * total_n);
		A.setFromTriplets(A_triplets.begin(), A_triplets.end());
		A.makeCompressed();

		if (use_gpu_for_local_step)
		{
			gpu_local_solver->gpu_local_step_solver_alloc(3 * total_n);
			//gpu_local_solver->gpu_object_creation_serial(model->constraints);

			gpu_local_solver->gpu_object_creation_parallel(models);  
		}

		// std::cout << A << "\n";
		// while (1);

		// std::cout << A.row(1) << "\n";
		// assert(false);

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
		(*linear_sys_solver)->set_A(A, models);

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
		if (gpu_local_solver != nullptr)
		{
			gpu_local_solver->free_local_gpu_memory_entry();
		}
	}

	void Solver::step(const std::unordered_map<MeshIDType, DataMatrixX3>& f_exts, int n_itr, int itr_solver_n_itr)
	{
		(*linear_sys_solver)->set_n_itr(itr_solver_n_itr);

		const SimScalar dtsqr = dt * dt;
		const SimScalar dt_inv = 1.0 / dt;
		const SimScalar dtsqr_inv = 1.0 / dtsqr;
		int total_n = 0; // #Vertex
		for (const auto& [id, model] : models)
		{
			total_n += model.positions().rows();
		}

		SimPositions q_nplus1;
		q_nplus1.resize(3 * total_n);
		int acc = 0;
		for (const auto& [id, model] : models)
		{
			const PositionData& q = model.positions();
			const VelocityData& v = model.v;

			// auto fails
			const DataMatrixX3 a = f_exts.at(id).array().colwise() / (model.m).array(); // M^{-1} f_{ext}
			SimMatrixX3 q_explicit = (q + v * dt + a * dtsqr).cast<SimScalar>(); // n * 3 matrix

			// resolve collision at desired vertex position
			DeformableMesh::resolve_collision(rigid_colliders, q_explicit);

			const int n = model.positions().rows();

			// use this code to fix vertex
			for (const VertexIndexType vi : model.fixed_vertices)
			{
				q_explicit.row(vi) = q.row(vi).cast<SimScalar>();
			}

			const auto flatten = [n](const SimMatrixX3& q) {
				assert(q.cols() == 3);
				assert(q.rows() == n);

				SimPositions ret;
				ret.resize(3 * n);
				for (int i = 0; i < n; i++)
				{
					ret.block<3, 1>(3 * i, 0) = q.row(i).transpose();
				}

				return ret;
			};

			const auto s_n = flatten(q_explicit); // 3n * 1 vector

			q_nplus1.block(3 * acc, 0, 3 * n, 1) = s_n;
			acc += n;
		}

		// Compute the value (M / dt^2) * s_n 
		// since M is diagonal we use vector product to optimize
		SimVectorX global_solve_b_mass_term; // (M / dt^2) * s_n 
		global_solve_b_mass_term.resize(3 * total_n);
		acc = 0;
		for (const auto& [id, model] : models)
		{
			const int n = model.positions().rows();
			for (int i = 0; i < n; i++)
			{
				SimMatrix3 m_i;
				m_i.setZero();
				for (int j = 0; j < 3; j++)
					m_i(j, j) = static_cast<float>(model.m(i));

				const auto sn_i = q_nplus1.block<3, 1>(3 * acc + 3 * i, 0);
				global_solve_b_mass_term.block<3, 1>(3 * acc + 3 * i, 0) = m_i * sn_i * static_cast<SimScalar>(dtsqr_inv);
			}
			acc += n;
		}

		//// Global solve b
		SimVectorX b;
		b.resize(3 * total_n);

		last_local_step_time = 0;
		last_global_step_time = 0;
		for (int k = 0; k < n_itr; k++)
		{
			b = global_solve_b_mass_term;

			timer.start();

			use_gpu_for_local_step ? local_step_gpu(q_nplus1, b) : local_step_cpu(q_nplus1, b);

			timer.stop();
			last_local_step_time += timer.elapsed_milliseconds();

			timer.start();

			q_nplus1 = (*linear_sys_solver)->solve(b);

			timer.stop();
			last_global_step_time += timer.elapsed_milliseconds();
		}

		// 3n * 1 vector to n * 3 matrix
		const auto unflatten = [total_n](const SimPositions& p) {
			assert(total_n * 3 == p.rows());

			SimMatrixX3 ret(total_n, 3);
			for (int i = 0; i < total_n; i++)
			{
				ret.row(i) = p.block<3, 1>(3 * i, 0).transpose();
			}
			return ret;
		};

		// resolve collision at the end of solver
		SimMatrixX3 p = unflatten(q_nplus1);
		DeformableMesh::resolve_collision(rigid_colliders, p);

		PositionData positions = p.cast<double>();
		acc = 0;
		for (auto& [id, model] : models)
		{
			int n = model.positions().rows();

			const PositionData& cur_model_positions = positions.block(acc, 0, n, 3);

			const VelocityData cur_model_velocities = (cur_model_positions - model.positions()) * dt_inv;
			model.update_positions_and_velocities(cur_model_positions, cur_model_velocities);

			acc += n;
		}
	}

	void Solver::test_step(const std::unordered_map<MeshIDType, DataMatrixX3>& f_exts, int n_itr, int itr_solver_n_itr)
	{
		(*linear_sys_solver)->set_n_itr(itr_solver_n_itr);

		const SimScalar dtsqr = dt * dt;
		const SimScalar dt_inv = 1.0 / dt;
		const SimScalar dtsqr_inv = 1.0 / dtsqr;
		int total_n = 0; // #Vertex
		for (const auto& [id, model] : models)
		{
			total_n += model.positions().rows();
		}

		SimPositions q_nplus1;
		q_nplus1.resize(3 * total_n);
		int acc = 0;
		for (const auto& [id, model] : models)
		{
			const PositionData& q = model.positions();
			const VelocityData& v = model.v;

			// auto fails
			// const DataMatrixX3 a = f_exts.at(id).array().colwise() / (model.m).array(); // M^{-1} f_{ext}

			// TODO: for test only
			DataMatrixX3 f_exts_test;
			f_exts_test.resizeLike(f_exts.at(id));
			f_exts_test.setZero();
        	f_exts_test.col(1).array() = -9.81;
			const DataMatrixX3 a = f_exts_test.array().colwise() / (model.m).array(); // M^{-1} f_{ext}

			// Eigen::MatrixX3f q_explicit = (q + dt * v + dtsqr * a).cast<float>(); // n * 3 matrix
			// Eigen::MatrixX3f 
			SimMatrixX3 q_explicit = (q + v * dt + a * dtsqr).cast<SimScalar>(); // n * 3 matrix

			static int cnt = 0;
			// std::cout << "q: " << q << "\n";
			// std::cout << "v: " << v << "\n";
			// std::cout << "a: " << a << "\n";
			// std::cout << "q_explicit" << q_explicit << "\n";
			cnt++;

			// resolve collision at desired vertex position
			DeformableMesh::resolve_collision(rigid_colliders, q_explicit);

			const int n = model.positions().rows();

			// TODO: determine whether use this code to fix vertex
			// for (const int vi : model.fixed_vertices)
			// {
			// 	q_explicit.row(vi) = q.row(vi).cast<float>();
			// }

			const auto flatten = [n](const SimMatrixX3& q) {
				assert(q.cols() == 3);
				assert(q.rows() == n);

				SimPositions ret;
				ret.resize(3 * n);
				for (int i = 0; i < n; i++)
				{
					ret.block<3, 1>(3 * i, 0) = q.row(i).transpose();
				}

				return ret;
			};

			const auto s_n = flatten(q_explicit); // 3n * 1 vector

			q_nplus1.block(3 * acc, 0, 3 * n, 1) = s_n;
			acc += n;
		}

		// Compute the value (M / dt^2) * s_n 
		// since M is diagonal we use vector product to optimize
		SimVectorX global_solve_b_mass_term; // (M / dt^2) * s_n 
		global_solve_b_mass_term.resize(3 * total_n);
		acc = 0;
		for (const auto& [id, model] : models)
		{
			const int n = model.positions().rows();
			for (int i = 0; i < n; i++)
			{
				SimMatrix3 m_i;
				m_i.setZero();
				for (int j = 0; j < 3; j++)
					m_i(j, j) = static_cast<float>(model.m(i));

				const auto sn_i = q_nplus1.block<3, 1>(3 * acc + 3 * i, 0);
				global_solve_b_mass_term.block<3, 1>(3 * acc + 3 * i, 0) = m_i * sn_i * static_cast<SimScalar>(dtsqr_inv);
			}
			acc += n;
		}

		// q_{n+1}
		// Eigen::VectorXf q_nplus1 = s_n;

		//// Global solve b
		SimVectorX b;
		b.resize(3 * total_n);

		last_local_step_time = 0;
		last_global_step_time = 0;
		for (int k = 0; k < n_itr; k++)
		{
			b = global_solve_b_mass_term;
			//if (k == 0)
			// std::cout << "b = " << b << "\n";
			// while (1);

			timer.start();

			use_gpu_for_local_step ? local_step_gpu(q_nplus1, b) : local_step_cpu(q_nplus1, b);

			timer.stop();
			last_local_step_time += timer.elapsed_milliseconds();

			timer.start();

			// while (1);
            static int cnt = 0;
            if (true)
			{
                cnt++;
                std::cout << cnt << ", " << k << ":\n";
				std::cout << b << "\n";
			}

			q_nplus1 = (*linear_sys_solver)->solve(b);

			timer.stop();
			last_global_step_time += timer.elapsed_milliseconds();

            // static int cnt = 0;
            // if (true)
			// {
                // cnt++;
				// std::cout << k << ": b(0, 1, 2) = " << b(0) << ", " << b(1) << ", " << b(2) << "\n";
				// std::cout << k << ": q(0, 1, 2) = " << q_nplus1(0) << ", " << q_nplus1(1) << ", " << q_nplus1(2) << "\n";
			// }
			//if (k == 0)
				//std::cout << "q_nplus1 = " << q_nplus1 << "\n";
		}

		// 3n * 1 vector to n * 3 matrix
		const auto unflatten = [total_n](const SimPositions& p) {
			assert(total_n * 3 == p.rows());

			SimMatrixX3 ret(total_n, 3);
			for (int i = 0; i < total_n; i++)
			{
				ret.row(i) = p.block<3, 1>(3 * i, 0).transpose();
			}
			return ret;
		};

		// resolve collision at the end of solver
		SimMatrixX3 p = unflatten(q_nplus1);
		DeformableMesh::resolve_collision(rigid_colliders, p);

		PositionData positions = p.cast<double>();
		acc = 0;
		for (auto& [id, model] : models)
		{
			int n = model.positions().rows();

			const PositionData& cur_model_positions = positions.block(acc, 0, n, 3);

			const VelocityData cur_model_velocities = (cur_model_positions - model.positions()) * dt_inv;
			model.update_positions_and_velocities(cur_model_positions, cur_model_velocities);

			acc += n;
		}
	}

	void Solver::local_step_cpu(const SimPositions& q_nplus1, SimVectorX& b)
	{
		int acc = 0;
		for (const auto& [id, model] : models)
		{
			const int n = model.positions().rows();
			for (const auto& constraint : model.constraints)
			{
				//constraint->project_c_AcTAchpc(b.data(), q_nplus1.data(), b.size());
				//continue;

				// const Eigen::VectorXf& cur_model_q_n_plus1 = q_nplus1.block(3 * acc, 0, 3 * n, 1);
				// assert(cur_model_q_n_plus1.size() == 3 * n);

				constraint->project_c_AcTAchpc(b.data(), q_nplus1.data());
			}
			acc += n;
		}
	}

	void Solver::local_step_gpu(const SimPositions& q_nplus1, SimVectorX& b)
	{
		gpu_local_solver->gpu_local_step_entry(q_nplus1, b);
	}
}