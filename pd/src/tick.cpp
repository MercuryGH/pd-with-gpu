#include <pd/tick.h>

namespace pd {
        // Physical frame calculation
	void tick(
		igl::opengl::glfw::Viewer& viewer,
		std::unordered_map<int, pd::DeformableMesh>& models,
		const ui::PhysicsParams& physics_params,
		const ui::SolverParams& solver_params,
		pd::Solver& solver,
		std::unordered_map<int, Eigen::MatrixX3d>& f_exts,
		bool always_recompute_normal
	)
	{
		if (models.empty() == true)
		{
			return;
		}

		if (physics_params.enable_gravity == true)
		{
			constexpr double GRAVITY = 9.8;
			for (auto& [id, f_ext] : f_exts)
			{
				// apply gravity at y direction on all vertices
				f_ext.col(1) -= GRAVITY * models[id].get_masses();
			}
		}

		if (solver.use_gpu_for_local_step != solver_params.use_gpu_for_local_step)
		{
			solver.use_gpu_for_local_step = solver_params.use_gpu_for_local_step;
			solver.dirty = true;
		}

		if (solver.algo_changed == true)
		{
			solver.clear_solver();
			solver.set_solver(solver_params.selected_solver);
			solver.algo_changed = false;
		}

		if (solver.dirty == true)
		{
			solver.clear_solver();
			solver.set_dt(solver_params.dt);
			solver.precompute();
			solver.dirty = false;
		}
		solver.step(f_exts, solver_params.n_solver_pd_iterations, solver_params.n_itr_solver_iterations);

		for (const auto& [id, model] : models)
		{
			f_exts[id].setZero();

			// If #V or #F is not changed, no need to call clear()
			// viewer.data_list[idx].clear();
			int idx = viewer.mesh_index(id);
			viewer.data_list[idx].set_vertices(model.positions());
			if (always_recompute_normal)
			{
				viewer.data_list[idx].compute_normals();
			}
		}
	}

    // Frame routine before rendering
    bool pre_draw_handler::operator()(igl::opengl::glfw::Viewer& viewer)
    {
        if (models.empty() == true)
        {
            return false;
        }

        // timer ready
        timer.start();

        // apply mass
        for (auto& [id, model] : models)
        {
            bool flag = model.apply_mass_per_vertex(physics_params.mass_per_vertex);
            if (flag == true)
            {
                solver.dirty = true;
            }
        }

        if (viewer.core().is_animating)
        {
            tick(viewer, models, physics_params, solver_params, solver, f_exts, always_recompute_normal);
        }

        // visualzie points
        const Eigen::RowVector3d RED_COLOR{ 1., 0., 0. };
        const Eigen::RowVector3d SUB_YELLOW_COLOR{ 0.3, 0.6, 0 };
        const Eigen::RowVector3d YELLOW_COLOR{ 0.6, 1, 0 };
        for (const auto& [id, model] : models)
        {
            int idx = viewer.mesh_index(id);
            viewer.data_list[idx].clear_points();
            viewer.data_list[idx].clear_labels();
            for (const int vi : model.get_fixed_vertices())
            {
                viewer.data_list[idx].add_points(model.positions().row(vi), RED_COLOR);
            }

            if (user_control.cur_sel_mesh_id == id)
            {
                Eigen::RowVector3d pos = model.positions().row(user_control.selected_vertex_idx);
                viewer.data_list[idx].add_points(pos, YELLOW_COLOR);

                const Eigen::RowVector3d OFFSET = Eigen::RowVector3d(0, 0.005, 0);
                viewer.data_list[idx].add_label(pos + OFFSET, std::to_string(user_control.selected_vertex_idx));

                int cnt = 1;
                for (const int v : model.get_adj_list().at(user_control.selected_vertex_idx))
                {
                    Eigen::RowVector3d v_p = model.positions().row(v);
                    viewer.data_list[idx].add_points(v_p, SUB_YELLOW_COLOR);
                    std::string neighbor_vertex_prompt = std::to_string(cnt) + "-" + std::to_string(v);
                    cnt++;
                    viewer.data_list[idx].add_label(v_p + OFFSET, neighbor_vertex_prompt);
                }
            }
        }

        timer.stop();
        last_elapse_time = timer.elapsed_milliseconds();
        last_global_step_time = solver.last_global_step_time;
        last_local_step_time = solver.last_local_step_time;
        last_precomputation_time = solver.last_precomputation_time;

        return false;
    }
	double pre_draw_handler::last_elapse_time = 0; 
	double pre_draw_handler::last_local_step_time = 0; 
	double pre_draw_handler::last_global_step_time = 0; 
	double pre_draw_handler::last_precomputation_time = 0;
}