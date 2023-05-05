#include <pd/tick.h>

#include <Eigen/Geometry>

namespace pd {
	void tick(
		igl::opengl::glfw::Viewer& viewer,
		std::unordered_map<pd::MeshIDType, pd::DeformableMesh>& models,
		const ui::PhysicsParams& physics_params,
		const ui::SolverParams& solver_params,
		pd::Solver& solver,
		std::unordered_map<MeshIDType, DataMatrixX3>& f_exts,
		const ui::UserControl& user_control
	)
	{
		// Physics tick stage

        // apply mass
        for (auto& [id, model] : models)
        {
            bool flag = model.apply_mass_per_vertex(physics_params.mass_per_vertex);
            if (flag == true)
            {
                solver.set_dirty(true);
            }
        }

		compute_external_force(viewer, physics_params, models, f_exts);

        physics_tick(models, physics_params, solver_params, solver, f_exts);

		// Rendering stage
		rendering_tick(viewer, models, f_exts, user_control);
	}

	void compute_external_force(
		igl::opengl::glfw::Viewer& viewer,
		const ui::PhysicsParams& physics_params,
		std::unordered_map<pd::MeshIDType, pd::DeformableMesh>& models,
		std::unordered_map<MeshIDType, DataMatrixX3>& f_exts
	)
	{
		if (models.empty())
		{
			return;
		}

		// gravity
		if (physics_params.enable_gravity == true)
		{
			constexpr DataScalar GRAVITY = 9.8;
			for (auto& [id, f_ext] : f_exts)
			{
				// apply gravity at y direction on all vertices
				f_ext.col(1) -= GRAVITY * models[id].get_masses();
			}
		}

		// wind
		if (physics_params.enable_wind == true)
		{
			for (auto& [id, f_ext] : f_exts)
			{
				// DataMatrixX3 vertex_normals = viewer.data_list[idx].V_normals;
				const int idx = viewer.mesh_index(id);
				DataMatrixX3 vertex_normals = viewer.data_list[idx].V_normals;

				// DataVector3 wind_dir(1, 1, 1);
				DataVector3 wind_dir = physics_params.wind_dir.cast<DataScalar>();
				wind_dir.normalize();

				DataScalar wind_force_val = static_cast<DataScalar>(physics_params.wind_force_val);

				// traverse vertices
				for (int i = 0; i < f_ext.rows(); i++)
				{
					DataScalar cos_theta = std::abs(vertex_normals.row(i).dot(wind_dir));
					f_ext.row(i) += wind_dir * wind_force_val * cos_theta;
				}
			}
		}
	}

	void physics_tick(
		std::unordered_map<pd::MeshIDType, pd::DeformableMesh>& models,
		const ui::PhysicsParams& physics_params,
		const ui::SolverParams& solver_params,
		pd::Solver& solver,
		std::unordered_map<MeshIDType, DataMatrixX3>& f_exts
	)
	{
		if (models.empty() == true)
		{
			return;
		}

		if (solver.use_gpu_for_local_step != solver_params.use_gpu_for_local_step)
		{
			solver.use_gpu_for_local_step = solver_params.use_gpu_for_local_step;
			solver.set_dirty(true);
		}

		if (solver.is_algo_changed())
		{
			solver.clear_solver();
			solver.set_solver(solver_params.selected_solver);
			solver.set_algo_changed(false);
		}

		if (solver.is_dirty())
		{
			solver.clear_solver();
			solver.set_dt(solver_params.dt);
			solver.precompute();
			solver.set_dirty(false);
		}
		solver.step(f_exts, solver_params.n_solver_pd_iterations, solver_params.n_itr_solver_iterations);
		// solver.test_step(f_exts, solver_params.n_solver_pd_iterations, solver_params.n_itr_solver_iterations);
	}

	void rendering_tick(
		igl::opengl::glfw::Viewer& viewer,
		std::unordered_map<pd::MeshIDType, pd::DeformableMesh>& models,
		std::unordered_map<MeshIDType, DataMatrixX3>& f_exts,
		const ui::UserControl& user_control
	)
	{
		// set visual data from calculation result
		for (const auto& [id, model] : models)
		{
			f_exts[id].setZero();

			// If #V or #F is not changed, no need to call clear()
			// viewer.data_list[idx].clear();
			int idx = viewer.mesh_index(id);
			viewer.data_list[idx].set_vertices(model.positions());
			if (user_control.always_recompute_normal)
			{
				viewer.data_list[idx].compute_normals();
			}
		}

		// draw debug points
        draw_debug_info(user_control.enable_debug_draw, viewer, models, user_control.cur_sel_mesh_id, user_control.selected_vertex_idx);
	}

    void draw_debug_info(
		bool enable_debug_draw,
		igl::opengl::glfw::Viewer& viewer,
		std::unordered_map<pd::MeshIDType, pd::DeformableMesh>& models,
		pd::MeshIDType sel_mesh_id,
		pd::VertexIndexType sel_vertex_idx
	)
	{
		// visualzie points
        const Eigen::RowVector3d RED_COLOR{ 1., 0., 0. };
        const Eigen::RowVector3d SUB_YELLOW_COLOR{ 0.3, 0.6, 0 };
        const Eigen::RowVector3d YELLOW_COLOR{ 0.6, 1, 0 };
        for (const auto& [id, model] : models)
        {
            int idx = viewer.mesh_index(id);
            viewer.data_list[idx].clear_points();
            viewer.data_list[idx].clear_labels();

			if (enable_debug_draw == false)
			{
				return;
			}

            for (const VertexIndexType vi : model.get_fixed_vertices())
            {
                viewer.data_list[idx].add_points(model.positions().row(vi), RED_COLOR);
            }

            if (sel_mesh_id == id)
            {
                pd::DataRowVector3 pos = model.positions().row(sel_vertex_idx);
                viewer.data_list[idx].add_points(pos, YELLOW_COLOR);

                const pd::DataRowVector3 normal = viewer.data_list[idx].V_normals.row(sel_vertex_idx);
                const pd::DataRowVector3 offset = 0.01 * normal;
                viewer.data_list[idx].add_label(pos + offset, std::to_string(sel_vertex_idx));

                int cnt = 1;
                for (const VertexIndexType v : model.get_adj_list().at(sel_vertex_idx))
                {
                    const pd::DataRowVector3 normal = viewer.data_list[idx].V_normals.row(v);
                    const pd::DataRowVector3 offset = 0.01 * normal;

                    pd::DataRowVector3 v_p = model.positions().row(v);
                    viewer.data_list[idx].add_points(v_p, SUB_YELLOW_COLOR);
                    std::string neighbor_vertex_prompt = std::to_string(cnt) + "-" + std::to_string(v);
                    cnt++;
                    viewer.data_list[idx].add_label(v_p + offset, neighbor_vertex_prompt);
                }
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

		compute_external_force(viewer, physics_params, models, f_exts);

		// Physics tick stage

        // timer ready
        timer.start();

        // apply mass
        for (auto& [id, model] : models)
        {
            bool flag = model.apply_mass_per_vertex(physics_params.mass_per_vertex);
            if (flag == true)
            {
                solver.set_dirty(true);
            }
        }

        if (viewer.core().is_animating)
        {
            physics_tick(models, physics_params, solver_params, solver, f_exts);
        }

        timer.stop();
        last_elapse_time = timer.elapsed_milliseconds();
        last_global_step_time = solver.last_global_step_time;
        last_local_step_time = solver.last_local_step_time;
        last_precomputation_time = solver.last_precomputation_time;

		// Rendering stage
		if (user_control.enable_tetrahedra_visualization == false)
		{
			rendering_tick(viewer, models, f_exts, user_control);
		}

        return false;
    }
	double pre_draw_handler::last_elapse_time = 0; 
	double pre_draw_handler::last_local_step_time = 0; 
	double pre_draw_handler::last_global_step_time = 0; 
	double pre_draw_handler::last_precomputation_time = 0;
}