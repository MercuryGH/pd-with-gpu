#include <rendering/rendering_tick.h>
#include <pd/physics_tick.h>

#include <Eigen/Geometry>

#include <io/mesh_io.h>

namespace rendering {
	void tick(
		igl::opengl::glfw::Viewer& viewer,
		std::unordered_map<pd::MeshIDType, pd::DeformableMesh>& models,
		const std::unordered_map<pd::MeshIDType, std::unique_ptr<primitive::Primitive>>& rigid_colliders,
		const pd::PhysicsParams& physics_params,
		const pd::SolverParams& solver_params,
		std::unordered_map<pd::MeshIDType, pd::DataMatrixX3>& f_exts,
		const pd::UserControl& user_control
	)
	{
		// Physics tick stage
        pd::physics_tick(models, rigid_colliders, physics_params, solver_params, f_exts, user_control);

		// Rendering stage
		rendering_tick(viewer, models, user_control);
	}

	void rendering_tick(
		igl::opengl::glfw::Viewer& viewer,
		std::unordered_map<pd::MeshIDType, pd::DeformableMesh>& models,
		const pd::UserControl& user_control
	)
	{
		if (user_control.headless_mode == true)
		{
			return;
		}

		// set visual data from calculation result
		for (auto& [id, model] : models)
		{
			// If #V or #F is not changed, no need to call clear()
			// viewer.data_list[idx].clear();
			int idx = viewer.mesh_index(id);
			viewer.data_list[idx].set_vertices(model.positions());
			if (user_control.always_recompute_normal)
			{
				viewer.data_list[idx].compute_normals();
			}

			// update vertex normals
			model.vertex_normals() = viewer.data_list[idx].V_normals;
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
				continue;
			}

            for (const pd::VertexIndexType vi : model.get_fixed_vertices())
            {
                viewer.data_list[idx].add_points(model.positions().row(vi), RED_COLOR);
            }

            if (sel_mesh_id == id)
            {
				// vertex index too large
				if (model.positions().rows() <= sel_vertex_idx || model.get_adj_list().size() <= sel_vertex_idx)
				{
					printf("Warning: current selected vertex index %d too large!\n", sel_vertex_idx);
					continue;
				}

                pd::DataRowVector3 pos = model.positions().row(sel_vertex_idx);
                viewer.data_list[idx].add_points(pos, YELLOW_COLOR);

                const pd::DataRowVector3 normal = viewer.data_list[idx].V_normals.row(sel_vertex_idx);
                const pd::DataRowVector3 offset = 0.01 * normal;
                viewer.data_list[idx].add_label(pos + offset, std::to_string(sel_vertex_idx));

                int cnt = 1;
                for (const pd::VertexIndexType v : model.get_adj_list().at(sel_vertex_idx))
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

		// Physics tick stage

        // timer ready
        timer.start();

        if (viewer.core().is_animating)
        {
            pd::physics_tick(models, rigid_colliders, physics_params, solver_params, f_exts, user_control);
        }

        timer.stop();
        last_elapse_time = timer.elapsed_milliseconds();
        last_global_step_time = pd::Solver::instance().last_global_step_time;
        last_local_step_time = pd::Solver::instance().last_local_step_time;
        last_precomputation_time = pd::Solver::instance().last_precomputation_time;

		// Rendering stage
		if (user_control.enable_tetrahedra_visualization == false)
		{
			rendering_tick(viewer, models, user_control);
		}

        return false;
    }
	double pre_draw_handler::last_elapse_time = 0;
	double pre_draw_handler::last_local_step_time = 0;
	double pre_draw_handler::last_global_step_time = 0;
	double pre_draw_handler::last_precomputation_time = 0;
}
