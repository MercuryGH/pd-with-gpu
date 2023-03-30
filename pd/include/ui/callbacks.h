#pragma once

#include <pd/deformable_mesh.h>
#include <pd/solver.h>

#include <primitive/primitive.h>

#include <ui/user_control.h>
#include <ui/physics_params.h>
#include <ui/solver_params.h>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject_onto_mesh.h>
#include <util/cpu_timer.h>

namespace ui
{
	using Button = igl::opengl::glfw::Viewer::MouseButton;

	// Note: The ret val of callbacks in this file does not matter currently.

	// Functor for mouse down callback.
	struct mouse_down_handler
	{
		std::unordered_map<int, pd::DeformableMesh>& models;

		ui::UserControl& user_control;
		bool operator()(igl::opengl::glfw::Viewer& viewer, int button, int modifier)
		{
			if (models.empty() == true)
			{
				return false;
			}

			// decode keycode
			if (static_cast<Button>(button) != Button::Left)
			{
				return false;
			}

			// get actual mouse hit viewport coordinate
			const float x = static_cast<float>(viewer.current_mouse_x);
			const float y = viewer.core().viewport(3) - static_cast<float>(viewer.current_mouse_y);

			bool has_vertex_hit = false;
			int mesh_id = -1;
			int fid; // id of the first face hit
			Eigen::Vector3f bc; // barycentric coordinates of hit
			for (const auto& [id, model] : models)
			{
				// raycast
				bool hit_vertex = igl::unproject_onto_mesh(
					Eigen::Vector2f(x, y),
					viewer.core().view, // model matrix
					viewer.core().proj, // projection matrix
					viewer.core().viewport, // viewport vector
					model.positions(),
					model.faces(),
					fid,
					bc
				);
				if (hit_vertex == true)
				{
					// has >= 1 mesh begin hit
					has_vertex_hit = true;
					// update mesh_id
					mesh_id = id;

					// if the user preferred mesh is hit, we breaks earlier.
					// otherwise we continue to find the hit.
					if (id == user_control.cur_sel_mesh_id)
					{
						break;
					}
				}
			}
			if (has_vertex_hit == false) 
			{
				return false;
			}

			const auto find_hit_vertex = [](const Eigen::Vector3f &bc) {
				int ret = 0;
				// find the nearest vertex around the hit point
				if (bc(1) > bc(0) && bc(1) > bc(2)) {
					ret = 1;
				}
				if (bc(2) > bc(0) && bc(2) > bc(1)) {
					ret = 2;
				}
				return ret;
			};

			int hit_vertex_idx = models[mesh_id].faces()(fid, find_hit_vertex(bc));

			// Handling user input

			bool flag = false;
			// Hold control
			if (modifier == GLFW_MOD_CONTROL)
			{
				flag = true;
				// apply external force
				user_control.apply_ext_force_mesh_id = mesh_id;
				user_control.apply_ext_force = true;
				user_control.ext_forced_vertex_idx = hit_vertex_idx;
				user_control.mouse_x = viewer.current_mouse_x;
				user_control.mouse_y = viewer.current_mouse_y;
			}
			else if (modifier == GLFW_MOD_SHIFT)
			{
				flag = true;
				// toggle fix/unfix
				user_control.toggle_vertex_fix = true;
				if (user_control.toggle_fixed_vertex_idxs.find(hit_vertex_idx) == user_control.toggle_fixed_vertex_idxs.end())
				{
					user_control.toggle_fixed_vertex_idxs.insert(hit_vertex_idx);
				}
				else
				{
					user_control.toggle_fixed_vertex_idxs.erase(hit_vertex_idx);
				}
			}
			else
			{
				user_control.cur_sel_mesh_id = mesh_id;
				user_control.selected_vertex_idx = hit_vertex_idx;
			}

			return flag;
		}
	};

	// This functor only works for holding LMC to continuously apply external force to the pointed vertex
	struct mouse_move_handler
	{
		const std::unordered_map<int, pd::DeformableMesh>& models;

		ui::UserControl& user_control;
		ui::PhysicsParams& physics_params;
		std::unordered_map<int, Eigen::MatrixX3d>& f_exts;
		bool operator()(igl::opengl::glfw::Viewer& viewer, int button, int modifier)
		{
			if (models.empty() == true)
			{
				return false;
			}
			if (user_control.apply_ext_force == false)
			{
				return false;
			}

			// get origin mouse hit viewport coordinate
			const float x0 = static_cast<float>(user_control.mouse_x);
			const float y0 = viewer.core().viewport(3) - static_cast<float>(user_control.mouse_y);

			// update user control
			user_control.mouse_x = viewer.current_mouse_x;
			user_control.mouse_y = viewer.current_mouse_y;
			// get current mouse hit viewport coordinate
			const float x1 = static_cast<float>(viewer.current_mouse_x);
			const float y1 = viewer.core().viewport(3) - static_cast<float>(viewer.current_mouse_y);

			const Eigen::Vector3f p0 = igl::unproject(
				Eigen::Vector3f(x0, y0, 0.5f),
				viewer.core().view,
				viewer.core().proj,
				viewer.core().viewport
			);
			const Eigen::Vector3f p1 = igl::unproject(
				Eigen::Vector3f(x1, y1, 0.5f),
				viewer.core().view,
				viewer.core().proj,
				viewer.core().viewport
			);
			const Eigen::Vector3f dir = (p1 - p0).normalized();

			//std::cout << f_ext.row(user_control.ext_forced_vertex_idx) << "\n";
			// add force
			int apply_f_ext_obj_id = user_control.apply_ext_force_mesh_id;
			f_exts[apply_f_ext_obj_id].row(user_control.ext_forced_vertex_idx) += (dir.transpose() * physics_params.external_force_val).cast<double>();
			//std::cout << f_ext.row(user_control.ext_forced_vertex_idx) << "\n\n";

			return true;
		}
	};

	// This functor only works for removing external force
	struct mouse_up_handler
	{
		ui::UserControl& user_control;
		bool operator()(igl::opengl::glfw::Viewer& viewer, int button, int modifier)
		{
			if (static_cast<Button>(button) != Button::Left)
			{
				return false;
			}
			if (user_control.apply_ext_force == true)
			{
				user_control.apply_ext_force = false;
			}

			return false;
		}
	};

	struct gizmo_handler
	{
		igl::opengl::glfw::Viewer& viewer;
		std::unordered_map<int, pd::DeformableMesh>& models;
		std::unordered_map<int, std::unique_ptr<primitive::Primitive>>& rigid_colliders;
		std::unordered_map<int, Eigen::MatrixXd>& obj_init_pos_map;
		ui::UserControl& user_control;

		void operator()(const Eigen::Matrix4f& T)
		{
			if (models.empty() && rigid_colliders.empty())
			{
				return;
			}

			const int obj_id = user_control.cur_sel_mesh_id;
			const Eigen::MatrixXd& V = obj_init_pos_map[obj_id];

			const Eigen::Matrix4d TT = T.cast<double>().transpose();
			const Eigen::MatrixXd positions = (V.rowwise().homogeneous() * TT).rowwise().hnormalized();
			// const Eigen::MatrixXd positions = (TT).rowwise().hnormalized();

			if (models.find(obj_id) != models.end())
			{
				// TODO: apply lazy update (not necessary indeed)
				models[obj_id].set_positions(positions);
			}
			else if (rigid_colliders.find(obj_id) != rigid_colliders.end())
			{
				Eigen::Vector3f center = 0.5 * (positions.colwise().maxCoeff() + positions.colwise().minCoeff()).transpose().cast<float>();
				rigid_colliders[obj_id]->set_center(center);
				// std::cout << center << "\n";
			}
			else
			{
				printf("Error: cannot find obj with id %d!\n", obj_id);
			}

			int idx = viewer.mesh_index(obj_id);
			viewer.data_list[idx].set_vertices(positions);
			viewer.data_list[idx].compute_normals();
		}
	};

	struct keypress_handler
	{
		igl::opengl::glfw::imgui::ImGuizmoWidget& gizmo;

		bool operator()(igl::opengl::glfw::Viewer& viewer, int button, int modifier)
		{
			switch (button)
			{
				case ' ': gizmo.visible = !gizmo.visible; return true;
				case 'W': case 'w': gizmo.operation = ImGuizmo::TRANSLATE; return true;
				case 'E': case 'e': gizmo.operation = ImGuizmo::ROTATE;    return true;
				case 'R': case 'r': gizmo.operation = ImGuizmo::SCALE;     return true;
			}
			return false;
		}
	};

	// Physical frame calculation
	void tick(
		igl::opengl::glfw::Viewer& viewer,
		std::unordered_map<int, pd::DeformableMesh>& models,
		ui::PhysicsParams& physics_params,
		ui::SolverParams& solver_params,
		pd::Solver& solver,
		std::unordered_map<int, Eigen::MatrixX3d>& f_exts)
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
				// TODO: don't enforce gravity on pinned vertex
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
		}
	}

	// Frame routine before rendering
	struct pre_draw_handler
	{
		pd::Solver& solver;
		std::unordered_map<int, pd::DeformableMesh>& models;

		ui::PhysicsParams& physics_params;
		std::unordered_map<int, Eigen::MatrixX3d>& f_exts;
		ui::SolverParams& solver_params;
		const ui::UserControl& user_control;

		util::CpuTimer timer;
		static double last_elapse_time;
		static double last_local_step_time;
		static double last_global_step_time;
		static double last_precomputation_time;

		bool operator()(igl::opengl::glfw::Viewer& viewer)
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
				tick(viewer, models, physics_params, solver_params, solver, f_exts);
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
					for (const int v : model.get_adj_list().at(user_control.selected_vertex_idx))
					{
						Eigen::RowVector3d v_p = model.positions().row(v);
						viewer.data_list[idx].add_points(v_p, SUB_YELLOW_COLOR);
						viewer.data_list[idx].add_label(v_p + OFFSET, std::to_string(v));
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
	};
	double pre_draw_handler::last_elapse_time = 0; 
	double pre_draw_handler::last_local_step_time = 0; 
	double pre_draw_handler::last_global_step_time = 0; 
	double pre_draw_handler::last_precomputation_time = 0;
}


