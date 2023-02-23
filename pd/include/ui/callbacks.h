#pragma once

#include <pd/deformable_mesh.h>
#include <pd/solver.h>
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
		pd::DeformableMesh* model;

		ui::UserControl* user_control;
		bool operator()(igl::opengl::glfw::Viewer& viewer, int button, int modifier)
		{
			if (model->empty() == true)
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

			int fid; // id of the first face hit
			Eigen::Vector3f bc; // barycentric coordinates of hit
			// raycast
			bool hit_vertex = igl::unproject_onto_mesh(
				Eigen::Vector2f(x, y),
				viewer.core().view, // model matrix
				viewer.core().proj, // projection matrix
				viewer.core().viewport, // viewport vector
				model->positions(),
				model->faces(),
				fid,
				bc
			);
			if (hit_vertex == false)
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

			int hit_vertex_idx = model->faces()(fid, find_hit_vertex(bc));

			// Handling user input

			bool flag = false;
			// Hold control
			if (modifier == GLFW_MOD_CONTROL)
			{
				flag = true;
				// apply external force
				user_control->apply_ext_force = true;
				user_control->ext_forced_vertex_idx = hit_vertex_idx;
				user_control->mouse_x = viewer.current_mouse_x;
				user_control->mouse_y = viewer.current_mouse_y;
			}

			if (modifier == GLFW_MOD_SHIFT)
			{
				flag = true;
				// toggle fix/unfix
				user_control->toggle_vertex_fix = true;
				if (user_control->toggle_fixed_vertex_idxs.find(hit_vertex_idx) == user_control->toggle_fixed_vertex_idxs.end())
				{
					user_control->toggle_fixed_vertex_idxs.insert(hit_vertex_idx);
				}
				else
				{
					user_control->toggle_fixed_vertex_idxs.erase(hit_vertex_idx);
				}
			}

			return flag;
		}
	};

	// This functor only works for holding LMC to continuously apply external force to the pointed vertex
	struct mouse_move_handler
	{
		pd::DeformableMesh* model;

		ui::UserControl* user_control;
		ui::PhysicsParams* physics_params;
		Eigen::MatrixX3d* f_ext; // use to apply force in realtime
		bool operator()(igl::opengl::glfw::Viewer& viewer, int button, int modifier)
		{
			if (model->empty() == true)
			{
				return false;
			}
			if (user_control->apply_ext_force == false)
			{
				return false;
			}

			// get origin mouse hit viewport coordinate
			const float x0 = static_cast<float>(user_control->mouse_x);
			const float y0 = viewer.core().viewport(3) - static_cast<float>(user_control->mouse_y);

			// update user control
			user_control->mouse_x = viewer.current_mouse_x;
			user_control->mouse_y = viewer.current_mouse_y;
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

			//std::cout << f_ext->row(user_control->ext_forced_vertex_idx) << "\n";
			// add force
			f_ext->row(user_control->ext_forced_vertex_idx) += (dir.transpose() * physics_params->external_force_val).cast<double>();
			//std::cout << f_ext->row(user_control->ext_forced_vertex_idx) << "\n\n";

			return true;
		}
	};

	// This functor only works for removing external force
	struct mouse_up_handler
	{
		ui::UserControl* user_control;
		bool operator()(igl::opengl::glfw::Viewer& viewer, int button, int modifier)
		{
			if (static_cast<Button>(button) != Button::Left)
			{
				return false;
			}
			if (user_control->apply_ext_force == true)
			{
				user_control->apply_ext_force = false;
			}

			return false;
		}
	};

	// Physical frame calculation
	void tick(
		igl::opengl::glfw::Viewer& viewer,
		pd::DeformableMesh* model,
		ui::PhysicsParams* physics_params,
		ui::SolverParams* solver_params,
		pd::Solver* solver,
		Eigen::MatrixX3d* f_ext)
	{
		if (physics_params->enable_gravity)
		{
			constexpr double GRAVITY = 9.8;

			// apply gravity at y direction on all vertices
			f_ext->col(1).array() -= GRAVITY;
		}

		solver->use_gpu_for_local_step = solver_params->use_gpu_for_local_step;
		if (solver->algo_changed == true)
		{
			solver->clear_solver();
			solver->set_solver(solver_params->selected_solver);
			solver->algo_changed = false;
		}

		if (solver->dirty == true)
		{
			solver->clear_solver();
			solver->set_dt(solver_params->dt);
			solver->precompute();
			solver->dirty = false;
		}
		solver->step(*f_ext, solver_params->n_solver_pd_iterations, solver_params->n_itr_solver_iterations);

		f_ext->setZero();
		// If #V or #F is not changed, no need to call clear()
		//viewer.data().clear();
		viewer.data().set_mesh(model->positions(), model->faces());
	}

	// Frame routine before rendering
	struct pre_draw_handler
	{
		pd::Solver* solver;
		pd::DeformableMesh* model;

		ui::PhysicsParams* physics_params;
		Eigen::MatrixX3d* f_ext; 
		ui::SolverParams* solver_params;

		util::CpuTimer timer;
		static double last_elapse_time;
		static double last_local_step_time;
		static double last_global_step_time;
		static double last_precomputation_time;

		bool operator()(igl::opengl::glfw::Viewer& viewer)
		{
			if (model->empty() == true)
			{
				return false;
			}

			// timer ready
			timer.start();

			bool flag = model->apply_mass_per_vertex(physics_params->mass_per_vertex);
			if (flag == true)
			{
				solver->dirty = true;
			}

			if (viewer.core().is_animating)
			{
				tick(viewer, model, physics_params, solver_params, solver, f_ext);
			}

			// Visualize fixed point
			const Eigen::RowVector3d RED_COLOR{ 1., 0., 0. };
			const Eigen::RowVector3d YELLOW_COLOR{ 0.6, 1., 0. };

			viewer.data().clear_points();
			for (int i = 0; i < model->positions().rows(); i++)
			{
				if (model->is_vertex_fixed(i))
				{
					viewer.data().add_points(model->positions().row(i), RED_COLOR);
				}
			}

			// debug draw
			viewer.data().add_points(model->positions().row(10), YELLOW_COLOR);

			timer.stop();
			last_elapse_time = timer.elapsed_milliseconds();
			last_global_step_time = solver->last_global_step_time;
			last_local_step_time = solver->last_local_step_time;
			last_precomputation_time = solver->last_precomputation_time;

			return false;
		}
	};
	double pre_draw_handler::last_elapse_time; 
	double pre_draw_handler::last_local_step_time; 
	double pre_draw_handler::last_global_step_time; 
	double pre_draw_handler::last_precomputation_time;
}


