#pragma once

#include <ui/user_control.h>
#include <ui/physics_params.h>
#include <ui/solver_params.h>

//#include <GLFW/glfw3.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject_onto_mesh.h>

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
			// TODO: implement
			return false;
		}
	};

	// This functor only works for remove external force
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

		if (solver->dirty == true)
		{
			solver->precompute(solver_params->dt);
		}
		solver->step(*f_ext, solver_params->n_solver_iterations);

		f_ext->setZero();
		viewer.data().clear();
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
		bool operator()(igl::opengl::glfw::Viewer& viewer)
		{
			if (model->empty() == true)
			{
				return false;
			}
			model->dimension_check();

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
			const Eigen::RowVector3d COLOR{ 1., 0., 0. };
			viewer.data().clear_points();
			for (int i = 0; i < model->positions().rows(); i++)
			{
				if (model->is_vertex_fixed(i))
				{
					viewer.data().add_points(model->positions().row(i), COLOR);
				}
			}

			return false;
		}
	};
}


