#pragma once

#include <pd/deformable_mesh.h>
#include <pd/solver.h>

#include <primitive/primitive.h>

#include <ui/obj_manager.h>
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
		bool operator()(igl::opengl::glfw::Viewer& viewer, int button, int modifier);
	};

	// This functor only works for holding LMC to continuously apply external force to the pointed vertex
	struct mouse_move_handler
	{
		const std::unordered_map<int, pd::DeformableMesh>& models;

		ui::UserControl& user_control;
		ui::PhysicsParams& physics_params;
		std::unordered_map<int, Eigen::MatrixX3d>& f_exts;
		bool operator()(igl::opengl::glfw::Viewer& viewer, int button, int modifier);
	};

	// This functor only works for removing external force
	struct mouse_up_handler
	{
		ui::UserControl& user_control;
		bool operator()(igl::opengl::glfw::Viewer& viewer, int button, int modifier);
	};

	struct gizmo_handler
	{
		igl::opengl::glfw::Viewer& viewer;
		std::unordered_map<int, pd::DeformableMesh>& models;
		std::unordered_map<int, std::unique_ptr<primitive::Primitive>>& rigid_colliders;
		ObjManager& obj_manager;
		std::unordered_map<int, Eigen::MatrixXd>& obj_init_pos_map;
		ui::UserControl& user_control;
		const bool& is_animating;

		void operator()(const Eigen::Matrix4f& T);
	};

	struct keypress_handler
	{
		igl::opengl::glfw::imgui::ImGuizmoWidget& gizmo;
		ObjManager& obj_manager;
		ui::UserControl& user_control;		

		bool operator()(igl::opengl::glfw::Viewer& viewer, int button, int modifier);
	};
}


