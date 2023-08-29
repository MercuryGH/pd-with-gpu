#pragma once

#include <pd/physics_tick.h>

#include <igl/opengl/glfw/Viewer.h>

#include <ui/obj_manager.h>

#include <pd/algo_ctrl.h>
#include <ui/screen_capture_plugin.h>

#include <rendering/rendering_tick.h>

#include <instancing/instantiator.h>

namespace ui {
	// menu windows
	struct WindowPositionSize
	{
		float pos_x, pos_y;
		float size_x, size_y;
	};

	const WindowPositionSize obj_menu_wps{ -200, 0, 200, -0.1 };
	const WindowPositionSize component_menu_wps{ -200, -150, 200, -0.1 };
	const WindowPositionSize constraint_menu_wps{ -500, 0, 300, 400 };
	const WindowPositionSize instantiator_menu_wps{ -320, -120, 120, 120 };
	const WindowPositionSize pd_menu_wps{ 0, 0, 300, -0.1 };

	void set_window_position_size(const Eigen::Vector4f& viewport_vec, WindowPositionSize wps);

	struct obj_menu_window_handler
	{
		igl::opengl::glfw::Viewer& viewer;
		ObjManager& obj_manager;
		pd::UserControl& user_control;

		void operator()();
	};

	struct constraint_menu_window_handler
	{
		igl::opengl::glfw::Viewer& viewer;
		ObjManager& obj_manager;
		pd::UserControl& user_control;
		pd::PhysicsParams& physics_params;

		void operator()();
	};

	struct instantiator_menu_window_handler
	{
		igl::opengl::glfw::Viewer& viewer;
		ObjManager& obj_manager;
		pd::PhysicsParams& physics_params;

		void operator()();
	};

	struct component_menu_window_handler
	{
		igl::opengl::glfw::Viewer& viewer;

		void operator()();
	};

	struct pd_menu_window_handler
	{
		igl::opengl::glfw::Viewer& viewer;
		ScreenCapturePlugin& screen_capture_plugin;
		ObjManager& obj_manager;
        pd::SolverParams& solver_params;
        pd::PhysicsParams& physics_params;
		pd::UserControl& user_control;
		rendering::pre_draw_handler& frame_callback;
	    std::unordered_map<pd::MeshIDType, pd::DataMatrixX3>& f_exts;
		igl::opengl::glfw::imgui::ImGuizmoWidget& gizmo;

		void operator()();
	};

	// menus
	template<typename T>
	void mesh_manager_menu(
		const std::string& obj_prompts,
		const std::unordered_map<int, T>& meshes,
		ObjManager& obj_manager,
		pd::UserControl& user_control,
		bool is_animating
	);

	void mesh_remove_menu(ObjManager& obj_manager, int id);

    void deformable_mesh_generate_menu(ObjManager& obj_manager, int id);
    void collider_generate_menu(ObjManager& obj_manager, int id);
    void set_constraints_menu(ObjManager& obj_manager, pd::PhysicsParams& physics_params, pd::UserControl& user_control);

	void physics_menu(pd::PhysicsParams& physics_params, const pd::UserControl& user_control);

	void visualization_menu(
		igl::opengl::glfw::Viewer& viewer,
		pd::UserControl& user_control,
		ScreenCapturePlugin& screen_capture_plugin,
		std::unordered_map<pd::MeshIDType, pd::DeformableMesh>& models,
		int id
	);

	void instantiator_menu(instancing::Instantiator& instantiator);

	void simulation_ctrl_menu(
		ObjManager& obj_manager,
		const pd::UserControl& user_control,
        pd::SolverParams& solver_params,
        const pd::PhysicsParams& physics_params,
        igl::opengl::glfw::Viewer& viewer,
        rendering::pre_draw_handler& frame_callback,
	    std::unordered_map<pd::MeshIDType, pd::DataMatrixX3>& f_exts,
		igl::opengl::glfw::imgui::ImGuizmoWidget& gizmo
    );
}
