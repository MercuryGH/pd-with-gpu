#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuizmoWidget.h>

#include <pd/constraint.h>
#include <pd/deformable_mesh.h>
#include <pd/solver.h>
#include <pd/tick.h>

#include <ui/obj_manager.h>
#include <ui/physics_params.h>
#include <ui/solver_params.h>
#include <ui/user_control.h>
#include <ui/menus.h>
#include <ui/input_callbacks.h>
#include <ui/screen_capture_plugin.h>

#include <meshgen/mesh_generator.h>

#include <primitive/primitive.h>

#include <util/gpu_helper.h>

int main(int argc, char* argv[])
{
	// TODO: hide igl::opengl::glfw::Viewer usage
	igl::opengl::glfw::Viewer viewer;
	ui::ScreenCapturePlugin screen_capture_plugin;
	viewer.plugins.push_back(&screen_capture_plugin); // place in front of menu to avoid capturing UI contents
	igl::opengl::glfw::imgui::ImGuiPlugin menu_plugin;
	viewer.plugins.push_back(&menu_plugin);

	igl::opengl::glfw::imgui::ImGuiMenu main_menu;
	menu_plugin.widgets.push_back(&main_menu);
	igl::opengl::glfw::imgui::ImGuiMenu obj_menu;
	menu_plugin.widgets.push_back(&obj_menu);
	igl::opengl::glfw::imgui::ImGuiMenu instantiator_menu;
	menu_plugin.widgets.push_back(&instantiator_menu);
	igl::opengl::glfw::imgui::ImGuiMenu constraint_menu;
	menu_plugin.widgets.push_back(&constraint_menu);
	igl::opengl::glfw::imgui::ImGuiMenu component_menu;
	menu_plugin.widgets.push_back(&component_menu);

	// Only 1 gizmo during simulation
	igl::opengl::glfw::imgui::ImGuizmoWidget gizmo;
	gizmo.visible = false;
	gizmo.operation = ImGuizmo::OPERATION::TRANSLATE;
	menu_plugin.widgets.push_back(&gizmo);

    // transformation aux
	std::unordered_map<pd::MeshIDType, Eigen::MatrixXd> obj_init_pos_map; // obj_id to its initial position matrix

	// pd simulatee
	std::unordered_map<pd::MeshIDType, pd::DeformableMesh> models;
	std::unordered_map<pd::MeshIDType, pd::DataMatrixX3> f_exts;

	// rigid body colliders
	std::unordered_map<pd::MeshIDType, std::unique_ptr<primitive::Primitive>> rigid_colliders;

	pd::Solver solver(models, rigid_colliders);

	ui::UserControl user_control;
	ui::PhysicsParams physics_params;
	ui::SolverParams solver_params;

	static int total_n_constraints = 0;
	static bool always_recompute_normal = false;
	ui::ObjManager obj_manager{ viewer, gizmo, solver, models, rigid_colliders, obj_init_pos_map, f_exts, user_control, solver_params, total_n_constraints };

	gizmo.callback = ui::gizmo_handler{ viewer, models, rigid_colliders, obj_manager, obj_init_pos_map, user_control, viewer.core().is_animating };
	viewer.callback_mouse_down = ui::mouse_down_handler{ models, user_control };
	viewer.callback_mouse_move = ui::mouse_move_handler{ models, user_control, physics_params, f_exts };
	viewer.callback_mouse_up = ui::mouse_up_handler{ user_control };
	viewer.callback_key_pressed = ui::keypress_handler{ gizmo, obj_manager, user_control };

	pd::pre_draw_handler frame_callback{ solver, models, physics_params, f_exts, solver_params, user_control, always_recompute_normal };
	viewer.callback_pre_draw = frame_callback; // frame routine

	obj_menu.callback_draw_viewer_window = ui::obj_menu_window_handler{ viewer, obj_manager, user_control };
	constraint_menu.callback_draw_viewer_window = ui::constraint_menu_window_handler{ viewer, obj_manager, user_control, physics_params };
	component_menu.callback_draw_viewer_window = ui::component_menu_window_handler{ viewer };
	instantiator_menu.callback_draw_viewer_window = ui::instantiator_menu_window_handler{ viewer, obj_manager, physics_params };

	main_menu.callback_draw_viewer_window = ui::pd_menu_window_handler{ viewer, screen_capture_plugin, obj_manager, solver, solver_params, physics_params, user_control, frame_callback, f_exts, gizmo, always_recompute_normal };

	viewer.launch(true, false, "Projective Dynamics", 0, 0);

	return 0;
}