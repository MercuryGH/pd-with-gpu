#define IGL_VIEWER_VIEWER_QUIET // disable extra debug info printing

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuizmoWidget.h>

#include <pd/constraint.h>
#include <pd/deformable_mesh.h>
#include <pd/solver.h>
#include <pd/physics_tick.h>

#include <rendering/rendering_tick.h>

#include <ui/obj_manager.h>
#include <pd/algo_ctrl.h>
#include <ui/menus.h>
#include <ui/input_callbacks.h>
#include <ui/screen_capture_plugin.h>

#include <meshgen/mesh_generator.h>

#include <primitive/primitive.h>

#include <io/mesh_io.h>
#include <io/io_data.h>

#include <instancing/instantiator.h>

int main(int argc, char* argv[])
{
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

	// only 1 gizmo during simulation
	igl::opengl::glfw::imgui::ImGuizmoWidget gizmo;
	gizmo.visible = false;
	gizmo.operation = ImGuizmo::OPERATION::TRANSLATE;
	menu_plugin.widgets.push_back(&gizmo);

    // transformation aux
	std::unordered_map<pd::MeshIDType, Eigen::MatrixXd> obj_init_pos_map; // obj_id to its initial position matrix

    io::MeshIO& mesh_io = io::MeshIO::instance();

	// pd simulatee
	std::unordered_map<pd::MeshIDType, pd::DeformableMesh>& models = io::IOData::instance().models;
	std::unordered_map<pd::MeshIDType, std::unique_ptr<primitive::Primitive>>& rigid_colliders = io::IOData::instance().rigid_colliders;
	std::unordered_map<pd::MeshIDType, pd::DataMatrixX3>& f_exts = io::IOData::instance().f_exts;
	pd::UserControl& user_control = io::IOData::instance().user_control;

	// rigid body colliders
	pd::PhysicsParams& physics_params = io::IOData::instance().physics_params;
	pd::SolverParams& solver_params = io::IOData::instance().solver_params;

	static int total_n_constraints = 0;
	ui::ObjManager obj_manager{ viewer, gizmo, models, rigid_colliders, obj_init_pos_map, f_exts, user_control, solver_params, total_n_constraints };

	gizmo.callback = ui::gizmo_handler{ viewer, models, rigid_colliders, obj_manager, obj_init_pos_map, user_control, viewer.core().is_animating };
	viewer.callback_mouse_down = ui::mouse_down_handler{ models, user_control };
	viewer.callback_mouse_move = ui::mouse_move_handler{ models, user_control, physics_params, f_exts };
	viewer.callback_mouse_up = ui::mouse_up_handler{ user_control };
	viewer.callback_key_pressed = ui::keypress_handler{ gizmo, obj_manager, user_control };

	rendering::pre_draw_handler frame_callback{ models, rigid_colliders, physics_params, f_exts, solver_params, user_control };
	viewer.callback_pre_draw = frame_callback; // frame routine

	obj_menu.callback_draw_viewer_window = ui::obj_menu_window_handler{ viewer, obj_manager, user_control };
	constraint_menu.callback_draw_viewer_window = ui::constraint_menu_window_handler{ viewer, obj_manager, user_control, physics_params };
	component_menu.callback_draw_viewer_window = ui::component_menu_window_handler{ viewer };
	instantiator_menu.callback_draw_viewer_window = ui::instantiator_menu_window_handler{ viewer, obj_manager, physics_params };

	main_menu.callback_draw_viewer_window = ui::pd_menu_window_handler{ viewer, screen_capture_plugin, obj_manager, solver_params, physics_params, user_control, frame_callback, f_exts, gizmo };

    viewer.launch_init(false, "Projective Dynamics", 0, 0);
	viewer.launch_rendering(true);
	viewer.launch_shut();

	// If don't call this, Solver will be destructed as a static var,
	// the time that it destructed is beyond the cuda lib unloaded,
	// thus causing an error in cuda code
    pd::Solver::instance().free_cuda_memory();

	return 0;
}
