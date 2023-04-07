#include <array>
#include <pd/constraint.h>
#include <pd/deformable_mesh.h>
#include <pd/solver.h>
#include <pd/tick.h>

#include <ui/obj_manager.h>
#include <ui/physics_params.h>
#include <ui/solver_params.h>
#include <ui/user_control.h>
#include <ui/callbacks.h>
#include <ui/menus.h>

#include <meshgen/mesh_generator.h>

#include <primitive/primitive.h>
#include <primitive/floor.h>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuizmoWidget.h>

#include <instancing/instantiator.h>

#include <util/gpu_helper.h>

int main(int argc, char* argv[])
{
	// TODO: hide igl::opengl::glfw::Viewer usage
	igl::opengl::glfw::Viewer viewer;
	igl::opengl::glfw::imgui::ImGuiPlugin menu_plugin;
	viewer.plugins.push_back(&menu_plugin);

	igl::opengl::glfw::imgui::ImGuiMenu main_menu;
	menu_plugin.widgets.push_back(&main_menu);
	igl::opengl::glfw::imgui::ImGuiMenu obj_menu;
	menu_plugin.widgets.push_back(&obj_menu);
	igl::opengl::glfw::imgui::ImGuiMenu instantiator_menu;
	menu_plugin.widgets.push_back(&instantiator_menu);

	// Only 1 gizmo during simulation
	igl::opengl::glfw::imgui::ImGuizmoWidget gizmo;
	gizmo.visible = false;
	gizmo.operation = ImGuizmo::OPERATION::TRANSLATE;
	menu_plugin.widgets.push_back(&gizmo);

    // transformation aux
	std::unordered_map<int, Eigen::MatrixXd> obj_init_pos_map; // obj_id to its initial position matrix

	// pd simulatee
	std::unordered_map<int, pd::DeformableMesh> models;
	std::unordered_map<int, Eigen::MatrixX3d> f_exts;

	// rigid body colliders
	std::unordered_map<int, std::unique_ptr<primitive::Primitive>> rigid_colliders;

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
	viewer.callback_key_pressed = ui::keypress_handler{ gizmo };

	pd::pre_draw_handler frame_callback{ solver, models, physics_params, f_exts, solver_params, user_control, always_recompute_normal };
	viewer.callback_pre_draw = frame_callback; // frame routine

	instancing::Instantiator instantiator { obj_manager, models };

	const float window_widths[] = {200, 120, 300};
	const float& viewport_horizontal_start = viewer.core().viewport(0);
	const float& viewport_horizontal_end = viewer.core().viewport(2);
	const float& viewport_vertical_start = viewer.core().viewport(1);
	const float& viewport_vertical_end = viewer.core().viewport(3);
	obj_menu.callback_draw_viewer_window = [&]()
	{
		ImGui::SetNextWindowPos(ImVec2(viewport_horizontal_end - window_widths[0], viewport_vertical_start));
		ImGui::SetNextWindowSize(ImVec2(window_widths[0], viewport_vertical_end));

		ImGui::Begin("Object Manager");

		ui::mesh_manager_menu("Deformable", models, obj_manager, user_control, viewer.core().is_animating);
		ui::mesh_manager_menu("Collider", rigid_colliders, obj_manager, user_control, viewer.core().is_animating);

		ui::mesh_remove_menu(obj_manager, user_control.cur_sel_mesh_id);

		ImGui::Separator();

		ui::deformable_mesh_generate_menu(obj_manager, user_control.cur_sel_mesh_id);
		ui::collider_generate_menu(obj_manager, user_control.cur_sel_mesh_id);

		ui::set_constraints_menu(obj_manager, physics_params, user_control);

		ImGui::End();
	};

	instantiator_menu.callback_draw_viewer_window = [&]()
	{
		constexpr float window_height = 80;
		ImGui::SetNextWindowPos(ImVec2(viewport_horizontal_end - window_widths[0] - window_widths[1], viewport_vertical_start));
		ImGui::SetNextWindowSize(ImVec2(window_widths[1], window_height));

		ImGui::Begin("Instanciator");

		ui::instantiator_menu(instantiator);

		ImGui::End();
	};

	main_menu.callback_draw_viewer_window = [&]() 
	{
		ImGui::SetNextWindowPos(ImVec2(viewport_horizontal_start, viewport_vertical_start));
		ImGui::SetNextWindowSize(ImVec2(window_widths[2], viewport_vertical_end));

		ImGui::Begin("PD Panel");

		ui::physics_menu(physics_params, user_control);
		ui::visualization_menu(viewer, models, always_recompute_normal, user_control.cur_sel_mesh_id);

		ui::simulation_ctrl_menu(solver, solver_params, physics_params, viewer, frame_callback, models, f_exts, always_recompute_normal);

		ImGui::End();
	};

	// use only for testing
	[&]()
	{
		instantiator.instance_floor();
		// instantiator.instance_cylinder();
		// instantiator.instance_test();
	}();

	viewer.launch(true, false, "Projective Dynamics", 0, 0);

	return 0;
}