#include <array>
#include <filesystem>
#include <pd/constraint.h>
#include <pd/deformable_mesh.h>
#include <pd/solver.h>

#include <ui/obj_manager.h>
#include <ui/physics_params.h>
#include <ui/solver_params.h>
#include <ui/user_control.h>
#include <ui/callbacks.h>

#include <model/cloth.h>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuizmoWidget.h>

#include <util/gpu_helper.h>

int main(int argc, char* argv[])
{
	igl::opengl::glfw::Viewer viewer;
	igl::opengl::glfw::imgui::ImGuiPlugin menu_plugin;
	viewer.plugins.push_back(&menu_plugin);

	igl::opengl::glfw::imgui::ImGuiMenu main_menu;
	menu_plugin.widgets.push_back(&main_menu);
	igl::opengl::glfw::imgui::ImGuiMenu obj_menu;
	menu_plugin.widgets.push_back(&obj_menu);
	// Only 1 gizmo during simulation
	igl::opengl::glfw::imgui::ImGuizmoWidget gizmo;
	menu_plugin.widgets.push_back(&gizmo);

	std::unordered_map<int, Eigen::Matrix4f> obj_t_map; // obj_id to its transformation matrix

	// pd simulatee
	std::unordered_map<int, pd::DeformableMesh> models;
	std::unordered_map<int, Eigen::MatrixX3d> f_exts;

	pd::Solver solver(models);

	ui::UserControl user_control;
	ui::PhysicsParams physics_params;
	ui::SolverParams solver_params;

	static int total_n_constraints = 0;
	ui::ObjManager obj_manager{ viewer, gizmo, obj_t_map, solver, models, f_exts, user_control, solver_params, total_n_constraints };

	gizmo.callback = ui::gizmo_handler{ viewer, models, obj_t_map, user_control };
	viewer.callback_mouse_down = ui::mouse_down_handler{ models, user_control };
	viewer.callback_mouse_move = ui::mouse_move_handler{ models, user_control, physics_params, f_exts };
	viewer.callback_mouse_up = ui::mouse_up_handler{ user_control };
	ui::pre_draw_handler frame_callback{ solver, models, physics_params, f_exts, solver_params };
	viewer.callback_pre_draw = frame_callback; // frame routine
	viewer.callback_key_pressed = [&](igl::opengl::glfw::Viewer& viewer, int button, int modifier)
	{
		return false;
	};

	const auto HelpMarker = [](const char* desc)
	{
		ImGui::TextDisabled("(?)");
		constexpr auto ImGuiHoveredFlags_DelayShort = 1 << 12;
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
		{
			ImGui::BeginTooltip();
			ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
			ImGui::TextUnformatted(desc);
			ImGui::PopTextWrapPos();
			ImGui::EndTooltip();
		}
	};

	obj_menu.callback_draw_viewer_window = [&]()
	{
		ImGui::SetNextWindowPos(ImVec2(viewer.core().viewport(2) - 200, viewer.core().viewport(1)), ImGuiCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(200, viewer.core().viewport(3)), ImGuiCond_FirstUseEver);

		ImGui::Begin("Object Manager");

		static int cur_select_id = -1;
		if (ImGui::BeginListBox("objects", ImVec2(-FLT_MIN, 5 * ImGui::GetTextLineHeightWithSpacing())))
		{
			for (const auto& [id, model] : models)
			{
				const std::string model_name = std::string("Mesh ") + std::to_string(id);

				const bool is_selected = (cur_select_id == id);
				if (ImGui::Selectable(model_name.c_str(), is_selected))
				{
					cur_select_id = id;
					user_control.cur_sel_mesh_id = id;

					obj_manager.bind_gizmo(id);
				}
				if (is_selected)
				{
					ImGui::SetItemDefaultFocus();
				}
			}
			ImGui::EndListBox();
		}
		// viewer.data_list.erase()

		ImGui::Separator();

		if (ImGui::CollapsingHeader("Model Generator", ImGuiTreeNodeFlags_DefaultOpen))
		{
			// user select add model or reset a model
			constexpr int OP_ADD = 0;
			constexpr int OP_RESET = 1;
			static int add_or_reset = OP_ADD;
			ImGui::RadioButton("Add", &add_or_reset, OP_ADD); ImGui::SameLine();
			ImGui::RadioButton("Reset", &add_or_reset, OP_RESET); 

			ImGui::SetNextItemOpen(true); // Helpful for testing, can be removed later
			if (ImGui::TreeNode("Cloth"))
			{
				static int w = 20;
				static int h = 20;
				ImGui::InputInt("width", &w);
				ImGui::InputInt("height", &h);
				if (ImGui::Button("Generate"))
				{
					auto [V, F] = model::generate_cloth(w, h);
					if (add_or_reset == OP_ADD)
					{
						obj_manager.add_model(V, F, F);
					}
					if (add_or_reset == OP_RESET)
					{
						obj_manager.reset_model(user_control.cur_sel_mesh_id, V, F, F);
					}
				}
				
				ImGui::TreePop();
			}
			if (ImGui::TreeNode(".obj File"))
			{
				if (ImGui::Button("Load .obj file"))
				{
					std::string file_name = igl::file_dialog_open();
					std::filesystem::path obj_file{ file_name };

					if (std::filesystem::exists(obj_file) && std::filesystem::is_regular_file(obj_file))
					{
						Eigen::MatrixXd V;
						Eigen::MatrixXi F;
						bool flag = igl::read_triangle_mesh(file_name, V, F);
						if (flag)
						{
							if (add_or_reset == OP_ADD)
							{
								obj_manager.add_model(V, F, F);
							}
							if (add_or_reset == OP_RESET)
							{
								obj_manager.reset_model(user_control.cur_sel_mesh_id, V, F, F);
							}
						}
						else
						{
							printf("Load .obj file failed!\n");
						}
					}
					ImGui::TreePop();
				}

				ImGui::TreePop();
			}
		}
		if (ImGui::CollapsingHeader("Constraint Setting", ImGuiTreeNodeFlags_DefaultOpen))
		{
			static bool enable_edge_length_constraint = false;

			static bool enable_positional_constraint = false;

			ImGui::SetNextItemOpen(true); // can be removed later
			if (ImGui::TreeNode("Edge Length"))
			{
				// Edge length params
				ImGui::InputFloat("wi", &physics_params.edge_length_constraint_wi, 1.f, 10.f, "%.1f");
				ImGui::Checkbox("Enable", &enable_edge_length_constraint);
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Positional"))
			{
				HelpMarker("Shift + LMC to toggle fix/unfix to a vertex.");
				std::string vertices_to_be_toggled = "";
				for (const int vi : user_control.toggle_fixed_vertex_idxs)
				{
					vertices_to_be_toggled += std::to_string(vi) + " ";
				}
				ImGui::Text("For mesh %d, Vertex indices to be toggled: %s", user_control.cur_sel_mesh_id, vertices_to_be_toggled.c_str());
				// Pinned 
				ImGui::InputFloat("wi", &physics_params.positional_constraint_wi, 10.f, 100.f, "%.1f");
				ImGui::Checkbox("Enable", &enable_positional_constraint);
				ImGui::TreePop();
			}

			if (ImGui::Button("Apply Constraints") && models.empty() == false)
			{
				pd::DeformableMesh& model = models[user_control.cur_sel_mesh_id];

				model.reset_constraints();
				solver.dirty = true;

				if (enable_edge_length_constraint)
				{
					model.set_edge_length_constraint(physics_params.edge_length_constraint_wi);
				}
				if (enable_positional_constraint && user_control.toggle_vertex_fix)
				{
					model.toggle_vertices_fixed(
						user_control.toggle_fixed_vertex_idxs, 
						physics_params.positional_constraint_wi, 
						physics_params.mass_per_vertex
					);
					user_control.toggle_vertex_fix = false;
					user_control.toggle_fixed_vertex_idxs.clear();
				}

				obj_manager.recalc_total_n_constraints();
			}
			ImGui::Text("#Constraints = %d", total_n_constraints);
		}

		ImGui::End();
	};

	main_menu.callback_draw_viewer_window = [&]() 
	{
		ImGui::SetNextWindowPos(ImVec2(viewer.core().viewport(0), viewer.core().viewport(1)), ImGuiCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(300, viewer.core().viewport(3)), ImGuiCond_FirstUseEver);
		ImGui::Begin("PD Panel");
		// ImGui::Text("viewer.core().viewport = %f, %f, %f, %f\n", 
		// 	viewer.core().viewport(0),  // 0
		// 	viewer.core().viewport(1),  // 0
		// 	viewer.core().viewport(2),  // 1280
		// 	viewer.core().viewport(3)   // 800
		// );

		if (ImGui::CollapsingHeader("Physics Setting"), ImGuiTreeNodeFlags_DefaultOpen)
		{
			// physics params
			ImGui::Checkbox("Enable Gravity", &physics_params.enable_gravity);

			ImGui::InputFloat("mass per vertex", &physics_params.mass_per_vertex, 1, 10, "%.1f");
		}
		if (ImGui::CollapsingHeader("Picking Setting"), ImGuiTreeNodeFlags_DefaultOpen)
		{
			HelpMarker("Ctrl + LMC to apply external force to the model.");
			ImGui::Text("State: %s", user_control.apply_ext_force ? "Applying" : "Not applying");
			if (user_control.apply_ext_force)
			{
				ImGui::Text("Vertex forced: %d", user_control.ext_forced_vertex_idx);
			}

			ImGui::InputFloat("dragging force", &physics_params.external_force_val, 1.f, 10.f, "%.3f");
		}
		if (ImGui::CollapsingHeader("Visualization Setting"))
		{
			ImGui::Checkbox("Wireframe", [&]() { 
					const int idx = viewer.mesh_index(user_control.cur_sel_mesh_id);
					return viewer.data_list[idx].show_lines != 0; 
				},
				[&](bool value) {
					const int idx = viewer.mesh_index(user_control.cur_sel_mesh_id);
					viewer.data_list[idx].show_lines = value;
				}
			);
			ImGui::InputFloat("Point Size", &viewer.data_list[user_control.cur_sel_mesh_id].point_size, 1.f, 10.f);
			// ImGui::Text("#Vertex = %d", model.positions().rows());
			// ImGui::Text("#Face = %d", model.faces().rows());
			// ImGui::Text("#DOF = %d", model.positions().rows() * 3 - model.n_edges);
		}

		if (ImGui::CollapsingHeader("Simulating Control"), ImGuiTreeNodeFlags_DefaultOpen)
		{
			ImGui::Text("Solver is %s", solver.dirty ? "not ready." : "ready.");

			ImGui::Checkbox("Use GPU for local step", &solver_params.use_gpu_for_local_step);
			ImGui::InputFloat("timestep", &solver_params.dt, 0.01f, 0.1f, "%.4f"); // TODO: verify time step
			ImGui::InputInt("solver #itr", &solver_params.n_itr_solver_iterations);
			ImGui::InputInt("PD #itr", &solver_params.n_solver_pd_iterations);

			// Solver Selector
			const char* items[] = {"Direct", "Parallel Jacobi", "A-Jacobi-1", "A-Jacobi-2", "A-Jacobi-3" };
			static const char* cur_select_item = "Direct";
			if (ImGui::BeginCombo("cur solver", cur_select_item))
			{
				for (int i = 0; i < IM_ARRAYSIZE(items); i++)
				{
					bool is_selected = (cur_select_item == items[i]);
					if (ImGui::Selectable(items[i], is_selected))
					{
						// If solver changed
						if (solver_params.selected_solver != static_cast<ui::LinearSysSolver>(i))
						{
							solver.algo_changed = true;
							solver.dirty = true;
						}

						// change solver
						cur_select_item = items[i];
						solver_params.selected_solver = static_cast<ui::LinearSysSolver>(i);
					}
					if (is_selected)
					{
						ImGui::SetItemDefaultFocus();
					}
				}
				ImGui::EndCombo();
			}

			ImGui::Separator();
			// statistics
			if (viewer.core().is_animating)
			{
				ImGui::Text("FPS = %lf", 1000.0 / frame_callback.last_elapse_time);
				ImGui::Text("Last frame time elapsed: %lf ms", frame_callback.last_elapse_time);
			}
			//ImGui::Text("Time for 1 tick: %lf ms", frame_callback.last_elapse_time);
			ImGui::Text("Time for global step: %lf ms", frame_callback.last_global_step_time);
			ImGui::Text("Time for local step: %lf ms", frame_callback.last_local_step_time);
			ImGui::Text("Time for precomputation: %lf ms", frame_callback.last_precomputation_time);

			ImGui::Separator();

			ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0, 0.6f, 0.6f));
			ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0, 0.7f, 0.7f));
			ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0, 0.8f, 0.8f));
			if (ImGui::Button("Simulate 1 Step") && viewer.core().is_animating == false)
			{
				ui::tick(viewer, models, physics_params, solver_params, solver, f_exts);
			}
			ImGui::PopStyleColor(3);

			ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0x66, 0xCC, 0xFF, 0xFF));
			ImGui::Checkbox("Auto Simulate!", &viewer.core().is_animating);
			ImGui::PopStyleColor();
		}

		ImGui::End();
	};

	viewer.launch();

	return 0;
}

