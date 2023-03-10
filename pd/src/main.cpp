#include <array>
#include <filesystem>
#include <pd/constraint.h>
#include <pd/deformable_mesh.h>
#include <pd/solver.h>

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

static void print_cuda_info()
{
	int n_gpu_devs;
	int best_dev_idx = util::select_best_device(n_gpu_devs);
	printf("Available #dev = %d\n", n_gpu_devs);
	util::test_device(best_dev_idx);
}

static void HelpMarker(const char* desc)
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
}

int main(int argc, char* argv[])
{
	igl::opengl::glfw::Viewer viewer;
	igl::opengl::glfw::imgui::ImGuiPlugin menu_plugin;
	viewer.plugins.push_back(&menu_plugin);

	igl::opengl::glfw::imgui::ImGuiMenu main_menu;
	menu_plugin.widgets.push_back(&main_menu);
	igl::opengl::glfw::imgui::ImGuiMenu obj_menu;
	menu_plugin.widgets.push_back(&obj_menu);

	std::unordered_map<int, Eigen::Matrix4f> obj_t_map; // obj_id to its transformation matrix
	std::unordered_map<int, int> obj_idx_map; // obj_id to idx in visualzed meshes list map

	// pd simulatee
	std::unordered_map<int, pd::DeformableMesh> models;
	// pd::DeformableMesh model;
	std::unordered_map<int, Eigen::MatrixX3d> f_exts;
	// Eigen::MatrixX3d f_ext; // external force

	// TODO: modify solver to execute all deformable mesh
	pd::Solver solver(models);

	ui::UserControl user_control;
	ui::PhysicsParams physics_params;
	ui::SolverParams solver_params;

	viewer.callback_mouse_down = ui::mouse_down_handler{ models, user_control };
	viewer.callback_mouse_move = ui::mouse_move_handler{ models, user_control, physics_params, f_exts };
	viewer.callback_mouse_up = ui::mouse_up_handler{ user_control };
	ui::pre_draw_handler frame_callback{ solver, models, physics_params, f_exts, solver_params };
	viewer.callback_pre_draw = frame_callback; // frame routine

	const auto rescale = [&](Eigen::MatrixXd& V)
	{
		// rescale the vertices to make all models look equal in size
		Eigen::RowVector3d v_mean = V.colwise().mean();
		V.rowwise() -= v_mean;
		V.array() /= (V.maxCoeff() - V.minCoeff());
	};

	static int total_n_constraints = 0;
	const auto recalc_total_n_constraints = [&]()
	{
		total_n_constraints = 0;
		for (const auto& [_, model] : models)
		{
			total_n_constraints += model.n_constraints();
		}
	};

	// Only 1 gizmo during simulation
	igl::opengl::glfw::imgui::ImGuizmoWidget gizmo;
	menu_plugin.widgets.push_back(&gizmo);
	gizmo.callback = [&](const Eigen::Matrix4f& T)
	{
		// const Eigen::Matrix4d& T0 = 
		// const Eigen::Matrix4d TT = (T * T0.inverse()).cast<double>().transpose();
		// const Eigen::MatrixXd positions = (model.positions().rowwise().homogeneous() * TT).rowwise().hnormalized();
		// model.set_positions(positions);
		// viewer.data().set_vertices(positions);
		// viewer.data().compute_normals();
	};

	// Bind the gizmo to a new mesh when needed.
	const auto bind_gizmo = [&](int obj_id) 
	{

	};

	const auto add_model = [&](Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXi& E)
	{
		rescale(V);

		// create a new mesh
		int obj_id = viewer.append_mesh();
		models.emplace(obj_id, pd::DeformableMesh(V, F, E, obj_id));
		pd::DeformableMesh& model = models[obj_id];

		// reset f_ext 
		f_exts[obj_id].resizeLike(model.positions()); // let external forces add to vertices in the new model
		f_exts[obj_id].setZero();

		// reset viewer
		int idx = viewer.mesh_index(obj_id);
		viewer.data_list[idx].set_mesh(model.positions(), model.faces());
		const Eigen::RowVector3d TEXTURE_COLOR = Eigen::RowVector3d((double)0x66 / 0xff, (double)0xcc / 0xff, 1.0);
		viewer.data_list[idx].set_colors(TEXTURE_COLOR);
		viewer.core().align_camera_center(model.positions());
		viewer.data_list[idx].point_size = 10.f;

		// if this is the only model, select it
		if (models.size() == 1)
		{
			user_control.cur_sel_mesh_id = obj_id;
		}
	};

	const auto remove_model = [&](int obj_id)
	{

	};

	const auto reset_model = [&](int obj_id, Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXi& E)
	{
		// rescale the vertices to make all models look equal in size
		rescale(V);

		// reset to a new mesh
		pd::DeformableMesh& model = models[obj_id];
		model = pd::DeformableMesh(V, F, E, obj_id);

		// reset f_ext 
		f_exts[obj_id].resizeLike(model.positions()); // let external forces add to vertices in the new model
		f_exts[obj_id].setZero();

		// reset viewer
		int idx = viewer.mesh_index(obj_id);
		viewer.data_list[idx].clear();
		viewer.data_list[idx].set_mesh(model.positions(), model.faces());
		const Eigen::RowVector3d TEXTURE_COLOR = Eigen::RowVector3d((double)0x66 / 0xff, (double)0xcc / 0xff, 1.0);
		viewer.data_list[idx].set_colors(TEXTURE_COLOR);
		viewer.core().align_camera_center(model.positions());

		recalc_total_n_constraints();
	};

	obj_menu.callback_draw_viewer_window = [&]()
	{
		ImGui::SetNextWindowPos(ImVec2(viewer.core().viewport(2) - 200, viewer.core().viewport(1)), ImGuiCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(200, viewer.core().viewport(3)), ImGuiCond_FirstUseEver);

		ImGui::Begin("Object Manager");

		// static const char* cur_select_model_name = "";
		static int cur_select_id = -1;
		if (ImGui::BeginListBox("objects", ImVec2(-FLT_MIN, 5 * ImGui::GetTextLineHeightWithSpacing())))
		{
			for (const auto& [id, model] : models)
			{
				const char* model_name = (std::string("Mesh ") + std::to_string(id)).c_str();

				const bool is_selected = (cur_select_id == id);
				if (ImGui::Selectable(model_name, is_selected))
				{
					cur_select_id = id;
					user_control.cur_sel_mesh_id = id;

					bind_gizmo(id);
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
						add_model(V, F, F);
					}
					if (add_or_reset == OP_RESET)
					{
						reset_model(user_control.cur_sel_mesh_id, V, F, F);
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
								add_model(V, F, F);
							}
							if (add_or_reset == OP_RESET)
							{
								reset_model(user_control.cur_sel_mesh_id, V, F, F);
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

				recalc_total_n_constraints();
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
			ImGui::InputFloat("timestep", &solver_params.dt, 0.01f, 0.1f, "%.4f"); // TODO: verify time step

			ImGui::Separator();

			ImGui::Checkbox("Use GPU for local step", &solver_params.use_gpu_for_local_step);

			ImGui::Separator();

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
			ImGui::InputInt("solver #itr", &solver_params.n_itr_solver_iterations);

			ImGui::Separator();
			// statistics
			if (viewer.core().is_animating)
			{
				ImGui::Text("FPS = %lf", 1000.0 / frame_callback.last_elapse_time);
				ImGui::Text("Last frame time elapsed: %lf ms", frame_callback.last_elapse_time);
			}
			ImGui::Text("Time for global step: %lf ms", frame_callback.last_global_step_time);
			ImGui::Text("Time for local step: %lf ms", frame_callback.last_local_step_time);
			ImGui::Text("Time for precomputation: %lf ms", frame_callback.last_precomputation_time);

			ImGui::Separator();

			ImGui::InputInt("PD #itr", &solver_params.n_solver_pd_iterations);

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

