#include <ui/menus.h>

#include <filesystem>

#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuizmoWidget.h>

#include <igl/png/writePNG.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>

#include <ui/input_callbacks.h>

#include <primitive/primitive.h>

#include <meshgen/mesh_generator.h>

namespace ui {
	std::string label_prefix(const char* label)
	{
		float width = ImGui::CalcItemWidth();

		float x = ImGui::GetCursorPosX();
		ImGui::Text(label); 
		ImGui::SameLine(); 
		ImGui::SetCursorPosX(x + width * 0.5f + ImGui::GetStyle().ItemInnerSpacing.x);
		ImGui::SetNextItemWidth(-1);

		std::string label_id = "##";
		label_id += label;

		return label_id;
	}
	#define LABEL(x) label_prefix(x).c_str()

    void help_marker(const char* desc)
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

	void set_window_position_size(const Eigen::Vector4f& viewport_vec, WindowPositionSize wps)
	{
		const float& viewport_horizontal_start = viewport_vec(0);
		const float& viewport_horizontal_end = viewport_vec(2);
		const float& viewport_vertical_start = viewport_vec(1);
		const float& viewport_vertical_end = viewport_vec(3);

		float pos_x = wps.pos_x < 0 ? viewport_horizontal_end + wps.pos_x : wps.pos_x;
		float pos_y = wps.pos_y < 0 ? viewport_vertical_end + wps.pos_y : wps.pos_y;
		float size_x = wps.size_x < 0 ? viewport_horizontal_end + wps.size_x : wps.size_x;
		float size_y = wps.size_y < 0 ? viewport_vertical_end + wps.size_y: wps.size_y;
		ImGui::SetNextWindowPos(ImVec2(pos_x, pos_y));
		ImGui::SetNextWindowSize(ImVec2(size_x, size_y));
	}

	void obj_menu_window_handler::operator()()
	{
		set_window_position_size(viewer.core().viewport, obj_menu_wps);
		ImGui::Begin("Object Manager");

		ui::mesh_manager_menu("Deformable", obj_manager.models, obj_manager, user_control, viewer.core().is_animating);
		ui::mesh_manager_menu("Collider", obj_manager.rigid_colliders, obj_manager, user_control, viewer.core().is_animating);

		ui::mesh_remove_menu(obj_manager, user_control.cur_sel_mesh_id);

		ImGui::Separator();

		ui::deformable_mesh_generate_menu(obj_manager, user_control.cur_sel_mesh_id);
		ui::collider_generate_menu(obj_manager, user_control.cur_sel_mesh_id);

		ImGui::End();
	}

	void constraint_menu_window_handler::operator()()
	{
		set_window_position_size(viewer.core().viewport, constraint_menu_wps);
		if (obj_manager.is_deformable_model(user_control.cur_sel_mesh_id) == false)
		{
			return;
		}

		ImGui::Begin("Constraint Setter");

		ui::set_constraints_menu(obj_manager, physics_params, user_control);

		ImGui::End();
	}

	void instantiator_menu_window_handler::operator()()
	{
		set_window_position_size(viewer.core().viewport, instantiator_menu_wps);
		ImGui::Begin("Instantiator");

		instancing::Instantiator instantiator{ obj_manager, physics_params };
		ui::instantiator_menu(instantiator);

		ImGui::End();
	};

	void component_menu_window_handler::operator()()
	{
		return; 

		// not implemented yet
		set_window_position_size(viewer.core().viewport, component_menu_wps);
		ImGui::Begin("Component");

		ImGui::End();		
	}

	void pd_menu_window_handler::operator()()
	{
		set_window_position_size(viewer.core().viewport, pd_menu_wps);
		ImGui::Begin("PD Panel");

		ui::physics_menu(physics_params, user_control);
		ui::visualization_menu(viewer, screen_capture_plugin, obj_manager.models, always_recompute_normal, user_control.cur_sel_mesh_id);

		ui::simulation_ctrl_menu(
			solver, 
			obj_manager,
			user_control,
			solver_params, 
			physics_params, 
			viewer, 
			frame_callback, 
			f_exts, 
			gizmo, 
			always_recompute_normal
		);

		ImGui::End();
	}

	template<typename T>
	void mesh_manager_menu(
		const std::string& obj_prompts, 
		const std::unordered_map<int, T>& meshes,
		ObjManager& obj_manager, 
		UserControl& user_control, 
		bool is_animating
	) 
	{
		const std::string title = obj_prompts + " objects";
		if (ImGui::BeginListBox(title.c_str(), ImVec2(-FLT_MIN, 5 * ImGui::GetTextLineHeightWithSpacing())))
		{
			for (const auto& [id, mesh] : meshes)
			{
				const std::string model_name = obj_prompts + " " + std::to_string(id);

				const bool is_selected = (user_control.cur_sel_mesh_id == id);
				if (ImGui::Selectable(model_name.c_str(), is_selected))
				{
					user_control.cur_sel_mesh_id = id;
					user_control.selected_vertex_idx = 0;
				}
				if (is_selected)
				{
					if ((obj_manager.is_deformable_model(id) && is_animating) == false)
					{
						obj_manager.bind_gizmo(id);  // To rebind gizmo in animating causes weird behavior sometimes 
					}
					ImGui::SetItemDefaultFocus();
				}
			}
			ImGui::EndListBox();
		}
	}
	// instantiate used template explicitly to avoid linker error and improve compiling speed
	template void mesh_manager_menu<pd::DeformableMesh>(const std::string& obj_prompts, const std::unordered_map<int, pd::DeformableMesh>& meshes, ObjManager& obj_manager, UserControl& user_control, bool is_animating);
	template void mesh_manager_menu<std::unique_ptr<primitive::Primitive>>(const std::string& obj_prompts, const std::unordered_map<int, std::unique_ptr<primitive::Primitive>>& meshes, ObjManager& obj_manager, UserControl& user_control, bool is_animating);

	void mesh_remove_menu(ObjManager& obj_manager, int id)
	{
		ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0, 0.6f, 0.6f));
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0, 0.7f, 0.7f));
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0, 0.8f, 0.8f));
		if (ImGui::Button("Remove selected mesh"))
		{
			if (obj_manager.is_deformable_model(id))
			{
		   		obj_manager.remove_model(id);
			}
			else if (obj_manager.is_rigid_collider(id))
			{
				obj_manager.remove_rigid_collider(id);
			}
			else 
			{
				printf("Error: Invalid mesh to remove\n!");
			}
		}
		ImGui::PopStyleColor(3);
	}

    void deformable_mesh_generate_menu(ObjManager& obj_manager, int id)
    {
        if (ImGui::CollapsingHeader("Model Generator", ImGuiTreeNodeFlags_DefaultOpen))
		{
			constexpr int OP_ADD = 0;
			constexpr int OP_RESET = 1;
			static int add_or_reset = OP_ADD;
			// user select add model or reset a model
			if (obj_manager.is_deformable_model(id))
			{
				ImGui::RadioButton("Add", &add_or_reset, OP_ADD); ImGui::SameLine();
				ImGui::RadioButton("Reset", &add_or_reset, OP_RESET); 
			}

			if (ImGui::TreeNode("Cloth"))
			{
				static int w = 20;
				static int h = 20;
				ImGui::InputInt(LABEL("width"), &w);
				ImGui::InputInt(LABEL("height"), &h);
				if (ImGui::Button("Generate"))
				{
					auto [V, F] = meshgen::generate_cloth(w, h);
					if (add_or_reset == OP_ADD)
					{
						obj_manager.add_model(V, F);
					}
					if (add_or_reset == OP_RESET)
					{
						obj_manager.reset_model(id, V, F);
					}
				}
				
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Hemisphere shell"))
			{
				static float radius = 1.0f;
				ImGui::InputFloat(LABEL("radius"), &radius);
				if (ImGui::Button("Generate"))
				{
					auto [V, F] = meshgen::generate_hemisphere(radius);
					if (add_or_reset == OP_ADD)
					{
						obj_manager.add_model(V, F);
					}
					if (add_or_reset == OP_RESET)
					{
						obj_manager.reset_model(id, V, F);
					}
				}
				
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Sphere shell"))
			{
				static float radius = 0.5f;
				ImGui::InputFloat(LABEL("radius"), &radius);
				if (ImGui::Button("Generate"))
				{
					auto [V, F] = meshgen::generate_sphere(radius);
					if (add_or_reset == OP_ADD)
					{
						obj_manager.add_model(V, F);
					}
					if (add_or_reset == OP_RESET)
					{
						obj_manager.reset_model(id, V, F);
					}
				}
				ImGui::TreePop();
			}		
			if (ImGui::TreeNode("Cylinder shell"))
			{
				static float radius = 0.5f;
				static float height = 1.2f;
				ImGui::InputFloat(LABEL("radius"), &radius);
				ImGui::InputFloat(LABEL("height"), &height);
				if (ImGui::Button("Generate"))
				{
					auto [V, F] = meshgen::generate_cylinder(radius, height);
					if (add_or_reset == OP_ADD)
					{
						obj_manager.add_model(V, F);
					}
					if (add_or_reset == OP_RESET)
					{
						obj_manager.reset_model(id, V, F);
					}
				}
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Cone shell"))
			{
				static float radius = 0.5f;
				static float height = 1.2f;
				ImGui::InputFloat(LABEL("radius"), &radius);
				ImGui::InputFloat(LABEL("height"), &height);
				if (ImGui::Button("Generate"))
				{
					auto [V, F] = meshgen::generate_cone(radius, height);
					if (add_or_reset == OP_ADD)
					{
						obj_manager.add_model(V, F);
					}
					if (add_or_reset == OP_RESET)
					{
						obj_manager.reset_model(id, V, F);
					}
				}
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Torus shell"))
			{
				static float main_radius = 1.2f;
				static float ring_radius = 0.4f;
				ImGui::InputFloat(LABEL("main radius"), &main_radius);
				ImGui::InputFloat(LABEL("ring radius"), &ring_radius);
				if (ImGui::Button("Generate"))
				{
					auto [V, F] = meshgen::generate_torus(main_radius, ring_radius);
					if (add_or_reset == OP_ADD)
					{
						obj_manager.add_model(V, F);
					}
					if (add_or_reset == OP_RESET)
					{
						obj_manager.reset_model(id, V, F);
					}
				}
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Bar"))
			{
				static int w = 3;
				static int h = 4;
				static int d = 5;
				ImGui::InputInt(LABEL("width"), &w);
				ImGui::InputInt(LABEL("height"), &h);
				ImGui::InputInt(LABEL("depth"), &d);
				if (ImGui::Button("Generate"))
				{
					auto [V, T, boundary_facets] = meshgen::generate_bar(w, h, d);
					if (add_or_reset == OP_ADD)
					{
						obj_manager.add_model(V, T, boundary_facets);
					}
					if (add_or_reset == OP_RESET)
					{
						obj_manager.reset_model(id, V, T, boundary_facets);
					}
				}
				
				ImGui::TreePop();
			}
			if (ImGui::TreeNode(".obj File"))
			{
				static bool tetrahedrize = false;
				ImGui::Checkbox("Tetrahedrize", &tetrahedrize);
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
							if (tetrahedrize == true)
							{
								Eigen::MatrixXd TV;
								Eigen::MatrixXi TT;
								Eigen::MatrixXi TF;
								// tetgen
								igl::copyleft::tetgen::tetrahedralize(V, F, "pq1.414Y", TV, TT, TF);
								if (add_or_reset == OP_ADD)
								{
									obj_manager.add_model(TV, TT, F);
								}
								if (add_or_reset == OP_RESET)
								{
									obj_manager.reset_model(id, TV, TT, F);
								}
							}

							if (add_or_reset == OP_ADD)
							{
								obj_manager.add_model(V, F);
							}
							if (add_or_reset == OP_RESET)
							{
								obj_manager.reset_model(id, V, F);
							}
						}
						else
						{
							printf("Load .obj file failed!\n");
						}
					}
				}
				if (ImGui::Button("Load Armadillo")) // TOOD: For test only, must be removed later
				{
					Eigen::MatrixXd V;
					Eigen::MatrixXi F;
					igl::read_triangle_mesh("../assets/meshes/armadillo.obj", V, F);
					Eigen::MatrixXd TV;
					Eigen::MatrixXi TT;
					Eigen::MatrixXi TF;
					// tetgen
					igl::copyleft::tetgen::tetrahedralize(V, F, "pq1.414Y", TV, TT, TF);
					if (add_or_reset == OP_ADD)
					{
						obj_manager.add_model(TV, TT, F);
					}
					if (add_or_reset == OP_RESET)
					{
						obj_manager.reset_model(id, TV, TT, F);
					}
				}

				ImGui::TreePop();
			}
		}
    }

    void collider_generate_menu(ObjManager& obj_manager, int id)
    {
        if (ImGui::CollapsingHeader("Static Object Generator", ImGuiTreeNodeFlags_DefaultOpen))
		{
			if (ImGui::TreeNode("Floor"))
			{
				static float y = -1;
				ImGui::InputFloat(LABEL("y"), &y);

				if (ImGui::Button("Generate"))
				{
					obj_manager.add_rigid_collider(std::make_unique<primitive::Floor>(y));
				}

				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Sphere"))
			{
				static float center_radius[4] = { 0.2f, 0.2f, 0.2f, 0.2f };
            	ImGui::InputFloat3(LABEL("center"), center_radius, "%.1f");
				ImGui::InputFloat(LABEL("radius"), &center_radius[3]);

				if (ImGui::Button("Generate"))
				{
					obj_manager.add_rigid_collider(std::make_unique<primitive::Sphere>(
						Eigen::Vector3f(center_radius).cast<pd::SimScalar>(),
						center_radius[3]
					));
				}

				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Torus"))
			{
				static float center_radius[5] = { 0, 0, 0, 0.5f, 0.1f };
				ImGui::InputFloat3(LABEL("center"), center_radius, "%.1f");
				ImGui::InputFloat(LABEL("main radius"), &center_radius[3]);
				ImGui::InputFloat(LABEL("ring radius"), &center_radius[4]);
				if (ImGui::Button("Generate"))
				{
					obj_manager.add_rigid_collider(std::make_unique<primitive::Torus>(
						Eigen::Vector3f(center_radius).cast<pd::SimScalar>(),
						center_radius[3],
						center_radius[4]
					));
				}
				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Block"))
			{
				static float center[3] = { 0.5f, 0.5f, 0.5f };
				static float xyz[3] = { 0.4f, 0.5f, 0.6f };
            	ImGui::InputFloat3(LABEL("center"), center, "%.1f");
            	ImGui::InputFloat3(LABEL("xyz"), xyz, "%.1f");

				if (ImGui::Button("Generate"))
				{
					obj_manager.add_rigid_collider(std::make_unique<primitive::Block>(
						Eigen::Vector3f(center).cast<pd::SimScalar>(),
						Eigen::Vector3f(xyz).cast<pd::SimScalar>()
					));
				}

				ImGui::TreePop();
			}
		}
    }

    void set_constraints_menu(ObjManager& obj_manager, PhysicsParams& physics_params, UserControl& user_control)
    {
		static bool enable_edge_strain_constraint = false;
		static bool enable_bending_constraint = false;
		static bool enable_tet_strain_constraint = false;
		static bool enable_positional_constraint = false;

		if (ImGui::TreeNode("Edge Strain"))
		{
			// Edge length params
			ImGui::InputFloat(LABEL("weight"), &physics_params.edge_strain_constraint_wc, 1.f, 10.f, "%.1f");
			ImGui::Checkbox("Enable", &enable_edge_strain_constraint);
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Bending"))
		{
			ImGui::InputFloat(LABEL("weight"), &physics_params.bending_constraint_wc, 1e-9f, 1e-5f, "%.9f");
			ImGui::Checkbox("Enable", &enable_bending_constraint);
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Tet Strain"))
		{
			ImGui::InputFloat(LABEL("weight"), &physics_params.tet_strain_constraint_wc, 1.f, 10.f, "%.3f");
			ImGui::InputFloat3(LABEL("strain min xyz"), physics_params.tet_strain_constraint_min_xyz.data(), "%.2f");
			ImGui::InputFloat3(LABEL("strain max xyz"), physics_params.tet_strain_constraint_max_xyz.data(), "%.2f");
			ImGui::Checkbox("Enable", &enable_tet_strain_constraint);
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Positional"))
		{
			help_marker("Shift + LMC to toggle fix/unfix to a vertex.");
			std::string vertices_to_be_toggled = "";
			for (const pd::VertexIndexType vi : user_control.toggle_fixed_vertex_idxs)
			{
				vertices_to_be_toggled += std::to_string(vi) + " ";
			}
			ImGui::TextWrapped("For mesh %d, Vertex indices to be toggled: %s", user_control.cur_sel_mesh_id, vertices_to_be_toggled.c_str());
			// Pinned 
			ImGui::InputFloat(LABEL("weight"), &physics_params.positional_constraint_wc, 10.f, 100.f, "%.1f");

			if (ImGui::Button("Save vertex group"))
			{
				user_control.vertex_idxs_memory = user_control.toggle_fixed_vertex_idxs;
			}
			ImGui::SameLine();
			if (ImGui::Button("Load vertex group"))
			{
				user_control.toggle_fixed_vertex_idxs = user_control.vertex_idxs_memory;
				user_control.toggle_vertex_fix = true;
			}

			ImGui::Checkbox("Enable", &enable_positional_constraint);
			ImGui::TreePop();
		}

		if (ImGui::Button("Apply Constraints") && obj_manager.is_deformable_model(user_control.cur_sel_mesh_id))
		{
			obj_manager.apply_constraints(
				user_control.cur_sel_mesh_id,
				physics_params,
				enable_edge_strain_constraint,
				enable_bending_constraint,
				enable_tet_strain_constraint,
				enable_positional_constraint
			);
		}
		ImGui::Text("Current mesh #Constriants = %d", obj_manager.is_deformable_model(user_control.cur_sel_mesh_id) ? obj_manager.models.at(user_control.cur_sel_mesh_id).n_constraints() : 0);
		ImGui::Text("Total #Constraints = %d", obj_manager.total_n_constraints);
    }

	void physics_menu(PhysicsParams& physics_params, const UserControl& user_control)
	{
		if (ImGui::CollapsingHeader("Physics Setting"), ImGuiTreeNodeFlags_DefaultOpen)
		{
			// physics params
			ImGui::Checkbox("Enable Gravity", &physics_params.enable_gravity);

			ImGui::InputDouble(LABEL("mass per vertex"), &physics_params.mass_per_vertex, 0.01, 0.1, "%.3f");

			ImGui::Separator();

			help_marker("Ctrl + LMC to apply external force to the model.");
			ImGui::Text("State: %s", user_control.apply_ext_force ? "Applying" : "Not applying");
			if (user_control.apply_ext_force)
			{
				ImGui::Text("Vertex forced: %d", user_control.ext_forced_vertex_idx);
			}

			ImGui::InputFloat(LABEL("dragging force"), &physics_params.external_force_val, 1.f, 10.f, "%.2f");
		}
	}

	void visualization_menu(
		igl::opengl::glfw::Viewer& viewer, 
		ScreenCapturePlugin& screen_capture_plugin,
		std::unordered_map<pd::MeshIDType, pd::DeformableMesh>& models,
		bool& always_recompute_normal, 
		int id
	)
    {
		if (ImGui::CollapsingHeader("Visualization Setting"), ImGuiTreeNodeFlags_DefaultOpen)
		{
			ImGui::Checkbox("Always recompute normals", &always_recompute_normal); ImGui::SameLine();
			if (ImGui::Button("Recompute normals"))
			{
				for (int i = 0; i < viewer.data_list.size(); i++)
					viewer.data_list[i].compute_normals();
			}

			const int idx = viewer.mesh_index(id);
			ImGui::Checkbox("Wireframe", [&]() { 
					return viewer.data_list[idx].show_lines != 0; 
				},
				[&](bool value) {
					viewer.data_list[idx].show_lines = value;
				}
			);
			ImGui::Checkbox("Double sided lighting", &viewer.data_list[idx].double_sided);
			ImGui::InputFloat(LABEL("Point Size"), &viewer.data_list[idx].point_size, 1.f, 10.f);

			int n_vertices, n_faces;
			n_vertices = n_faces = 0;
			for (const auto& [id, model] : models)
			{
				n_vertices += model.positions().rows();
				n_faces += model.faces().rows();
			}
			ImGui::Text("#Vertex = %d", n_vertices);
			ImGui::Text("#Face = %d", n_faces);
			// ImGui::Text("#DOF = %d", model.positions().rows() * 3 - model.n_edges);

			ImGui::Separator();

			if (screen_capture_plugin.is_capturing() == false)
			{
				if (ImGui::Button("Start capture", ImVec2(-1, 0))) 
				{
					std::string capture_path = igl::file_dialog_save();
					if (capture_path.empty() == false)
					{
						screen_capture_plugin.start_capture(capture_path);
					}
				}
			}
			else
			{
				ImGui::Text("Captured %d frames", screen_capture_plugin.cur_capture_frame_id());
				if (ImGui::Button("Stop capture", ImVec2(-1, 0)))
				{
					screen_capture_plugin.stop_capture();
				}
			}
		}
    }

	void instantiator_menu(instancing::Instantiator& instantiator)
	{
		std::vector<std::function<void(instancing::Instantiator&)>> instantiate_caller = {
			&instancing::Instantiator::instance_test,
			&instancing::Instantiator::instance_bar,
			&instancing::Instantiator::instance_bridge,
			&instancing::Instantiator::instance_floor,
			&instancing::Instantiator::instance_cloth,
			&instancing::Instantiator::instance_4hanged_cloth,
			&instancing::Instantiator::instance_bending_hemisphere,
			&instancing::Instantiator::instance_cylinder,
			&instancing::Instantiator::instance_cone,
			&instancing::Instantiator::instance_armadillo,
			&instancing::Instantiator::instance_bunny
		};

		const char* instances[] = { 
			"Test",
			"Bar",
			"Bridge",
			"Floor", 
			"Cloth", 
			"Corners-pinned cloth",
			"Bending Hemisphere",
			"Cylinder",
			"Cone",
			"Armadillo",
			"Bunny"
		};
		static int item_current = 0;
		ImGui::PushItemWidth(-1);
		ImGui::Combo("", &item_current, instances, IM_ARRAYSIZE(instances));
		ImGui::PopItemWidth();

		if (ImGui::Button("Load instance", ImVec2(-1, 0)))
		{
			instantiate_caller.at(item_current)(instantiator);
		}

		ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0, 0.6f, 0.6f));
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0, 0.7f, 0.7f));
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0, 0.8f, 0.8f));
		if (ImGui::Button("Reset all", ImVec2(-1, 0)))
		{
			instantiator.reset_all();
		}
		ImGui::PopStyleColor(3);
	}

	void simulation_ctrl_menu(
        pd::Solver& solver,
		ObjManager& obj_manager,
		const UserControl& user_control,
        SolverParams& solver_params,
        const PhysicsParams& physics_params,
        igl::opengl::glfw::Viewer& viewer,
        pd::pre_draw_handler& frame_callback,
	    std::unordered_map<pd::MeshIDType, pd::DataMatrixX3>& f_exts,
		igl::opengl::glfw::imgui::ImGuizmoWidget& gizmo,
        bool always_recompute_normal
    )
    {
        if (ImGui::CollapsingHeader("Simulating Control"), ImGuiTreeNodeFlags_DefaultOpen)
		{
			ImGui::Text("Solver is %s", solver.dirty ? "not ready." : "ready.");

			ImGui::Checkbox("Use GPU for local step", &solver_params.use_gpu_for_local_step);
			ImGui::InputDouble(LABEL("timestep"), &solver_params.dt, 0.01, 0.1, "%.4f"); // n_solver_pd_iterations in PD is 1 timestep
			ImGui::InputInt(LABEL("solver #itr"), &solver_params.n_itr_solver_iterations);
			ImGui::InputInt(LABEL("PD #itr"), &solver_params.n_solver_pd_iterations);

			// Solver Selector
			const char* items[] = { "Direct", "Parallel Jacobi", "A-Jacobi-1", "A-Jacobi-2", "A-Jacobi-3" };
			static const char* cur_select_item = "Direct";
			if (ImGui::BeginCombo(LABEL("current solver"), cur_select_item))
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
			if (ImGui::Button("Simulate Single Step", ImVec2(-1, 0)) && viewer.core().is_animating == false)
			{
				pd::tick(viewer, obj_manager.models, physics_params, solver_params, solver, f_exts, always_recompute_normal);
			}
			ImGui::PopStyleColor(3);

			ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0x66, 0xCC, 0xFF, 0xFF));
			ImGui::Checkbox("Auto Simulate!", [&]() { 
					bool is_animating = viewer.core().is_animating;
					if (is_animating)
					{
						// disable gizmo control when animating
						gizmo.visible = !obj_manager.is_deformable_model(user_control.cur_sel_mesh_id);
					}
					return is_animating;
				},
				[&](bool value) {
					if (value == false)
					{
						// enable gizmo when disable animating
						gizmo.visible = true;
					}
					viewer.core().is_animating = value;
				}
			);
			ImGui::PopStyleColor();
		}
    }
}