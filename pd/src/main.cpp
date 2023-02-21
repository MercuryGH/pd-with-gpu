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

#include <util/gpu_helper.h>

int main(int argc, char* argv[])
{
	igl::opengl::glfw::Viewer viewer;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin;
	viewer.plugins.push_back(&plugin);
	igl::opengl::glfw::imgui::ImGuiMenu menu;
	plugin.widgets.push_back(&menu);

	// print CUDA selection info
	//int n_gpu_devs;
	//int best_dev_idx = util::select_best_device(n_gpu_devs);
	//printf("Available #dev = %d\n", n_gpu_devs);
	//util::test_device(best_dev_idx);

	// pd simulatee
	pd::DeformableMesh model;
	Eigen::MatrixX3d f_ext; // external force
	pd::Solver solver;
	ui::UserControl user_control;
	ui::PhysicsParams physics_params;
	ui::SolverParams solver_params;

	viewer.data().point_size = 10.f;
	viewer.callback_mouse_down = ui::mouse_down_handler{ &model, &user_control };
	viewer.callback_mouse_move = ui::mouse_move_handler{ &model, &user_control, &physics_params, &f_ext };
	viewer.callback_mouse_up = ui::mouse_up_handler{ &user_control };
	ui::pre_draw_handler frame_callback{ &solver, &model, &physics_params, &f_ext, &solver_params };
	viewer.callback_pre_draw = frame_callback; // frame routine

	const auto reset_model = [&](Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXi& E)
	{
		// rescale the vertices to make all models look equal in size
		Eigen::RowVector3d v_mean = V.colwise().mean();
		V.rowwise() -= v_mean;
		V.array() /= (V.maxCoeff() - V.minCoeff());

		// create a new mesh
		model = pd::DeformableMesh(V, F, E);
		solver.set_model(&model);

		// reset f_ext 
		f_ext.resizeLike(model.positions()); // let external forces add to vertices in the new model
		f_ext.setZero();

		// reset viewer
		viewer.data().clear();
		viewer.data().set_mesh(model.positions(), model.faces());
		const Eigen::RowVector3d TEXTURE_COLOR = Eigen::RowVector3d((double)0x66 / 0xff, (double)0xcc / 0xff, 1.0);
		viewer.data().set_colors(TEXTURE_COLOR);
		viewer.core().align_camera_center(model.positions());
	};

	menu.callback_draw_viewer_window = [&]() 
	{
		ImGui::SetNextWindowSize(ImVec2(400.0f, 700.0f));
		ImGui::Begin("PD Panel");

		//const float w = ImGui::GetContentRegionAvailWidth();
		//const float p = ImGui::GetStyle().FramePadding.x;

		if (ImGui::CollapsingHeader("Model Generator", ImGuiTreeNodeFlags_DefaultOpen))
		{
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
					reset_model(V, F, F);
				}
				
				ImGui::TreePop();
			}
			if (ImGui::TreeNode(".obj file"))
			{
				if (ImGui::Button("Load .obj file"))
				{
					std::string file_name = igl::file_dialog_open();
					std::filesystem::path obj_file{ file_name };

					if (std::filesystem::exists(obj_file) && std::filesystem::is_regular_file(obj_file))
					{
						Eigen::MatrixXd V;
						Eigen::MatrixXi F;
						if (igl::read_triangle_mesh(file_name, V, F))
						{
							reset_model(V, F, F);
						}
						else
						{
							printf("Load .obj file failed!\n");
						}
					}
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
				ImGui::Text("Shift + LMC to toggle fix/unfix to a vertex.");
				std::string vertices_to_be_toggled = "";
				for (const int vi : user_control.toggle_fixed_vertex_idxs)
				{
					vertices_to_be_toggled += std::to_string(vi) + " ";
				}
				ImGui::Text("Vertex indices to be toggled: %s", vertices_to_be_toggled.c_str());
				// Pinned 
				ImGui::InputFloat("wi", &physics_params.positional_constraint_wi, 10.f, 100.f, "%.1f");
				ImGui::Checkbox("Enable", &enable_positional_constraint);
				ImGui::TreePop();
			}

			if (ImGui::Button("Apply Constraints"))
			{
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
			}

			ImGui::Text("#Constraints = %u", model.n_constraints());
		}
		if (ImGui::CollapsingHeader("Physics Setting"), ImGuiTreeNodeFlags_DefaultOpen)
		{
			// physics params
			ImGui::Checkbox("Enable Gravity", &physics_params.enable_gravity);

			ImGui::InputFloat("Mass Per Vertex", &physics_params.mass_per_vertex, 1, 10, "%.1f");
		}
		if (ImGui::CollapsingHeader("Picking Setting"), ImGuiTreeNodeFlags_DefaultOpen)
		{
			ImGui::Text("Ctrl + LMC to apply external force to the model.");
			ImGui::Text("State: %s", user_control.apply_ext_force ? "Applying" : "Not applying");
			if (user_control.apply_ext_force)
			{
				ImGui::Text("Vertex forced: %d", user_control.ext_forced_vertex_idx);
			}

			ImGui::InputFloat("Dragging Force", &physics_params.external_force_val, 1.f, 10.f, "%.3f");
		}
		if (ImGui::CollapsingHeader("Visualization Setting"))
		{
			ImGui::Checkbox("Wireframe", [&]() { return viewer.data().show_lines != 0; },
				[&](bool value) {
					viewer.data().show_lines = value;
				});
			ImGui::InputFloat("Point Size", &viewer.data().point_size, 1.f, 10.f);
			ImGui::Text("#Vertex = %d", model.positions().rows());
		}

		if (ImGui::CollapsingHeader("Simulating Control"), ImGuiTreeNodeFlags_DefaultOpen)
		{
			ImGui::Text("Solver is %s", solver.dirty ? "not ready" : "ready");
			ImGui::InputFloat("Timestep", &solver_params.dt, 0.01f, 0.1f, "%.4f");

			ImGui::Separator();

			ImGui::Checkbox("Use GPU for local step", &solver_params.use_gpu_for_local_step);

			ImGui::Separator();

			// Solver Selector
			const char* items[] = {"Direct", "Parallel Jacobi", "A-Jacobi-1", "A-Jacobi-2", "A-Jacobi-3" };
			static const char* cur_select_item = "Direct";
			if (ImGui::BeginCombo("##Current selected solver", cur_select_item))
			{
				for (int i = 0; i < IM_ARRAYSIZE(items); i++)
				{
					bool is_selected = (cur_select_item == items[i]);
					if (ImGui::Selectable(items[i], is_selected))
					{
						// If changed solver
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
			ImGui::InputInt("Itr Solver #Itr", &solver_params.n_itr_solver_iterations);

			ImGui::Separator();
			// statistics
			if (viewer.core().is_animating)
			{
				ImGui::Text("Last frame time elapsed: %lf ms", frame_callback.last_elapse_time);
				ImGui::Text("FPS = %lf", 1000.0 / frame_callback.last_elapse_time);
				ImGui::Text("Time for global step: %lf ms", frame_callback.last_global_step_time);
				ImGui::Text("Time for local step: %lf ms", frame_callback.last_local_step_time);

				// update every 1 seconds
			}
			ImGui::Text("Time for precomputation: %lf ms", frame_callback.last_precomputation_time);

			ImGui::Separator();

			ImGui::InputInt("Solver PD #Itr", &solver_params.n_solver_pd_iterations);
			if (ImGui::Button("Simulate 1 Step") && viewer.core().is_animating == false)
			{
				ui::tick(viewer, &model, &physics_params, &solver_params, &solver, &f_ext);
			}

			ImGui::Checkbox("Auto Simulate!", &viewer.core().is_animating);
		}

		ImGui::End();
	};

	viewer.launch();

	return 0;
}

