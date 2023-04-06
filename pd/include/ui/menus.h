#pragma once

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>

#include <ui/obj_manager.h>

#include <meshgen/mesh_generator.h>

namespace ui {
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

			ImGui::SetNextItemOpen(true); // Helpful for testing, can be removed later
			if (ImGui::TreeNode("Cloth"))
			{
				static int w = 20;
				static int h = 20;
				ImGui::InputInt("width", &w);
				ImGui::InputInt("height", &h);
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
				ImGui::InputFloat("radius", &radius);
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
			if (ImGui::TreeNode("Bar"))
			{
				static int w = 5;
				static int h = 3;
				static int d = 2;
				ImGui::InputInt("width", &w);
				ImGui::InputInt("height", &h);
				ImGui::InputInt("depth", &d);
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
					igl::read_triangle_mesh("/home/xinghai/codes/pd-with-gpu/assets/meshes/armadillo.obj", V, F);
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
		}
    }
    void collider_generate_menu();
}