#pragma once

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuizmoWidget.h>

namespace texturegen
{
    void checkerboard_texture(const int s, const int f,
                         Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& X,
                         Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& A);
}