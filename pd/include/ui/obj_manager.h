#pragma once

#include <pd/deformable_mesh.h>
#include <pd/solver.h>

#include <ui/user_control.h>
#include <ui/physics_params.h>
#include <ui/solver_params.h>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuizmoWidget.h>

namespace ui
{
	struct ObjManager
	{
		igl::opengl::glfw::Viewer& viewer;
		igl::opengl::glfw::imgui::ImGuizmoWidget& gizmo;
		std::unordered_map<int, Eigen::Matrix4f>& obj_t_map;
		pd::Solver& solver;
		std::unordered_map<int, pd::DeformableMesh>& models;
		std::unordered_map<int, Eigen::MatrixX3d>& f_exts;
		ui::UserControl& user_control;
		ui::SolverParams& solver_params;
		int& total_n_constraints;

		void recalc_total_n_constraints() const;
		void rescale(Eigen::MatrixXd& V) const;

		void add_model(Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXi& E);
		void reset_model(int obj_id, Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXi& E);
		void remove_model(int obj_id);

		// Bind the gizmo to a new mesh when needed.
		void bind_gizmo(int obj_id);
	};
}