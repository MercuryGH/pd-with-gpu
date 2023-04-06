#pragma once

#include <pd/deformable_mesh.h>
#include <pd/solver.h>

#include <ui/user_control.h>
#include <ui/physics_params.h>
#include <ui/solver_params.h>

#include <primitive/primitive.h>
#include <primitive/block.h>
#include <primitive/sphere.h>
#include <primitive/floor.h>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuizmoWidget.h>

namespace ui
{
	struct ObjManager
	{
		igl::opengl::glfw::Viewer& viewer;
		igl::opengl::glfw::imgui::ImGuizmoWidget& gizmo;
		pd::Solver& solver;
		std::unordered_map<int, pd::DeformableMesh>& models;
		std::unordered_map<int, std::unique_ptr<primitive::Primitive>>& rigid_colliders;

		std::unordered_map<int, Eigen::MatrixXd>& obj_init_pos_map;
		std::unordered_map<int, Eigen::MatrixX3d>& f_exts;
		ui::UserControl& user_control;
		ui::SolverParams& solver_params;
		int& total_n_constraints;

		void recalc_total_n_constraints();
		void rescale(Eigen::MatrixXd& V) const;
		bool is_deformable_model(int obj_id) const
		{
			return models.find(obj_id) != models.end();
		}
		bool is_rigid_collider(int obj_id) const
		{
			return rigid_colliders.find(obj_id) != rigid_colliders.end();
		}

	private:
		void add_simulation_model_info(int obj_id);
		void reset_simulation_model_info(int obj_id);

	public:	
	    // add triangle mesh model
		int add_model(Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
		// add tetrahedron mesh model
		int add_model(Eigen::MatrixXd& V, const Eigen::MatrixXi& T, const Eigen::MatrixXi& boundray_facets);

		void reset_model(int obj_id, Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
		void reset_model(int obj_id, Eigen::MatrixXd& V, const Eigen::MatrixXi& T, const Eigen::MatrixXi& boundray_facets);

		void remove_model(int obj_id);

		void add_rigid_collider(std::unique_ptr<primitive::Primitive> primitive);
		void remove_rigid_collider(int obj_id);

		void apply_constraints(
			int obj_id,
			const PhysicsParams& physics_params,
			bool enable_edge_strain_constraint,
			bool enable_bending_constraint,
			bool enable_tet_strain_constraint,
			bool enable_positional_constraint
		);

		// Bind the gizmo to a new mesh when needed.
		void bind_gizmo(int obj_id);
	};
}