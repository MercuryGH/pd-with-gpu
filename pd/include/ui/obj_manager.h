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
#include <primitive/torus.h>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuizmoWidget.h>

namespace ui
{
	struct ObjManager
	{
		igl::opengl::glfw::Viewer& viewer;
		igl::opengl::glfw::imgui::ImGuizmoWidget& gizmo;
		pd::Solver& solver;
		std::unordered_map<pd::MeshIDType, pd::DeformableMesh>& models;
		std::unordered_map<pd::MeshIDType, std::unique_ptr<primitive::Primitive>>& rigid_colliders;

		std::unordered_map<pd::MeshIDType, Eigen::MatrixXd>& obj_init_pos_map;
		std::unordered_map<pd::MeshIDType, pd::DataMatrixX3>& f_exts;
		ui::UserControl& user_control;
		ui::SolverParams& solver_params;
		int& total_n_constraints;

		void recalc_total_n_constraints();
		void rescale(Eigen::MatrixXd& V) const;
		bool is_deformable_model(pd::MeshIDType obj_id) const
		{
			return models.find(obj_id) != models.end();
		}
		bool is_rigid_collider(pd::MeshIDType obj_id) const
		{
			return rigid_colliders.find(obj_id) != rigid_colliders.end();
		}

	    // add triangle mesh model
		int add_model(Eigen::MatrixXd& V, const Eigen::MatrixXi& F, bool enable_rescale=true);
		// add tetrahedron mesh model
		int add_model(Eigen::MatrixXd& V, const Eigen::MatrixXi& T, const Eigen::MatrixXi& boundray_facets, bool enable_rescale=true);

		void reset_model(pd::MeshIDType obj_id, Eigen::MatrixXd& V, const Eigen::MatrixXi& F, bool enable_rescale=true);
		void reset_model(pd::MeshIDType obj_id, Eigen::MatrixXd& V, const Eigen::MatrixXi& T, const Eigen::MatrixXi& boundray_facets, bool enable_rescale=true);

		bool remove_model(pd::MeshIDType obj_id, bool recalc=true);

		int add_rigid_collider(std::unique_ptr<primitive::Primitive> primitive);
		bool remove_rigid_collider(pd::MeshIDType obj_id, bool recalc=true);

		void apply_constraints(
			pd::MeshIDType obj_id,
			const PhysicsParams& physics_params,
			bool enable_edge_strain_constraint,
			bool enable_bending_constraint,
			bool enable_tet_strain_constraint,
			bool enable_positional_constraint
		);

		bool reset_all();

		// Bind the gizmo to a new mesh when needed.
		void bind_gizmo(pd::MeshIDType obj_id);

		const Eigen::RowVector3d DEFORMABLE_TRI_MESH_TEXTURE_COLOR = Eigen::RowVector3d((double)0x66 / 0xff, (double)0xcc / 0xff, 1.0);
		const Eigen::RowVector3d DEFORMABLE_TET_MESH_TEXTURE_COLOR = Eigen::RowVector3d((double)0x88 / 0xff, (double)0xee / 0xff, 1.0);
		const Eigen::RowVector3d RIGID_COLLIDER_TEXTURE_COLOR = Eigen::RowVector3d((double)125 / 255, (double)220 / 255, (double)117 / 255);

	private:
		void add_simulation_model_info(pd::MeshIDType obj_id);
		void reset_simulation_model_info(pd::MeshIDType obj_id);

		void recalc_data();
	};
}