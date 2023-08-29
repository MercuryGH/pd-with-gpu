#pragma once

#include <pd/deformable_mesh.h>
#include <pd/solver.h>

#include <primitive/primitive.h>

#include <pd/algo_ctrl.h>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject_onto_mesh.h>
#include <util/cpu_timer.h>

namespace rendering {
	// tick physics and rendering frame
	void tick(
		igl::opengl::glfw::Viewer& viewer,
		std::unordered_map<pd::MeshIDType, pd::DeformableMesh>& models,
		const std::unordered_map<pd::MeshIDType, std::unique_ptr<primitive::Primitive>>& rigid_colliders,
		const pd::PhysicsParams& physics_params,
		const pd::SolverParams& solver_params,
		std::unordered_map<pd::MeshIDType, pd::DataMatrixX3>& f_exts,
		const pd::UserControl& user_control
	);

	void rendering_tick(
		igl::opengl::glfw::Viewer& viewer,
		std::unordered_map<pd::MeshIDType, pd::DeformableMesh>& models,
		const pd::UserControl& user_control
	);

	void draw_debug_info(
		bool enable_debug_draw,
		igl::opengl::glfw::Viewer& viewer,
		std::unordered_map<pd::MeshIDType, pd::DeformableMesh>& models,
		pd::MeshIDType sel_mesh_id,
		pd::VertexIndexType sel_vertex_idx
	);

	// Frame routine before rendering
	struct pre_draw_handler
	{
		std::unordered_map<pd::MeshIDType, pd::DeformableMesh>& models;
		const std::unordered_map<pd::MeshIDType, std::unique_ptr<primitive::Primitive>>& rigid_colliders;

		pd::PhysicsParams& physics_params;
		std::unordered_map<pd::MeshIDType, pd::DataMatrixX3>& f_exts;
		pd::SolverParams& solver_params;
		const pd::UserControl& user_control;

		util::CpuTimer timer;
		static double last_elapse_time;
		static double last_local_step_time;
		static double last_global_step_time;
		static double last_precomputation_time;

		bool operator()(igl::opengl::glfw::Viewer& viewer);
	};
}
