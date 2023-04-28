#pragma once

#include <pd/deformable_mesh.h>
#include <pd/solver.h>

#include <primitive/primitive.h>

#include <ui/user_control.h>
#include <ui/physics_params.h>
#include <ui/solver_params.h>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject_onto_mesh.h>
#include <util/cpu_timer.h>

namespace pd {
	// tick physics and rendering frame
	void tick(
		igl::opengl::glfw::Viewer& viewer,
		std::unordered_map<pd::MeshIDType, pd::DeformableMesh>& models,
		const ui::PhysicsParams& physics_params,
		const ui::SolverParams& solver_params,
		pd::Solver& solver,
		std::unordered_map<MeshIDType, DataMatrixX3>& f_exts,
		const ui::UserControl& user_control
	);

	void rendering_tick(
		igl::opengl::glfw::Viewer& viewer,
		std::unordered_map<pd::MeshIDType, pd::DeformableMesh>& models,
		std::unordered_map<MeshIDType, DataMatrixX3>& f_exts,
		const ui::UserControl& user_control
	);

	void physics_tick(
		std::unordered_map<pd::MeshIDType, pd::DeformableMesh>& models,
		const ui::PhysicsParams& physics_params,
		const ui::SolverParams& solver_params,
		pd::Solver& solver,
		std::unordered_map<MeshIDType, DataMatrixX3>& f_exts
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
		pd::Solver& solver;
		std::unordered_map<pd::MeshIDType, pd::DeformableMesh>& models;

		ui::PhysicsParams& physics_params;
		std::unordered_map<MeshIDType, DataMatrixX3>& f_exts;
		ui::SolverParams& solver_params;
		const ui::UserControl& user_control;

		util::CpuTimer timer;
		static double last_elapse_time;
		static double last_local_step_time;
		static double last_global_step_time;
		static double last_precomputation_time;

		bool operator()(igl::opengl::glfw::Viewer& viewer);
	};
}