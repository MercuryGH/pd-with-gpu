#pragma once

#include <pd/deformable_mesh.h>
#include <pd/solver.h>

#include <pd/algo_ctrl.h>

namespace primitive
{
	class Primitive;
}

namespace pd {
	void debug_tick(std::unordered_map<pd::MeshIDType, pd::DeformableMesh>& models);

	void compute_external_force(
		const pd::PhysicsParams& physics_params,
		std::unordered_map<pd::MeshIDType, pd::DeformableMesh>& models,
		std::unordered_map<MeshIDType, DataMatrixX3>& f_exts
	);

	void physics_tick(
		std::unordered_map<pd::MeshIDType, pd::DeformableMesh>& models,
		const std::unordered_map<pd::MeshIDType, std::unique_ptr<primitive::Primitive>>& rigid_colliders,
		const pd::PhysicsParams& physics_params,
		const pd::SolverParams& solver_params,
		std::unordered_map<MeshIDType, DataMatrixX3>& f_exts,
		const pd::UserControl& user_control
	);
}
