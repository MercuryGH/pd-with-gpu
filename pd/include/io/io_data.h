#pragma once

#include <pd/deformable_mesh.h>
#include <pd/types.h>
#include <pd/algo_ctrl.h>

#include <util/singleton.h>

namespace io {
    struct IOData final: public util::Singleton<IOData>
    {
        IOData(token) {}

        std::unordered_map<pd::MeshIDType, pd::DeformableMesh> models;
        std::unordered_map<pd::MeshIDType, std::unique_ptr<primitive::Primitive>> rigid_colliders;
        std::unordered_map<pd::MeshIDType, pd::DataMatrixX3> f_exts;
        pd::UserControl user_control;
		pd::PhysicsParams physics_params;
        pd::SolverParams solver_params;
    };
}
