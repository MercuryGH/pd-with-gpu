#pragma once

#include <unordered_set>

#include <pd/types.h>

#include <util/singleton.h>

namespace io {
    struct SimulationControl final: public util::Singleton<SimulationControl>
    {
        SimulationControl(token) {}

        auto physics_tick() -> void;

        auto add_positional_constraint(pd::MeshIDType mesh_id, const std::unordered_set<pd::VertexIndexType> &v, pd::SimScalar wc) -> void;
        auto add_positional_constraint(pd::MeshIDType mesh_id, pd::VertexIndexType vi, pd::SimScalar wc) -> void;
		auto set_edge_strain_constraints(pd::MeshIDType mesh_id, pd::SimScalar wc) -> void;
		auto set_bending_constraints(pd::MeshIDType mesh_id, pd::SimScalar wc, bool discard_quadratic_term=false) -> void;
		auto set_tet_strain_constraints(pd::MeshIDType mesh_id, pd::SimScalar wc, pd::SimVector3 min_strain_xyz=pd::SimVector3::Ones(), pd::SimVector3 max_strain_xyz=pd::SimVector3::Ones()) -> void;
        auto apply_translation(pd::MeshIDType mesh_id, float translate_x, float translate_y, float translate_z) -> void;

        auto check_gravity_enabled() -> bool;
        auto enable_gravity(bool val) -> void;

        auto check_wind_enabled() -> bool;
        auto enable_wind(bool val) -> void;
        auto set_wind_force_val_dir(float val, float x, float y, float z) -> void;

        auto check_gpu_local_step_algo_enabled() -> bool;
        auto enable_gpu_local_step_algo(bool val) -> void;

        auto set_dt(float dt) -> void;
        auto set_pd_itr(int n_itr) -> void;

        auto check_external_force_enabled() -> bool;
        auto enable_external_force(bool val) -> void;
        auto set_external_force_val_dir(float val, float x, float y, float z) -> void;
        auto apply_external_force(pd::MeshIDType mesh_id, int vertex_idx) -> bool;
    };
}
