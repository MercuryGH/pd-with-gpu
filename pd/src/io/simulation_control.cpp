#include <io/simulation_control.h>
#include <io/io_data.h>

#include <pd/physics_tick.h>
#include <pd/solver.h>
#include <pd/deformable_mesh.h>

#include <primitive/primitive.h>

namespace io
{
    auto SimulationControl::physics_tick() -> void
    {
        auto& io_data = IOData::instance();
        pd::physics_tick(
            io_data.models,
            io_data.rigid_colliders,
            io_data.physics_params,
            io_data.solver_params,
            io_data.f_exts,
            io_data.user_control
        );
    }

    auto SimulationControl::add_positional_constraint(pd::MeshIDType mesh_id, const std::unordered_set<pd::VertexIndexType>& v, pd::SimScalar wc) -> void
    {
        auto& model = IOData::instance().models.at(mesh_id);
        model.toggle_vertices_fixed(v, wc);
    }
    auto SimulationControl::add_positional_constraint(pd::MeshIDType mesh_id, pd::VertexIndexType vi, pd::SimScalar wc) -> void
    {
        add_positional_constraint(mesh_id, std::unordered_set<pd::VertexIndexType>({vi}), wc);
    }

    auto SimulationControl::set_edge_strain_constraints(pd::MeshIDType mesh_id, pd::SimScalar wc) -> void
    {
        auto& model = IOData::instance().models.at(mesh_id);
        model.set_edge_strain_constraints(wc);
    }
    auto SimulationControl::set_bending_constraints(pd::MeshIDType mesh_id, pd::SimScalar wc, bool discard_quadratic_term) -> void
    {
        auto& model = IOData::instance().models.at(mesh_id);
        model.set_bending_constraints(wc, discard_quadratic_term);
    }
    auto SimulationControl::set_tet_strain_constraints(pd::MeshIDType mesh_id, pd::SimScalar wc, pd::SimVector3 min_strain_xyz, pd::SimVector3 max_strain_xyz) -> void
    {
        auto& model = IOData::instance().models.at(mesh_id);
        model.set_tet_strain_constraints(wc, min_strain_xyz, max_strain_xyz);
    }

    auto SimulationControl::apply_translation(pd::MeshIDType mesh_id, float translate_x, float translate_y, float translate_z) -> void
    {
        Eigen::Vector3f translate(translate_x, translate_y, translate_z);
        auto& model = IOData::instance().models.at(mesh_id);

        model.apply_translation(translate.cast<pd::DataScalar>());
    }

    auto SimulationControl::check_gravity_enabled() -> bool
    {
        return IOData::instance().physics_params.enable_gravity;
    }

    auto SimulationControl::enable_gravity(bool val) -> void
    {
        IOData::instance().physics_params.enable_gravity = val;
    }

    auto SimulationControl::check_wind_enabled() -> bool
    {
        return IOData::instance().physics_params.enable_wind;
    }

    auto SimulationControl::enable_wind(bool val) -> void
    {
        IOData::instance().physics_params.enable_wind = val;
    }

    auto SimulationControl::set_wind_force_val_dir(float val, float x, float y, float z) -> void
    {
        auto& physics_params = IOData::instance().physics_params;
        physics_params.wind_force_val = val;
        physics_params.wind_dir = Eigen::Vector3f(x, y, z);
    }

    auto SimulationControl::check_gpu_local_step_algo_enabled() -> bool
    {
        return IOData::instance().solver_params.use_gpu_for_local_step;
    }

    auto SimulationControl::enable_gpu_local_step_algo(bool val) -> void
    {
        IOData::instance().solver_params.use_gpu_for_local_step = val;
    }

    auto SimulationControl::set_dt(float dt) -> void
    {
        IOData::instance().solver_params.dt = static_cast<double>(dt);
    }

    auto SimulationControl::set_pd_itr(int n_itr) -> void
    {
        IOData::instance().solver_params.n_solver_pd_iterations = n_itr;
    }

    auto SimulationControl::check_external_force_enabled() -> bool
    {
        return IOData::instance().user_control.apply_ext_force;
    }

    auto SimulationControl::enable_external_force(bool val) -> void
    {
        IOData::instance().user_control.apply_ext_force = val;
    }

    auto SimulationControl::set_external_force_val_dir(float val, float x, float y, float z) -> void
    {
        auto& physics_params = IOData::instance().physics_params;
        physics_params.external_force_val = val;
        physics_params.external_force_dir_cache = Eigen::Vector3f(x, y, z);
    }

    auto SimulationControl::apply_external_force(pd::MeshIDType mesh_id, int vertex_idx) -> bool
    {
        auto& user_control = IOData::instance().user_control;
        if (user_control.apply_ext_force == false)
        {
            return false;
        }

        user_control.apply_ext_force_mesh_id = mesh_id;
        user_control.ext_forced_vertex_idx = vertex_idx;

        auto& physics_params = IOData::instance().physics_params;
        auto val = physics_params.external_force_val;
        auto dir = physics_params.external_force_dir_cache.cast<pd::DataScalar>();

        IOData::instance().f_exts.at(mesh_id).row(vertex_idx) += val * dir;

        return true;
    }
}
