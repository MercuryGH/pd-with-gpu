#include <pd/physics_tick.h>
#include <igl/per_vertex_normals.h>

#include <Eigen/Geometry>

#include <primitive/primitive.h>

#include <io/mesh_io.h>

namespace pd {
	void debug_tick(std::unordered_map<pd::MeshIDType, pd::DeformableMesh>& models)
	{
		const auto& model = models.at(114514);

        io::MeshIO& mesh_io = io::MeshIO::instance();

		using D3Test = struct {
            double r, g, b;
        };

        std::vector<D3Test> v2;

        mesh_io.export_triangle_mesh(114514, v2);

        // for (int i = 0; i < v2.size(); i++)
        // {
        //     std::cout << v2[i].r << " " << v2[i].g << " " << v2[i].b << "\n";
        // }
	}

	void compute_external_force(
		const pd::PhysicsParams& physics_params,
		std::unordered_map<pd::MeshIDType, pd::DeformableMesh>& models,
		std::unordered_map<MeshIDType, DataMatrixX3>& f_exts
	)
	{
		if (models.empty())
		{
			return;
		}

		// gravity
		if (physics_params.enable_gravity == true)
		{
			constexpr DataScalar GRAVITY = 9.8;
			for (auto& [id, f_ext] : f_exts)
			{
				// apply gravity at y direction on all vertices
				f_ext.col(1) -= GRAVITY * models.at(id).masses();
			}
		}

		// wind
		if (physics_params.enable_wind == true)
		{
			for (auto& [id, f_ext] : f_exts)
			{
				const DataMatrixX3& vertex_normals = models.at(id).vertex_normals();

				// DataVector3 wind_dir(1, 1, 1);
				DataVector3 wind_dir = physics_params.wind_dir.cast<DataScalar>();
				wind_dir.normalize();

				DataScalar wind_force_val = static_cast<DataScalar>(physics_params.wind_force_val);

				// traverse vertices
				for (int i = 0; i < f_ext.rows(); i++)
				{
					DataScalar cos_theta = std::abs(vertex_normals.row(i).dot(wind_dir));
					f_ext.row(i) += wind_dir * wind_force_val * cos_theta;
				}
			}
		}
	}

	void physics_tick(
		std::unordered_map<pd::MeshIDType, pd::DeformableMesh>& models,
		const std::unordered_map<pd::MeshIDType, std::unique_ptr<primitive::Primitive>>& rigid_colliders,
		const pd::PhysicsParams& physics_params,
		const pd::SolverParams& solver_params,
		std::unordered_map<MeshIDType, DataMatrixX3>& f_exts,
		const pd::UserControl& user_control
	)
	{
		auto& solver = pd::Solver::instance();
		if (models.empty() == true)
		{
			return;
		}

	    // apply mass
        for (auto& [id, model] : models)
        {
            bool flag = model.apply_mass_per_vertex(physics_params.mass_per_vertex);
            if (flag == true)
            {
                solver.set_dirty(true);
            }
        }

		compute_external_force(physics_params, models, f_exts);

		if (solver.use_gpu_for_local_step != solver_params.use_gpu_for_local_step)
		{
			solver.use_gpu_for_local_step = solver_params.use_gpu_for_local_step;
			solver.set_dirty(true);
		}

		if (solver.is_algo_changed())
		{
			solver.clear_solver();
			solver.set_solver(solver_params.selected_solver);
			solver.set_algo_changed(false);
		}

		if (solver.is_dirty())
		{
			solver.clear_solver();
			solver.set_dt(solver_params.dt);
			solver.precompute(models);
			solver.set_dirty(false);
		}
		solver.step(models, f_exts, rigid_colliders, solver_params.n_solver_pd_iterations, solver_params.n_itr_solver_iterations);

		// if (user_control.headless_mode == true)
		// {
		// 	debug_tick(models);
		// }

		// reset external force
		for (auto& [id, f_ext] : f_exts)
		{
			f_ext.setZero();
		}

		// calculate vertex normals
		for (auto& [id, model] : models)
		{
			if (model.is_tet_mesh() == false)
			{
				igl::per_vertex_normals(model.positions(), model.faces(), model.vertex_normals());
			}
		}
	}
}
