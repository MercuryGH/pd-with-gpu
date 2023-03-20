#include <ui/obj_manager.h>

namespace ui
{
	void ObjManager::recalc_total_n_constraints() 
	{
		total_n_constraints = 0;
		for (const auto& [_, model] : models)
		{
			total_n_constraints += model.get_all_constraints().size();
		}
	}

	void ObjManager::rescale(Eigen::MatrixXd& V) const
	{
		// rescale the vertices to make all models look equal in size
		Eigen::RowVector3d v_mean = V.colwise().mean();
		V.rowwise() -= v_mean;
		V.array() /= (V.maxCoeff() - V.minCoeff());
	}

	void ObjManager::add_model(Eigen::MatrixXd& V, const Eigen::MatrixXi& E)
	{
		rescale(V);

		// create a new mesh
		int obj_id = viewer.append_mesh();
		models.emplace(obj_id, pd::DeformableMesh(V, E, obj_id));
		pd::DeformableMesh& model = models[obj_id];

		obj_init_pos_map[obj_id] = V;

		// reset f_ext 
		f_exts[obj_id].resizeLike(model.positions()); // let external forces add to vertices in the new model
		f_exts[obj_id].setZero();

		// reset viewer
		int idx = viewer.mesh_index(obj_id);
		viewer.data_list[idx].set_mesh(model.positions(), model.faces());
		const Eigen::RowVector3d TEXTURE_COLOR = Eigen::RowVector3d((double)0x66 / 0xff, (double)0xcc / 0xff, 1.0);
		viewer.data_list[idx].set_colors(TEXTURE_COLOR);
		viewer.core().align_camera_center(model.positions());
		viewer.data_list[idx].point_size = 10.f;

		// reset solver
		solver.dirty = true;

		// if this is the only model, select it
		if (models.size() == 1)
		{
			user_control.cur_sel_mesh_id = obj_id;
			gizmo.visible = true;
			bind_gizmo(obj_id);
		}
	}

	void ObjManager::reset_model(int obj_id, Eigen::MatrixXd& V, const Eigen::MatrixXi& E)
	{
		// rescale the vertices to make all models look equal in size
		rescale(V);

		// reset to a new mesh
		pd::DeformableMesh& model = models[obj_id];
		model = pd::DeformableMesh(V, E, obj_id);

		obj_init_pos_map[obj_id] = V;

		// reset f_ext 
		f_exts[obj_id].resizeLike(model.positions()); // let external forces add to vertices in the new model
		f_exts[obj_id].setZero();

		// reset viewer
		int idx = viewer.mesh_index(obj_id);
		viewer.data_list[idx].clear();
		viewer.data_list[idx].set_mesh(model.positions(), model.faces());
		const Eigen::RowVector3d TEXTURE_COLOR = Eigen::RowVector3d((double)0x66 / 0xff, (double)0xcc / 0xff, 1.0);
		viewer.data_list[idx].set_colors(TEXTURE_COLOR);
		viewer.core().align_camera_center(model.positions());

		// reset solver
		solver.dirty = true;

		recalc_total_n_constraints();
	}

	void ObjManager::remove_model(int obj_id)
	{
		if (models.size() <= 1)
		{
			printf("Error: Cannot remove the last mesh!\n");
			return;
		}
		int idx = viewer.mesh_index(obj_id);
		viewer.erase_mesh(idx);

		obj_init_pos_map.erase(obj_id);
		models.erase(obj_id);
		f_exts.erase(obj_id);

		// reset solver
		solver.dirty = true;

		recalc_total_n_constraints();

		// select the first mesh
		user_control.cur_sel_mesh_id = models.begin()->first;
		bind_gizmo(user_control.cur_sel_mesh_id);
	}

	void ObjManager::add_rigid_collider(std::unique_ptr<primitive::Primitive> primitive)
	{
		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		primitive->generate_visualized_model(V, F);

		// create a new mesh
		int obj_id = viewer.append_mesh();
		rigid_colliders.emplace(obj_id, std::move(primitive));
		obj_init_pos_map[obj_id] = V;

		// reset viewer
		int idx = viewer.mesh_index(obj_id);
		viewer.data_list[idx].clear();
		viewer.data_list[idx].set_mesh(V, F);
		const Eigen::RowVector3d TEXTURE_COLOR = Eigen::RowVector3d((double)0xcc / 0xff, (double)0xcc / 0xff, 1);
		viewer.data_list[idx].set_colors(TEXTURE_COLOR);
		viewer.data_list[idx].show_lines = 0;

		// if this is the only model, select it
		if (models.size() == 1)
		{
			user_control.cur_sel_mesh_id = obj_id;
			gizmo.visible = true;
			bind_gizmo(obj_id);
		}
	}

	void ObjManager::remove_rigid_collider(int obj_id)
	{
	    if (rigid_colliders.size() <= 1)
		{
			printf("Error: Cannot remove the last mesh!\n");
			return;
		}
		int idx = viewer.mesh_index(obj_id);
		viewer.erase_mesh(idx);

		obj_init_pos_map.erase(obj_id);
		rigid_colliders.erase(obj_id);

		// select the first model
		user_control.cur_sel_mesh_id = rigid_colliders.begin()->first;
		bind_gizmo(user_control.cur_sel_mesh_id);
	}

	void ObjManager::bind_gizmo(int obj_id)
	{
		if (is_deformable_model(obj_id))
		{
			const Eigen::MatrixXd& V = models[obj_id].positions();

			gizmo.T.block(0, 3, 3, 1) =
				0.5 * (V.colwise().maxCoeff() + V.colwise().minCoeff()).transpose().cast<float>();
		}
		else if (is_rigid_collider(obj_id))
		{
			// Note: only translation is available for rigid colliders.
			// For floor, only vertical translation is allowed.
			const Eigen::Vector3f center = rigid_colliders[obj_id]->center();
			gizmo.T.block(0, 3, 3, 1) = center;
		}
		else
		{
			printf("Error: Cannot find mesh with id = %d!\n", obj_id);
			return;
		}
	}
}
