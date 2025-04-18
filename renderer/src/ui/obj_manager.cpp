#include <ui/obj_manager.h>

namespace ui
{
	void ObjManager::recalc_total_n_constraints()
	{
		total_n_constraints = 0;
		for (const auto& [_, model] : models)
		{
			total_n_constraints += model.n_constraints();
		}
	}

	void ObjManager::rescale(Eigen::MatrixXd& V) const
	{
		// rescale the vertices to make all models look equal in size
		Eigen::RowVector3d v_mean = V.colwise().mean();
		V.rowwise() -= v_mean;
		V.array() /= (V.maxCoeff() - V.minCoeff());
	}

	int ObjManager::add_model(Eigen::MatrixXd& V, const Eigen::MatrixXi& F, bool enable_rescale)
	{
		if (enable_rescale)
		{
			rescale(V);
		}

		// create a new mesh
		pd::MeshIDType obj_id = viewer.append_mesh();
		models.emplace(obj_id, pd::DeformableMesh(V, F));

		add_simulation_model_info(obj_id);

		return obj_id;
	}

	int ObjManager::add_model(Eigen::MatrixXd& V, const Eigen::MatrixXi& T, const Eigen::MatrixXi& boundray_facets, bool enable_rescale)
	{
		if (enable_rescale)
		{
			rescale(V);
		}

		// create a new mesh
		pd::MeshIDType obj_id = viewer.append_mesh();
		models.emplace(obj_id, pd::DeformableMesh(V, T, boundray_facets));

		add_simulation_model_info(obj_id);

		return obj_id;
	}

	void ObjManager::add_simulation_model_info(pd::MeshIDType obj_id)
	{
		const pd::DeformableMesh& model = models.at(obj_id);

		obj_init_pos_map[obj_id] = model.positions();

		// reset f_ext
		f_exts[obj_id].resizeLike(model.positions()); // let external forces add to vertices in the new model
		f_exts[obj_id].setZero();

		// reset viewer
		int idx = viewer.mesh_index(obj_id);
		viewer.data_list[idx].set_mesh(model.positions(), model.faces());
		viewer.data_list[idx].set_colors(model.is_tet_mesh() ? DEFORMABLE_TET_MESH_TEXTURE_COLOR : DEFORMABLE_TRI_MESH_TEXTURE_COLOR);
		viewer.core().align_camera_center(model.positions());
		viewer.data_list[idx].point_size = 10.f;
		// viewer.data_list[idx].label_size = 2.f;
		viewer.data_list[idx].show_custom_labels = true;

		// reset solver
		pd::Solver::instance().set_dirty(true);

		user_control.cur_sel_mesh_id = obj_id;
		user_control.selected_vertex_idx = 0;

		// if this is the only model, select it
		if (models.size() == 1)
		{
			gizmo.visible = true;
			bind_gizmo(obj_id);
		}
	}

	void ObjManager::reset_simulation_model_info(pd::MeshIDType obj_id)
	{
		const pd::DeformableMesh& model = models.at(obj_id);
		obj_init_pos_map[obj_id] = model.positions();

		// reset f_ext
		f_exts[obj_id].resizeLike(model.positions()); // let external forces add to vertices in the new model
		f_exts[obj_id].setZero();

		// reset viewer
		int idx = viewer.mesh_index(obj_id);
		viewer.data_list[idx].clear();
		viewer.data_list[idx].set_mesh(model.positions(), model.faces());
		viewer.data_list[idx].set_colors(model.is_tet_mesh() ? DEFORMABLE_TET_MESH_TEXTURE_COLOR : DEFORMABLE_TRI_MESH_TEXTURE_COLOR);
		viewer.data_list[idx].double_sided = true;
		viewer.core().align_camera_center(model.positions());

		// reset solver
		pd::Solver::instance().set_dirty(true);

		recalc_total_n_constraints();
	}

	void ObjManager::reset_model(pd::MeshIDType obj_id, Eigen::MatrixXd& V, const Eigen::MatrixXi& F, bool enable_rescale)
	{
		// check if obj_id corresponods to a deformable mesh
		if (is_deformable_model(obj_id) == false)
		{
			return;
		}

		if (enable_rescale)
		{
			// rescale the vertices to make all models look equal in size
			rescale(V);
		}

		// reset to a new mesh
		pd::DeformableMesh& model = models.at(obj_id);
		model = pd::DeformableMesh(V, F);

		reset_simulation_model_info(obj_id);
	}

	void ObjManager::reset_model(pd::MeshIDType obj_id, Eigen::MatrixXd& V, const Eigen::MatrixXi& T, const Eigen::MatrixXi& boundray_facets, bool enable_rescale)
	{
		// check if obj_id corresponods to a deformable mesh
		if (is_deformable_model(obj_id) == false)
		{
			return;
		}
		if (enable_rescale)
		{
			// rescale the vertices to make all models look equal in size
			rescale(V);
		}

		// reset to a new mesh
		pd::DeformableMesh& model = models.at(obj_id);
		model = pd::DeformableMesh(V, T, boundray_facets);

		reset_simulation_model_info(obj_id);
	}

	void ObjManager::recalc_data()
	{
		recalc_total_n_constraints();

		// select the first mesh
		if (models.empty() == false)
		{
			user_control.cur_sel_mesh_id = models.begin()->first;
		}
		else
		{
			user_control.cur_sel_mesh_id = rigid_colliders.begin()->first;
		}
		user_control.selected_vertex_idx = 0;
		bind_gizmo(user_control.cur_sel_mesh_id);
	}

	bool ObjManager::remove_model(pd::MeshIDType obj_id, bool recalc)
	{
		if (is_deformable_model(obj_id) == false || models.size() <= 1)
		{
			return false;
		}

		int idx = viewer.mesh_index(obj_id);
		viewer.erase_mesh(idx);

		obj_init_pos_map.erase(obj_id);
		models.erase(obj_id);
		f_exts.erase(obj_id);

		// reset solver
		pd::Solver::instance().set_dirty(true);

		if (recalc)
		{
			recalc_data();
		}

		return true;
	}

	int ObjManager::add_rigid_collider(std::unique_ptr<primitive::Primitive> primitive)
	{
		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		primitive->generate_visualized_model(V, F);

		// create a new mesh
		pd::MeshIDType obj_id = viewer.append_mesh();
		rigid_colliders.emplace(obj_id, std::move(primitive));
		obj_init_pos_map[obj_id] = V;

		// apply offset
		for (int i = 0; i < V.rows(); i++)
		{
			V.row(i) += rigid_colliders[obj_id]->center().transpose().cast<pd::DataScalar>();
		}

		// reset viewer
		int idx = viewer.mesh_index(obj_id);
		viewer.data_list[idx].clear();
		viewer.data_list[idx].set_mesh(V, F);
		viewer.data_list[idx].set_colors(RIGID_COLLIDER_TEXTURE_COLOR);
		viewer.data_list[idx].show_lines = 0;

		user_control.cur_sel_mesh_id = obj_id;
		user_control.selected_vertex_idx = 0;

		// if this is the only model, select it
		if (models.size() == 1)
		{
			gizmo.visible = true;
			bind_gizmo(obj_id);
		}
		return obj_id;
	}

	bool ObjManager::remove_rigid_collider(pd::MeshIDType obj_id, bool recalc)
	{
		if (is_rigid_collider(obj_id) == false || rigid_colliders.size() <= 1)
		{
			return false;
		}
		int idx = viewer.mesh_index(obj_id);
		viewer.erase_mesh(idx);

		obj_init_pos_map.erase(obj_id);
		rigid_colliders.erase(obj_id);

		if (recalc)
		{
			recalc_data();
		}
		return true;
	}

	void ObjManager::apply_constraints(
		pd::MeshIDType obj_id,
		const pd::PhysicsParams& physics_params,
		bool enable_edge_strain_constraint,
		bool enable_bending_constraint,
		bool enable_tet_strain_constraint,
		bool enable_positional_constraint
	)
	{
		pd::DeformableMesh& model = models.at(obj_id);

		model.reset_constraints();
		pd::Solver::instance().set_dirty(true);

		if (enable_edge_strain_constraint)
		{
			model.set_edge_strain_constraints((pd::SimScalar)physics_params.edge_strain_constraint_wc);
		}
		if (enable_bending_constraint)
		{
			model.set_bending_constraints((pd::SimScalar)physics_params.bending_constraint_wc);
		}
		if (enable_tet_strain_constraint)
		{
			model.set_tet_strain_constraints(
				(pd::SimScalar)physics_params.tet_strain_constraint_wc,
				physics_params.tet_strain_constraint_min_xyz.cast<pd::SimScalar>(),
				physics_params.tet_strain_constraint_max_xyz.cast<pd::SimScalar>()
			);
		}
		if (enable_positional_constraint && user_control.toggle_vertex_fix)
		{
			model.toggle_vertices_fixed(
				user_control.toggle_fixed_vertex_idxs,
				(pd::SimScalar)physics_params.positional_constraint_wc
			);
			user_control.toggle_vertex_fix = false;
			user_control.toggle_fixed_vertex_idxs.clear();
		}

		recalc_total_n_constraints();
	}

	bool ObjManager::reset_all()
	{
		while (rigid_colliders.size() > 1 && remove_rigid_collider(rigid_colliders.begin()->first, false));
		while (models.size() > 1 && remove_model(models.begin()->first, false));

		recalc_data();

		return true;
	}

	void ObjManager::bind_gizmo(pd::MeshIDType obj_id)
	{
		if (is_deformable_model(obj_id))
		{
			const Eigen::MatrixXd& V = models.at(obj_id).positions();

			gizmo.T.block<3, 1>(0, 3) =
				0.5 * (V.colwise().maxCoeff() + V.colwise().minCoeff()).transpose().cast<float>();
		}
		else if (is_rigid_collider(obj_id))
		{
			// Note: only translation is available for rigid colliders.
			// For floor, only vertical translation is allowed.
			const pd::SimVector3 center = rigid_colliders[obj_id]->center();
			gizmo.T.block<3, 1>(0, 3) = center.cast<float>();
		}
		else
		{
			printf("Error: Cannot find mesh with id = %d!\n", obj_id);
			return;
		}
	}
}
