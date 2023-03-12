#include <ui/obj_manager.h>

namespace ui
{
	void ObjManager::recalc_total_n_constraints() const
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

	void ObjManager::add_model(Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXi& E)
	{
		rescale(V);

		// create a new mesh
		int obj_id = viewer.append_mesh();
		models.emplace(obj_id, pd::DeformableMesh(V, F, E, obj_id));
		pd::DeformableMesh& model = models[obj_id];

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
			bind_gizmo(obj_id);
		}
	}

	void ObjManager::reset_model(int obj_id, Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXi& E)
	{
		// rescale the vertices to make all models look equal in size
		rescale(V);

		// reset to a new mesh
		pd::DeformableMesh& model = models[obj_id];
		model = pd::DeformableMesh(V, F, E, obj_id);

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

	}

	void ObjManager::bind_gizmo(int obj_id)
	{
		const Eigen::MatrixXd& V = models[obj_id].positions();

		gizmo.T.block(0, 3, 3, 1) =
			0.5 * (V.colwise().maxCoeff() + V.colwise().minCoeff()).transpose().cast<float>();

		obj_t_map[obj_id] = gizmo.T;
	}
}
