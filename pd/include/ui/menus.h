#pragma once

#include <pd/tick.h>

#include <igl/opengl/glfw/Viewer.h>

#include <ui/solver_params.h>
#include <ui/obj_manager.h>
#include <ui/user_control.h>
#include <ui/physics_params.h>

#include <instancing/instantiator.h>

namespace ui {
	template<typename T>
	void mesh_manager_menu(
		const std::string& obj_prompts, 
		const std::unordered_map<int, T>& meshes,
		ObjManager& obj_manager, 
		UserControl& user_control, 
		bool is_animating
	);

	void mesh_remove_menu(ObjManager& obj_manager, int id);

    void deformable_mesh_generate_menu(ObjManager& obj_manager, int id);
    void collider_generate_menu(ObjManager& obj_manager, int id);
    void set_constraints_menu(ObjManager& obj_manager, PhysicsParams& physics_params, const UserControl& user_control);

	void physics_menu(PhysicsParams& physics_params, const UserControl& user_control);

	void visualization_menu(
		igl::opengl::glfw::Viewer& viewer, 
		std::unordered_map<int, pd::DeformableMesh>& models,
		bool& always_recompute_normal, 
		int id
	);

	void instantiator_menu(instancing::Instantiator& instantiator);

	void simulation_ctrl_menu(
        pd::Solver& solver,
        SolverParams& solver_params,
        const PhysicsParams& physics_params,
        igl::opengl::glfw::Viewer& viewer,
        pd::pre_draw_handler& frame_callback,
		std::unordered_map<int, pd::DeformableMesh>& models,
	    std::unordered_map<int, Eigen::MatrixX3d>& f_exts,
        bool always_recompute_normal
    );
}