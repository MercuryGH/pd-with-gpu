#include <ui/input_callbacks.h>

namespace ui {
    bool mouse_down_handler::operator()(igl::opengl::glfw::Viewer& viewer, int button, int modifier)
    {
        if (models.empty() == true)
        {
            return false;
        }

        // decode keycode
        if (static_cast<Button>(button) != Button::Left)
        {
            return false;
        }

        // get actual mouse hit viewport coordinate
        const float x = static_cast<float>(viewer.current_mouse_x);
        const float y = viewer.core().viewport(3) - static_cast<float>(viewer.current_mouse_y);

        bool has_vertex_hit = false;
        int mesh_id = -1;
        int fid; // id of the first face hit
        Eigen::Vector3f bc; // barycentric coordinates of hit
        for (const auto& [id, model] : models)
        {
            // raycast
            bool hit_vertex = igl::unproject_onto_mesh(
                Eigen::Vector2f(x, y),
                viewer.core().view, // model matrix
                viewer.core().proj, // projection matrix
                viewer.core().viewport, // viewport vector
                model.positions(),
                model.faces(),
                fid,
                bc
            );
            if (hit_vertex == true)
            {
                // has >= 1 mesh begin hit
                has_vertex_hit = true;
                // update mesh_id
                mesh_id = id;

                // if the user preferred mesh is hit, we breaks earlier.
                // otherwise we continue to find the hit.
                if (id == user_control.cur_sel_mesh_id)
                {
                    break;
                }
            }
        }
        if (has_vertex_hit == false) 
        {
            return false;
        }

        const auto find_hit_vertex = [](const Eigen::Vector3f &bc) {
            int ret = 0;
            // find the nearest vertex around the hit point
            if (bc(1) > bc(0) && bc(1) > bc(2)) {
                ret = 1;
            }
            if (bc(2) > bc(0) && bc(2) > bc(1)) {
                ret = 2;
            }
            return ret;
        };

        int hit_vertex_idx = models[mesh_id].faces()(fid, find_hit_vertex(bc));

        // Handling user input

        bool flag = false;
        // Hold control
        if (modifier == GLFW_MOD_CONTROL)
        {
            flag = true;
            // apply external force
            user_control.apply_ext_force_mesh_id = mesh_id;
            user_control.apply_ext_force = true;
            user_control.ext_forced_vertex_idx = hit_vertex_idx;
            user_control.mouse_x = viewer.current_mouse_x;
            user_control.mouse_y = viewer.current_mouse_y;
        }
        else if (modifier == GLFW_MOD_SHIFT)
        {
            flag = true;
            // toggle fix/unfix
            user_control.toggle_vertex_fix = true;
            if (user_control.toggle_fixed_vertex_idxs.find(hit_vertex_idx) == user_control.toggle_fixed_vertex_idxs.end())
            {
                user_control.toggle_fixed_vertex_idxs.insert(hit_vertex_idx);
            }
            else
            {
                user_control.toggle_fixed_vertex_idxs.erase(hit_vertex_idx);
            }
        }
        else
        {
            user_control.cur_sel_mesh_id = mesh_id;
            user_control.selected_vertex_idx = hit_vertex_idx;
        }

        return flag;
    }

    bool mouse_move_handler::operator()(igl::opengl::glfw::Viewer& viewer, int button, int modifier)
    {
        if (models.empty() == true)
        {
            return false;
        }
        if (user_control.apply_ext_force == false)
        {
            return false;
        }

        // get origin mouse hit viewport coordinate
        const float x0 = static_cast<float>(user_control.mouse_x);
        const float y0 = viewer.core().viewport(3) - static_cast<float>(user_control.mouse_y);

        // update user control
        user_control.mouse_x = viewer.current_mouse_x;
        user_control.mouse_y = viewer.current_mouse_y;
        // get current mouse hit viewport coordinate
        const float x1 = static_cast<float>(viewer.current_mouse_x);
        const float y1 = viewer.core().viewport(3) - static_cast<float>(viewer.current_mouse_y);

        const Eigen::Vector3f p0 = igl::unproject(
            Eigen::Vector3f(x0, y0, 0.5f),
            viewer.core().view,
            viewer.core().proj,
            viewer.core().viewport
        );
        const Eigen::Vector3f p1 = igl::unproject(
            Eigen::Vector3f(x1, y1, 0.5f),
            viewer.core().view,
            viewer.core().proj,
            viewer.core().viewport
        );
        const Eigen::Vector3f dir = (p1 - p0).normalized();

        //std::cout << f_ext.row(user_control.ext_forced_vertex_idx) << "\n";
        // add force
        int apply_f_ext_obj_id = user_control.apply_ext_force_mesh_id;
        f_exts[apply_f_ext_obj_id].row(user_control.ext_forced_vertex_idx) += (dir.transpose() * physics_params.external_force_val).cast<double>();
        //std::cout << f_ext.row(user_control.ext_forced_vertex_idx) << "\n\n";

        return true;
    }

    bool mouse_up_handler::operator()(igl::opengl::glfw::Viewer& viewer, int button, int modifier)
    {
        if (static_cast<Button>(button) != Button::Left)
        {
            return false;
        }
        if (user_control.apply_ext_force == true)
        {
            user_control.apply_ext_force = false;
        }

        return false;
    }

    void gizmo_handler::operator()(const Eigen::Matrix4f& T)
    {
        if (models.empty() && rigid_colliders.empty())
        {
            return;
        }

        const int obj_id = user_control.cur_sel_mesh_id;
        // disable user manipulation on deformable mesh while animating to avoid weird problem
        if (obj_manager.is_deformable_model(obj_id) && is_animating) 
        {
            return;
        }

        const Eigen::MatrixXd& V = obj_init_pos_map[obj_id];

        const Eigen::Matrix4d TT = T.cast<double>().transpose();
        const Eigen::MatrixXd positions = (V.rowwise().homogeneous() * TT).rowwise().hnormalized();
        // const Eigen::MatrixXd positions = (TT).rowwise().hnormalized();

        if (models.find(obj_id) != models.end())
        {
            // TODO: apply lazy update (not necessary indeed)
            models[obj_id].set_positions(positions);
        }
        else if (rigid_colliders.find(obj_id) != rigid_colliders.end())
        {
            Eigen::Vector3f center = 0.5 * (positions.colwise().maxCoeff() + positions.colwise().minCoeff()).transpose().cast<float>();
            rigid_colliders[obj_id]->set_center(center);
        }
        else
        {
            printf("Error: cannot find obj with id %d!\n", obj_id);
        }

        int idx = viewer.mesh_index(obj_id);
        viewer.data_list[idx].set_vertices(positions);
        viewer.data_list[idx].compute_normals();
    }

    bool keypress_handler::operator()(igl::opengl::glfw::Viewer& viewer, int button, int modifier)
    {
        switch (button)
        {
            case ' ': gizmo.visible = !gizmo.visible; return true;
            case 'W': case 'w': gizmo.operation = ImGuizmo::TRANSLATE; return true;
            case 'E': case 'e': gizmo.operation = ImGuizmo::ROTATE;    return true;
            case 'R': case 'r': gizmo.operation = ImGuizmo::SCALE;     return true;
        }
        return false;
    }

}