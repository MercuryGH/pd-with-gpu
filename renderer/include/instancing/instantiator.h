#pragma once

#include <ui/obj_manager.h>
#include <pd/algo_ctrl.h>

namespace instancing {
    struct Instantiator
    {
        ui::ObjManager& obj_manager;
        pd::PhysicsParams& physics_params;

        void reset_all();

        void instance_floor();

        void _instance_bending_hemisphere(pd::SimScalar wc, pd::DataVector3 translation);
        void instance_bending_hemisphere();

        void instance_cloth();
        void instance_cloth_not_bend();
        void instance_cloth_bend();

        void instance_4hanged_cloth();
        void instance_4hanged_cloth_not_bend();
        void instance_4hanged_cloth_bend();

        void instance_cylinder();
        void instance_cylinder_not_bend();
        void instance_cylinder_bend();

        void instance_bar();
        void instance_bridge();

        void instance_ball();

        void instance_tet_obj_model(const std::string& file_path);
        void instance_tri_obj_model(const std::string& file_path);

        void instance_bunny();

        void instance_pinned_armadillo();
        void instance_armadillo();
        void instance_spot();
        void instance_dragon();

        void instance_cone();

        void instance_large_cloth();

        void instance_test();
    };
}
