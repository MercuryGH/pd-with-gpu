#pragma once

#include <ui/obj_manager.h>

namespace instancing {
    struct Instantiator
    {
        ui::ObjManager& obj_manager;
	    std::unordered_map<int, pd::DeformableMesh>& models;

        void instance_floor();

        void instance_bending_hemisphere();
        void instance_cloth();
        void instance_cylinder();
        void instance_bar();
        void instance_obj_model(const std::string& file_path);
        void instance_bunny();
        void instance_armadillo();

        void instance_cone();

        void instance_test();
    };
}