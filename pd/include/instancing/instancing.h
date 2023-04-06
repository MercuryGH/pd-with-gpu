#pragma once

#include <ui/obj_manager.h>

namespace instancing {
    struct Instantiator
    {
        ui::ObjManager& obj_manager;
	    std::unordered_map<int, pd::DeformableMesh>& models;

        void instance_floor();

        void instance_bending_skirt();
        void instance_cloth();
        void instance_box();
        void instance_armadillo();
    };
}