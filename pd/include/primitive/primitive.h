#pragma once

#include <Eigen/Core>

#include <pd/types.h>

namespace primitive
{
    enum class PrimitiveType
    {
        NONE,
        BLOCK,
        FLOOR,
        SPHERE,
        TORUS
    };

    class Primitive
    {
    public:
        Primitive(PrimitiveType type): type(type) {}

        // returns true if there is a collision 
        virtual bool collision_handle(pd::SimVector3& pos) const = 0;

        virtual void generate_visualized_model(Eigen::MatrixXd& V, Eigen::MatrixXi& F) const = 0;

        // getter and setter
        virtual pd::SimVector3 center() const = 0;
        virtual void set_center(pd::SimVector3 center) = 0;

    public:
        PrimitiveType type{ PrimitiveType::NONE };
        bool visible{ true };
    };
}