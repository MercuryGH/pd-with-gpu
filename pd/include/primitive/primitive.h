#pragma once

#include <Eigen/Core>

namespace primitive
{
    enum class PrimitiveType
    {
        NONE,
        BLOCK,
        FLOOR,
        SPHERE
    };

    class Primitive
    {
    public:
        Primitive(PrimitiveType type): type(type) {}

        // returns true if there is a collision 
        virtual bool collision_handle(Eigen::Vector3f& pos) const = 0;

        virtual void generate_visualized_model(Eigen::MatrixXd& V, Eigen::MatrixXi& F) const = 0;

        // getter and setter
        virtual Eigen::Vector3f center() const = 0;
        virtual void set_center(Eigen::Vector3f center) = 0;

    public:
        PrimitiveType type{ PrimitiveType::NONE };
        bool visible{ true };
    };
}