#pragma once

#include <Eigen/Core>

namespace primitive
{
    class Primitive
    {
    public:
        virtual void collision_handle(Eigen::Vector3f& pos) const = 0;

        virtual void generate_visualized_model(Eigen::MatrixXd& V, Eigen::MatrixXi& F) const = 0;
    };
}