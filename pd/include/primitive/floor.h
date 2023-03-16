#pragma once
#include <primitive/primitive.h>

namespace primitive
{
    class Floor: public Primitive
    {
    public:
        void collision_handle(Eigen::Vector3f& pos) const override;
        void generate_visualized_model(Eigen::MatrixXd& V, Eigen::MatrixXi& F) const override;

    private:
        int y;
    };
}