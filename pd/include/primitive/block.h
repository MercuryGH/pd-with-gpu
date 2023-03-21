#pragma once
#include <primitive/primitive.h>

namespace primitive
{
    class Block: public Primitive
    {
    public:
        Block(Eigen::Vector3f center_point, Eigen::Vector3f xyz): Primitive(PrimitiveType::BLOCK), center_point(center_point), half_xyz(0.5f * xyz) {}
        bool collision_handle(Eigen::Vector3f& pos) const override;
        void generate_visualized_model(Eigen::MatrixXd& V, Eigen::MatrixXi& F) const override;
        Eigen::Vector3f center() const override;
        void set_center(Eigen::Vector3f center) override;

    private:
        Eigen::Vector3f center_point;
        Eigen::Vector3f half_xyz;
    };
}