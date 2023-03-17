#pragma once
#include <primitive/primitive.h>

namespace primitive
{
    class Block: public Primitive
    {
    public:
        Block(Eigen::Vector3f center_point, float x, float y, float z): Primitive(PrimitiveType::BLOCK), center_point(center_point), x(x), y(y), z(z) {}
        void collision_handle(Eigen::Vector3f& pos) const override;
        void generate_visualized_model(Eigen::MatrixXd& V, Eigen::MatrixXi& F) const override;
        Eigen::Vector3f center() const override;
        void set_center(Eigen::Vector3f center) override;

    private:
        Eigen::Vector3f center_point;
        float x, y, z;
    };
}