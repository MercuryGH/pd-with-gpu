#pragma once
#include <primitive/primitive.h>

namespace primitive
{
    class Floor: public Primitive
    {
    public:
        Floor(int y): Primitive(PrimitiveType::FLOOR), center_point(Eigen::Vector3f(0, y, 0)) {}
        void collision_handle(Eigen::Vector3f& pos) const override;
        void generate_visualized_model(Eigen::MatrixXd& V, Eigen::MatrixXi& F) const override;
        Eigen::Vector3f center() const override;
        void set_center(Eigen::Vector3f center) override;

    private:
        Eigen::Vector3f center_point;
    };
}