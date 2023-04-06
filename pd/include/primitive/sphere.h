#pragma once
#include <primitive/primitive.h>

namespace primitive
{
    class Sphere: public Primitive
    {
    public:
        Sphere(Eigen::Vector3f center_point, float radius): Primitive(PrimitiveType::SPHERE), center_point(center_point), radius(radius) {}
        bool collision_handle(Eigen::Vector3f& pos) const override;
        void generate_visualized_model(Eigen::MatrixXd& V, Eigen::MatrixXi& F) const override;
        Eigen::Vector3f center() const override;
        void set_center(Eigen::Vector3f center) override;

    private:
        Eigen::Vector3f center_point;
        float radius;
    };
}