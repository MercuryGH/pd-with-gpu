#pragma once

#include <primitive/primitive.h>

namespace primitive {
    class Torus: public Primitive
    {
    public:
        Torus(Eigen::Vector3f center_point, float main_radius, float ring_radius): Primitive(PrimitiveType::TORUS), center_point(center_point), main_radius(main_radius), ring_radius(ring_radius) {}
        bool collision_handle(Eigen::Vector3f& pos) const override;
        void generate_visualized_model(Eigen::MatrixXd& V, Eigen::MatrixXi& F) const override;
        Eigen::Vector3f center() const override;
        void set_center(Eigen::Vector3f center) override;

    private:
        Eigen::Vector3f center_point;
        float main_radius;
        float ring_radius;
    };
}