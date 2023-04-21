#pragma once

#include <primitive/primitive.h>

namespace primitive {
    class Torus: public Primitive
    {
    public:
        Torus(pd::SimVector3 center_point, pd::SimScalar main_radius, pd::SimScalar ring_radius): Primitive(PrimitiveType::TORUS), center_point(center_point), main_radius(main_radius), ring_radius(ring_radius) {}
        bool collision_handle(pd::SimVector3& pos) const override;
        void generate_visualized_model(Eigen::MatrixXd& V, Eigen::MatrixXi& F) const override;
        pd::SimVector3 center() const override;
        void set_center(pd::SimVector3 center) override;

    private:
        pd::SimVector3 center_point;
        pd::SimScalar main_radius;
        pd::SimScalar ring_radius;
    };
}