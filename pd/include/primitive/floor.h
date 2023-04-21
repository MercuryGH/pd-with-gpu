#pragma once
#include <primitive/primitive.h>

namespace primitive
{
    class Floor: public Primitive
    {
    public:
        Floor(pd::SimScalar y): Primitive(PrimitiveType::FLOOR), center_point(pd::SimVector3(0, y, 0)) {}
        bool collision_handle(pd::SimVector3& pos) const override;
        void generate_visualized_model(Eigen::MatrixXd& V, Eigen::MatrixXi& F) const override;
        pd::SimVector3 center() const override;
        void set_center(pd::SimVector3 center) override;

    private:
        pd::SimVector3 center_point;
    };
}