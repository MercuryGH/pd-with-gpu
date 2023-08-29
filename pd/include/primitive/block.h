#pragma once
#include <primitive/primitive.h>

namespace primitive
{
    class Block: public Primitive
    {
    public:
        Block(pd::SimVector3 center_point, pd::SimVector3 xyz): Primitive(PrimitiveType::BLOCK), center_point(center_point), half_xyz(0.5 * xyz) {}
        bool collision_handle(pd::SimVector3& pos) const override;
        void generate_visualized_model(Eigen::MatrixXd& V, Eigen::MatrixXi& F) const override;
        pd::SimVector3 center() const override;
        void set_center(pd::SimVector3 center) override;

    private:
        pd::SimVector3 center_point;
        pd::SimVector3 half_xyz;
    };
}
