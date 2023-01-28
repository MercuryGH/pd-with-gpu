#pragma once

#include <pd/constraint.h>

#include <vector>
#include <Eigen/Core>

namespace pd
{
	using Positions = Eigen::MatrixXd;
	using Masses = Eigen::VectorXd;
	using Velocities = Eigen::MatrixXd; 
	using Faces = Eigen::MatrixXi;
	using Elements = Eigen::MatrixXi;
	using Constraints = std::vector<std::unique_ptr<pd::Constraint>>;

	using Float = float; // scalar type
}