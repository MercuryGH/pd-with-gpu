#pragma once

#include <memory>

#include <vector>
#include <Eigen/Core>

namespace pd
{
	using VertexIndexType = int;
	using MeshIDType = int;
	using Float = float; // scalar type
	using Double = double;
	using DataScalar = Double;
	using SimScalar = Double;

	using DataVector3 = Eigen::Matrix<DataScalar, 3, 1>;
	using DataRowVector3 = Eigen::Matrix<DataScalar, 1, 3>;
	using DataMatrix3 = Eigen::Matrix<DataScalar, 3, 3>;
	using DataMatrixX3 = Eigen::Matrix<DataScalar, -1, 3>;

	using DataVectorX = Eigen::Matrix<DataScalar, -1, 1>;

	using SimVector3 = Eigen::Matrix<SimScalar, 3, 1>;
	using SimRowVector3 = Eigen::Matrix<SimScalar, 1, 3>;
	using SimMatrix3 = Eigen::Matrix<SimScalar, 3, 3>;
	using SimMatrixX3 = Eigen::Matrix<SimScalar, -1, 3>;
	using SimMatrixX = Eigen::Matrix<SimScalar, -1, -1>;

	using IndexRowVector3 = Eigen::Matrix<VertexIndexType, 1, 3>;
	using IndexRowVector4 = Eigen::Matrix<VertexIndexType, 1, 4>;

	using SimPositions = Eigen::Matrix<SimScalar, -1, 1>;
	using SimVectorX = Eigen::Matrix<SimScalar, -1, 1>;

	using PositionData = Eigen::Matrix<DataScalar, -1, -1>;
	using MassData = Eigen::Matrix<DataScalar, -1, 1>;
	using VelocityData = Eigen::Matrix<DataScalar, -1, -1>;
	using FaceData = Eigen::Matrix<VertexIndexType, -1, 3>;
	using ElementData = Eigen::Matrix<VertexIndexType, -1, -1>;
}
