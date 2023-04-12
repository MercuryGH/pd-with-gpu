#pragma once

#include <vector>
#include <Eigen/Core>

namespace meshgen {
	// visitor pattern
	std::pair<Eigen::MatrixXd, Eigen::MatrixXi> generate_plane(int x, int y, int usub, int vsub);

	std::pair<Eigen::MatrixXd, Eigen::MatrixXi> generate_cloth(int n_rows, int n_cols);

	std::pair<Eigen::MatrixXd, Eigen::MatrixXi> generate_sphere(float radius, int usub=20, int vsub=16, float urange=1.0f, float vrange=1.0f);
	std::pair<Eigen::MatrixXd, Eigen::MatrixXi> generate_hemisphere(float radius, int usub=20, int vsub=16);

	std::pair<Eigen::MatrixXd, Eigen::MatrixXi> generate_cylinder(float radius, float height, int usub=20, int vsub=16, int capsub=0, float urange=1.0f, float vrange=1.0f);
	std::pair<Eigen::MatrixXd, Eigen::MatrixXi> generate_cone(float radius, float height, int usub=20, int vsub=8, int capsub=0, float urange=1.0f, float vrange=1.0f);

	std::pair<Eigen::MatrixXd, Eigen::MatrixXi> generate_torus(float main_radius, float ring_radius, int usub=32, int vsub=16, float urange=1.0f, float vrange=1.0f);

	std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXi> generate_bar(int x, int y, int z, int usub=2, int vsub=2);
}
