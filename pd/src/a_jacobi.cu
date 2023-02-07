#include <pd/a_jacobi.h>

namespace pd
{
	void AJacobi::set_A(const Eigen::SparseMatrix<float>& A)
	{
	}

	Eigen::VectorXf AJacobi::solve(const Eigen::VectorXf& b)
	{
		return b;
	}
}