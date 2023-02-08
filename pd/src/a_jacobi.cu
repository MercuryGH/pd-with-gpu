#include <pd/a_jacobi.h>

namespace pd
{
	void AJacobi::set_A(const Eigen::SparseMatrix<float>& A)
	{
		// set precomputation values

		is_allocated = true;
	}

	Eigen::VectorXf AJacobi::solve(const Eigen::VectorXf& b)
	{
		return b;
	}

	void AJacobi::clear()
	{
		if (is_allocated)
		{
			// free memory

			is_allocated = false;
		}
	}
}