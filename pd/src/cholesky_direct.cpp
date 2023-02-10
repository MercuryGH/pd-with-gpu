#include <pd/cholesky_direct.h>

namespace pd
{
	void CholeskyDirect::set_A(const Eigen::SparseMatrix<float>& A, const Constraints& constraints)
	{
		A_cholesky_decomp.compute(A);
	}

	Eigen::VectorXf CholeskyDirect::solve(const Eigen::VectorXf& b) 
	{
		Eigen::VectorXf ret = A_cholesky_decomp.solve(b);
		assert(A_cholesky_decomp.info() == Eigen::Success);
		return ret;
	}

	void CholeskyDirect::clear()
	{ 
		// do nothing 
	}
}