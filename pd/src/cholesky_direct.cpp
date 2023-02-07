#include <pd/cholesky_direct.h>

namespace pd
{
	void CholeskyDirect::set_A(const Eigen::SparseMatrix<float>& A) 
	{
		A_cholesky_decomp.compute(A);
	}

	Eigen::VectorXf CholeskyDirect::solve(const Eigen::VectorXf& b) 
	{
		return A_cholesky_decomp.solve(b);
	}
}