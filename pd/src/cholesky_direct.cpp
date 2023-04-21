#include <pd/cholesky_direct.h>

namespace pd
{
	void CholeskyDirect::set_A(const Eigen::SparseMatrix<SimScalar>& A, const std::unordered_map<MeshIDType, DeformableMesh>& models)
	{
		A_cholesky_decomp.compute(A);
	}

	SimVectorX CholeskyDirect::solve(const SimVectorX& b) 
	{
		SimVectorX ret = A_cholesky_decomp.solve(b);
		assert(A_cholesky_decomp.info() == Eigen::Success);
		return ret;
	}

	void CholeskyDirect::clear()
	{ 
		// do nothing 
	}
}