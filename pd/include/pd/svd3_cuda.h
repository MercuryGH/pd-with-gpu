/**************************************************************************
**
**  svd3
**
**  Quick singular value decomposition as described by:
**  A. McAdams, A. Selle, R. Tamstorf, J. Teran and E. Sifakis,
**  Computing the Singular Value Decomposition of 3x3 matrices
**  with minimal branching and elementary floating point operations,
**  University of Wisconsin - Madison technical report TR1690, May 2011
**
**	Identical GPU version
** 	Implementated by: Kui Wu
**	kwu@cs.utah.edu
**
**  May 2018
**
**************************************************************************/

#pragma once

#include <cuda_runtime.h>

namespace pd {

#define gone					1065353216
#define gsine_pi_over_eight		1053028117
#define gcosine_pi_over_eight   1064076127
#define gone_half				0.5f
#define gsmall_number			1.e-12f
#define gtiny_number			1.e-20f
#define gfour_gamma_squared		5.8284273147583007813f

union un { float f; unsigned int ui; };

__device__
void svd(
	float a11, float a12, float a13, float a21, float a22, float a23, float a31, float a32, float a33,			// input A
    float &u11, float &u12, float &u13, float &u21, float &u22, float &u23, float &u31, float &u32, float &u33,	// output U
	float &s11,
	//float &s12, float &s13, float &s21,
	float &s22,
	//float &s23, float &s31, float &s32,
	float &s33,	// output S
	float &v11, float &v12, float &v13, float &v21, float &v22, float &v23, float &v31, float &v32, float &v33	// output V
);

}
