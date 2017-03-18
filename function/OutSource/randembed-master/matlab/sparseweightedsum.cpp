#include "mex.h"
#include "blas.h"
#include "matrix.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <emmintrin.h>

/*
 * compilation, at matlab prompt: (adjust NUM_THREADS as appropriate)
 * 
 * == windows ==
 * 
 * mex OPTIMFLAGS="/O2" -largeArrayDims -DNUM_THREADS=4 sparseweightedsum.cpp
 * 
 * == linux ==
 * 
 * mex OPTIMFLAGS="-O3" -largeArrayDims -DNUM_THREADS=4 -v CXXFLAGS='$CXXFLAGS -std=c++0x -fPIC' sparseweightedsum.cpp
 */


#define L_MATRIX_PARAMETER_IN           prhs[0]
#define DIAG_W_VECTOR_PARAMETER_IN      prhs[1]
#define POW_PARAMETER_IN                prhs[2]

// Y = sum(bsxfun(@times,L.^p,W),2), p == 1 or p == 2

static void
sparseweightedsum (const mxArray* prhs[],
                   double*        Y,
                   size_t         start,
                   size_t         end)
{
  mwIndex* Lir = mxGetIr(L_MATRIX_PARAMETER_IN);
  mwIndex* Ljc = mxGetJc(L_MATRIX_PARAMETER_IN);
  double* Ls = mxGetPr(L_MATRIX_PARAMETER_IN);

  double* W = mxGetPr(DIAG_W_VECTOR_PARAMETER_IN);

  int pow = mxGetScalar(POW_PARAMETER_IN);

  if (pow == 1) {
    for (size_t n = start; n < end; ++n) {
      double wi = W[n];

      mwIndex Lstop = Ljc[n + 1];
      for (mwIndex j = Ljc[n]; j < Lstop; ++j) {
        Y[Lir[j]] += wi * Ls[j];
      }
    }
  }
  else {
    for (size_t n = start; n < end; ++n) {
      double wi = W[n];

      mwIndex Lstop = Ljc[n + 1];
      for (mwIndex j = Ljc[n]; j < Lstop; ++j) {
        Y[Lir[j]] += wi * Ls[j] * Ls[j];
      }
    }
  }
}

static int first = 1;

void mexFunction( int nlhs, mxArray *plhs[], 
                  int nrhs, const mxArray*prhs[] ) { 
  if (first) {
    mexPrintf("sparseweightedsum using NUM_THREADS=%u\n",NUM_THREADS);
    first=0;
  }

  switch (nrhs) {
    case 3:
      if (! mxIsSparse (L_MATRIX_PARAMETER_IN) || 
          mxIsSparse (DIAG_W_VECTOR_PARAMETER_IN) || 
          mxGetM (POW_PARAMETER_IN) != 1 || 
          mxGetN (POW_PARAMETER_IN) != 1) {
        mexErrMsgTxt("Parameters must be sparse, dense, and a scalar.");
        return;
      }

      if (mxGetN(L_MATRIX_PARAMETER_IN) != mxGetN(DIAG_W_VECTOR_PARAMETER_IN)) {
        mexErrMsgTxt("L and W have incompatible shape.");
        return;
      }

      if (mxGetScalar(POW_PARAMETER_IN) != 1 &&
          mxGetScalar(POW_PARAMETER_IN) != 2) {
        mexErrMsgTxt("pow must be 1 or 2.");
        return;
      }

      break;

    default:
      mexErrMsgTxt("Wrong number of arguments.");
      return;
  }

  size_t dl = mxGetM(L_MATRIX_PARAMETER_IN);
  size_t n = mxGetN(L_MATRIX_PARAMETER_IN);

  plhs[0] = mxCreateDoubleMatrix(1, dl, mxREAL);
  double* Y = mxGetPr(plhs[0]);

  std::thread t[NUM_THREADS];
  double* s[NUM_THREADS];
  size_t quot = n/NUM_THREADS;

  for (size_t i = 0; i + 1 < NUM_THREADS; ++i) {
    s[i] = (double*) mxCalloc(dl, sizeof(double));
  }

  for (size_t i = 0; i + 1 < NUM_THREADS; ++i) {
    t[i] = std::thread(sparseweightedsum,
                       prhs,
                       s[i],
                       i * quot,
                       (i + 1) * quot);

  }

  sparseweightedsum (prhs, Y, (NUM_THREADS - 1) * quot, n);

  for (int i = 0; i + 1 < NUM_THREADS; ++i) {
    double oned = 1.0;
    ptrdiff_t one = 1;
    ptrdiff_t dlmega = dl;

    t[i].join ();
    daxpy (&dlmega, &oned, s[i], &one, Y, &one);
    mxFree (s[i]);
  }

  return;
}
