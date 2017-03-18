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
 * mex OPTIMFLAGS="/O2" -largeArrayDims -lmwblas -DNUM_THREADS=4 sparsequad.cpp
 * 
 * == linux ==
 * 
 * mex OPTIMFLAGS="-O3" -largeArrayDims -DNUM_THREADS=4 -v CXXFLAGS='$CXXFLAGS -std=c++0x -fPIC' sparsequad.cpp
 */


#define LTIC_MATRIX_PARAMETER_IN        prhs[0]
#define DIAG_W_VECTOR_PARAMETER_IN      prhs[1]
#define RTIC_MATRIX_PARAMETER_IN        prhs[2]
#define ZTIC_MATRIX_PARAMETER_IN        prhs[3]

// Y = Z'*R'*diag(W)*L

template<typename scalar>
static void
sparsequad (const mxArray* prhs[],
            scalar*        Y,
            scalar*        ZticR,
            size_t         start,
            size_t         end)
{
  mwIndex* Lticir = mxGetIr(LTIC_MATRIX_PARAMETER_IN);
  mwIndex* Lticjc = mxGetJc(LTIC_MATRIX_PARAMETER_IN);
  double* Ltics = mxGetPr(LTIC_MATRIX_PARAMETER_IN);

  double* W = mxGetPr(DIAG_W_VECTOR_PARAMETER_IN);

  mwIndex* Rticir = mxGetIr(RTIC_MATRIX_PARAMETER_IN);
  mwIndex* Rticjc = mxGetJc(RTIC_MATRIX_PARAMETER_IN);
  double* Rtics = mxGetPr(RTIC_MATRIX_PARAMETER_IN);

  size_t k = mxGetM(ZTIC_MATRIX_PARAMETER_IN);
  scalar* Ztic = (scalar*) mxGetData(ZTIC_MATRIX_PARAMETER_IN);

  for (size_t n = start; n < end; ++n) {
    double wi = W[n];

    memset (ZticR, 0, k * sizeof (scalar));

    mwIndex Rticstop = Rticjc[n + 1];
    for (mwIndex j = Rticjc[n]; j < Rticstop; ++j) {
      scalar* Zticdr = Ztic + Rticir[j] * k;
      double Rs = wi * Rtics[j];

      for (size_t i = 0; i < k; ++i) {
        ZticR[i] += Rs * Zticdr[i];
      }
    }

    mwIndex Lticstop = Lticjc[n + 1];

    for (mwIndex j = Lticjc[n]; j < Lticstop; ++j) {
      scalar* Yout = Y + Lticir[j] * k;
      double Ls = Ltics[j];

      for (size_t i = 0; i < k; ++i) {
        Yout[i] += Ls * ZticR[i];
      }
    }
  }
}

static int first = 1;

void mexFunction( int nlhs, mxArray *plhs[], 
                  int nrhs, const mxArray*prhs[] ) { 

  bool singlePrec=false;

  if (first) {
    mexPrintf("sparsequad using NUM_THREADS=%u\n",NUM_THREADS);
    first=0;
  }

  switch (nrhs) {
    case 4:
      if (! mxIsSparse (LTIC_MATRIX_PARAMETER_IN) || 
          mxIsSparse (DIAG_W_VECTOR_PARAMETER_IN) || 
          ! mxIsSparse (RTIC_MATRIX_PARAMETER_IN) || 
          mxIsSparse (ZTIC_MATRIX_PARAMETER_IN)) {
        mexErrMsgTxt("Parameters must be sparse, dense, sparse, and dense.");
        return;
      }

      if (! mxIsDouble (DIAG_W_VECTOR_PARAMETER_IN)) {
        mexErrMsgTxt("Weights must be double precision (for now).");
        return;
      }

      if (mxIsDouble (ZTIC_MATRIX_PARAMETER_IN)) {
        singlePrec=false;
      }
      else if (mxIsSingle (ZTIC_MATRIX_PARAMETER_IN)) {
        singlePrec=true;
      }
      else {
        mexErrMsgTxt("Last dense parameter (Z) must be either double or single precision.");
        return;
      }

      if (mxGetN(ZTIC_MATRIX_PARAMETER_IN) != mxGetM(RTIC_MATRIX_PARAMETER_IN)) {
        mexErrMsgTxt("Ztic and Rtic have incompatible shape.");
        return;
      }

      if (mxGetN(RTIC_MATRIX_PARAMETER_IN) != mxGetN(DIAG_W_VECTOR_PARAMETER_IN)) {
        mexErrMsgTxt("Rtic and W have incompatible shape.");
        return;
      }

      if (mxGetN(DIAG_W_VECTOR_PARAMETER_IN) != mxGetN(LTIC_MATRIX_PARAMETER_IN)) {
        mexErrMsgTxt("W and Ltic have incompatible shape.");
        return;
      }

      break;

    default:
      mexErrMsgTxt("Wrong number of arguments.");
      return;
  }

  size_t k = mxGetM(ZTIC_MATRIX_PARAMETER_IN);
  size_t dl = mxGetM(LTIC_MATRIX_PARAMETER_IN);
  size_t n = mxGetN(LTIC_MATRIX_PARAMETER_IN);

  plhs[0] = mxCreateNumericMatrix (k, dl, (singlePrec) ? mxSINGLE_CLASS : mxDOUBLE_CLASS, mxREAL);
  void* Y = mxGetData(plhs[0]);
  std::thread t[NUM_THREADS];
  void* s[NUM_THREADS];
  size_t quot = n/NUM_THREADS;

  for (size_t i = 0; i + 1 < NUM_THREADS; ++i) {
    s[i] = mxCalloc((dl + 1) * k, 
                    (singlePrec) ? sizeof(float) : sizeof(double));
  }

  void* ZticR = mxCalloc (k, (singlePrec) ? sizeof (float) : sizeof(double));

  if (singlePrec) {
    for (size_t i = 0; i + 1 < NUM_THREADS; ++i) {
      float* si = (float*) s[i];
      t[i] = std::thread(sparsequad<float>,
                         prhs,
                         si + k,
                         si,
                         i * quot,
                         (i + 1) * quot);
    }
    sparsequad (prhs, (float*) Y, (float*) ZticR, (NUM_THREADS - 1) * quot, n);
  }
  else {
    for (size_t i = 0; i + 1 < NUM_THREADS; ++i) {
      double* si = (double*) s[i];
      t[i] = std::thread(sparsequad<double>,
                         prhs,
                         si + k,
                         si,
                         i * quot,
                         (i + 1) * quot);
    }
    sparsequad (prhs, (double*) Y, (double*) ZticR, (NUM_THREADS - 1) * quot, n);
  }

  mxFree (ZticR);

  for (int i = 0; i + 1 < NUM_THREADS; ++i) {
    ptrdiff_t one = 1;
    ptrdiff_t dltimesk = dl * k;

    t[i].join ();
    if (singlePrec) {
      float onef = 1.0f;
      float* si = (float*) s[i];
      saxpy (&dltimesk, &onef, si + k, &one, (float*) Y, &one);
    }
    else {
      double oned = 1.0;
      double* si = (double*) s[i];
      daxpy (&dltimesk, &oned, si + k, &one, (double*) Y, &one);
    }

    mxFree(s[i]);
  }
}
