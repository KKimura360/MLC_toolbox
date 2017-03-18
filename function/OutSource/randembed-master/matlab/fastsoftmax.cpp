#include "mex.h"
#include "blas.h"
#include <cmath>
#include <cstdint>
#include <thread>

#if !defined(_WIN32)
#include <x86intrin.h>
#endif

/* these 2 #define lines help make the later code more readable */
/* Input Arguments */
#define MATRIX_PARAMETER_IN    prhs[0]
#define SHIFT_PARAMETER_IN     prhs[1]

/* Output Arguments */
#define RESULT_OUT  plhs[0]

#include "fastexpsse.cc"

static void 
mexsoftmax(float* y, float* shift, mwSize m, mwSize n) {
  __m128 i1, i2;
  __m128 o1, o2;
 
  while (m>0)
    {
      mwSize curn = n;
      float sum = 0.0f;
      declconst128(zero, 0.0f);
      
      while (curn>0 && ((unsigned long)(y+curn) & 15) != 0)
        {
          --curn;
          y[curn]=fastexp(y[curn]-*shift);
          sum += y[curn];
        }

      __m128 s1 = _mm_load1_ps (shift);
      __m128 sum1 = zero;

      while (curn>7) {
        i1 = _mm_load_ps (y+curn-4);
        i2 = _mm_load_ps (y+curn-8);
        i1 = _mm_sub_ps (i1, s1);
        i2 = _mm_sub_ps (i2, s1);
        o1 = vfastexp(i1);
        o2 = vfastexp(i2);
        _mm_store_ps (y+curn-4, o1);
        sum1 = _mm_add_ps (sum1, o1);
        _mm_store_ps (y+curn-8, o2);
        sum1 = _mm_add_ps (sum1, o2);
        curn-=8;
      }

      sum1 = _mm_hadd_ps (sum1, sum1);
      sum1 = _mm_hadd_ps (sum1, sum1);
      sum += _mm_cvtss_f32 (sum1);
     
      while(curn>0) {
        --curn;
        y[curn]=fastexp(y[curn]-*shift);
        sum += y[curn];
      }

      sum = 1.0f / sum;

      ptrdiff_t n_pdt = n;
      ptrdiff_t one_pdt = 1;

      sscal (&n_pdt, &sum, y, &one_pdt);

      ++shift;
      y+=n;
      --m;
    }
}

static int first = 1;

void mexFunction( int nlhs, mxArray *plhs[], 
                  int nrhs, const mxArray*prhs[] ) { 
  float *y;
  mwSize m,n,curn; 

  if (first)
  {
      mexPrintf("fastsoftmax using NUM_THREADS=%u\n",NUM_THREADS);
      first=0;
  }
  
  switch (nrhs)
  {
    case 2:
      if (! mxIsSingle(SHIFT_PARAMETER_IN)) {
        mexErrMsgTxt("Single precision shift argument required.");
        return;
      }

      if (mxGetN(SHIFT_PARAMETER_IN) != 1 && mxGetM(SHIFT_PARAMETER_IN) != 1) {
        mexErrMsgTxt("Shift needs to be a vector.");
        return;
      }

      if (mxGetN(SHIFT_PARAMETER_IN) * mxGetM(SHIFT_PARAMETER_IN) != 
          mxGetN(MATRIX_PARAMETER_IN)) {
        mexErrMsgTxt("Shift has incompatible shape.");
        return;
      }

      if (! mxIsSingle(MATRIX_PARAMETER_IN)) {
        mexErrMsgTxt("Single precision matrix argument required.");
        return;
      }
  
      break;
      
    default:
      mexErrMsgTxt("Incorrect number of input arguments provided."); 
      return;
  }
  
  m = mxGetM(MATRIX_PARAMETER_IN); 
  n = mxGetN(MATRIX_PARAMETER_IN);
  
  /* Cheat ... do things in place */
  RESULT_OUT = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL); 
  float* outptr = (float*) mxGetPr(RESULT_OUT);
  *outptr = 1.0;
  
  /* Assign pointers to the various parameters */ 
  y = (float*) mxGetPr(MATRIX_PARAMETER_IN);
      
  std::thread t[NUM_THREADS];

  float* shiftptr = (float*) mxGetPr(SHIFT_PARAMETER_IN);
  mwSize quot = (n/NUM_THREADS);

  for (int i = 0; i + 1 < NUM_THREADS; ++i) {
    t[i] = std::thread(mexsoftmax,
                       y + i * m * quot, 
                       shiftptr + i * quot,
                       quot, 
                       m);
  }

  mexsoftmax (y + (NUM_THREADS - 1) * m * quot, 
              shiftptr + (NUM_THREADS - 1) * quot,
              n - (NUM_THREADS - 1) * quot,
              m);

  for (int i = 0; i + 1 < NUM_THREADS; ++i) {
    t[i].join ();
  }

  return;
}
