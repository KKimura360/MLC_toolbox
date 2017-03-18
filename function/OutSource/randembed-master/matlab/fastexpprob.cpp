#include "mex.h"
#include <cmath>
#include <cstdint>
#include <thread>
#include <emmintrin.h>

#define MATRIX_PARAMETER_IN    prhs[0]
#define SHIFT_PARAMETER_IN     prhs[1]

#define RESULT_OUT  plhs[0]

#include "fastexpsse.cc"

static void 
mexexpprob(float*y, mwSize m) {
  /* ensure y is aligned */

  while (m>0 && ((unsigned long)(y+m) & 15) != 0)
    {
      m--;
      y[m]=fastexp(y[m]);
      y[m]=y[m]/(1.0f+y[m]);
    }

  declconst128(one, 1.0f);

  while (m>7) {
    __m128 i1 = _mm_load_ps (y+m-4);
    __m128 i2 = _mm_load_ps (y+m-8);
    __m128 o1 = vfastexp(i1);
    __m128 o2 = vfastexp(i2);
    __m128 o3 = _mm_add_ps (o1, one);
    __m128 o4 = _mm_add_ps (o2, one);
    o1 = _mm_div_ps (o1, o3);
    o2 = _mm_div_ps (o2, o4);
    _mm_store_ps (y+m-4, o1);
    _mm_store_ps (y+m-8, o2);
    m=m-8;
  }
     
  while(m>0) {
    m--;
    y[m]=fastexp(y[m]);
    y[m]=y[m]/(1.0f+y[m]);
  }
}

static int first = 1;

void mexFunction( int nlhs, mxArray *plhs[], 
                  int nrhs, const mxArray*prhs[] ) { 
  float *y;
  mwSize m,n,curn; 

  if (first)
  {
      mexPrintf("fastexpprob using NUM_THREADS=%u\n",NUM_THREADS);
      first=0;
  }
  
  switch (nrhs)
  {
    case 1:
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

  mwSize quot = (m*n)/NUM_THREADS;

  for (int i = 0; i + 1 < NUM_THREADS; ++i) {
    t[i] = std::thread(mexexpprob, y + i * quot, quot);
  }

  mexexpprob (y + (NUM_THREADS - 1) * quot, 
              m*n - (NUM_THREADS - 1) * quot);

  for (int i = 0; i + 1 < NUM_THREADS; ++i) {
    t[i].join ();
  }

  return;
}
