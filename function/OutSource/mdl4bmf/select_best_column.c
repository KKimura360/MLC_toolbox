#include <string.h>
#include <stdint.h>
#include "mex.h"
#include "matrix.h"

#define element(rows, row, col) (rows*col + row)
#define MAX(a,b) ((a>b)?a:b)
#define MIN(a,b) ((a<b)?a:b)

static long
cover(const uint8_t * restrict A, 
      const mwSize n, 
      const mwSize m, 
      const uint8_t * restrict mask, 
      const uint8_t * restrict C, 
      const mwSize j, 
      long long * restrict tmpR);
  
/*
 * [id, r] = select_best_column(A, C, mask)
 *
 * Selects the best column of C to add to an BMF of A of which mask elements 
 * are already covered. Returns the index of the column in C, and r, the row 
 * to define the use of the column.
 *
 */ 

void 
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  mwSize n, m, c, foo;
  double * restrict id, * restrict r; 
  long long * restrict tmpR;
  uint8_t * restrict A, * restrict C, * restrict mask;
  mwSize i, j;
  long tmpCoverval, coverval;

  /* Check for input and output argument numbers */
  if (nrhs != 3) 
    mexErrMsgTxt("Three input arguments, A, C, and mask, required.");
  if (nlhs != 2)
    mexErrMsgTxt("Two output arguments required.");
  
  /* Check and load input matrices. */
  if (!mxIsUint8(prhs[0]))
    mexErrMsgTxt("Input matrices must be of uint8_t; first is not.");
  A = (uint8_t *)mxGetData(prhs[0]);
  n = mxGetM(prhs[0]);
  m = mxGetN(prhs[0]);
  
  if (!mxIsUint8(prhs[1]))
    mexErrMsgTxt("Input matrices must be of uint8_t; second is not.");
  C = (uint8_t *)mxGetData(prhs[1]);
  foo = mxGetM(prhs[1]);
  if (foo != n) 
    mexErrMsgTxt("Matrix C must have same number of rows as A.");
  c = mxGetN(prhs[1]);
  
  if (!mxIsUint8(prhs[2]))
    mexErrMsgTxt("Input matrices must be of uint8_t; third is not.");
  mask = (uint8_t *)mxGetData(prhs[2]);
  foo = mxGetM(prhs[2]);
  if (foo != n)
    mexErrMsgTxt("Mask must have same number of rows as A.");
  foo = mxGetN(prhs[2]);
  if (foo != m)
    mexErrMsgTxt("Mask must have same number of columns as A.");
  


  /* Output argument id */
  plhs[0] = mxCreateDoubleScalar(mxGetNaN());
  id = mxGetPr(plhs[0]);
  
  /* Output argument r */
  plhs[1] = mxCreateDoubleMatrix(1, m, mxREAL);
  r = mxGetPr(plhs[1]);
  
  /* Initialize temporary row and "best" covervalue */
  coverval = -1; /* by definition, cover() returns values >= 0. */
  tmpR = (long long *)mxMalloc(m*sizeof(long long));
  /* Try each column */
  
  for (i=0; i<c; i++) {
    tmpCoverval = cover(A, n, m, mask, C, i, tmpR);
    if (tmpCoverval > coverval) {
      coverval = tmpCoverval;
      id[0] = (double)(i+1); /* correct for Matlab-style indices */
      for (j=0; j < m; j++) /* copy the row tmpR to r*/
	r[j] = (double)tmpR[j];
    }
  }
}


/*
 * Return the overall cover value for column c of C and the corresponding 
 * usage row in tmpR.
 */
static long
cover(const uint8_t * restrict A, 
      const mwSize n, 
      const mwSize m, 
      const uint8_t * restrict mask, 
      const uint8_t * restrict C, 
      const mwSize idx, 
      long long * restrict tmpR)
{
  mwSize i, j;
  long coverval;
 
  /* Empty tmpR */
  /* Obsolete, will use v instead */
  /* memset(tmpR, 0, m*sizeof(long long)); */

  for (j=0; j<m; j++) {
    /* For all columns of A */
    long long v = 0;
    for (i=0; i<n; i++) {
      /* How good column idx is in covering column j */
      if (mask[element(n, i, j)] == 0 && A[element(n, i, j)] == 1 
	  && C[element(n, i, idx)] == 1) {
	/* A(i,j) = 1 and C(i,idx) = 1, and A(i,j) is not covered */
	//tmpR[j]++;
	v++;
      } else if (mask[element(n, i, j)] == 0 && A[element(n,i,j)] == 0
		 && C[element(n,i,idx)] == 1) {
	/* This would cover an uncovered zero element */
	//tmpR[j]--;
	v--;
      }
    }
    tmpR[j] = v;
  }

  /* Now we know, for each col of A, should we use idx to cover it or not */
  /* Let's update tempR and sum up the coverval */

  coverval = 0;

  for (j=0; j<m; j++) {
    if (tmpR[j] > 0) {
      /* Use idx to cover j */
      coverval += tmpR[j]; /* add to coverval */
      tmpR[j] = 1; /* Booleanize tmpR */
    } else {
      tmpR[j] = 0; /* Booleanize tmpR */
    }
  }
  return coverval;
}
