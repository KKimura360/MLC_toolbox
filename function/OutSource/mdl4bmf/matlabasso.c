#include <string.h>
#include <math.h>
#include <limits.h>
#include "mex.h"
#include "approx.h"
#include "utils.h"

/*
 * Helping function sab, computes error
 */

unsigned long int
sab(matrix A, matrix B, matrix C, int n, int m, int k);

/*
 * Input arguments: <data matrix> <k> <threshold> [<bonus>]
 *
 * <bonus> defaults to 1, and denotes how much covering 1s is appreciated
 * over covering 0s.
 *
 * <k> must be a non-negative integer.
 *
 * <threshold> must be a real number between 0 and 1.
 *
 * <threshold> and <bonus> can be row vectors, in which case all
 * values of the vectors are tried, and the returned answer is the
 * best over all restarts.
 *
 * Output arguments: <B> [<X>] [<error>] [<best threshold>] [<best bonus 1>]
 *                   [<best penalty 0>]
 *
 * <B> and <X> are matrices such that <data matrix>' ~ (<B>' o <X>')'.
 *
 * <error> reports the error. 
 *
 * <best threshold> returns the threshold used to get <error>.
 *
 * <best bonus 1> and <best penalty 0> report values s.t. 
 * <bonus>=<best bonus 1>/<best penalty 0> was used to get <error>.
 */

/* The gateway function */
void 
mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[])
{
  const double DEFAULT_B = 1.0;
  mwSize n, m, t_length, b_length;
  int i, j, k, ti, bi, bestBonusCovered, bestPenaltyOvercovered;
  unsigned long int error, bestError;
  double *t, bonus, *b, bp, bn, *Dorig, *Bfinal, *Sfinal, bestThreshold;
  matrix D, S, B, bestS, bestB;
  options opti = {0,     /* error_max */
		  10,    /* cut_size */
		  0,     /* noisy_vectors */
		  0,     /* iterations */
		  1,     /* remove_covered */
		  0,     /* seed */
		  0,     /* verbose */
		  NULL,  /* original_basis */
		  1.0,   /* threshold */
		  0,     /* majority */
		  1,     /* bonus_covered */
		  1,     /* penalty_overcovered */
		  NULL}; /* decomp matrix */



  /* Check input and output argument numbers */
  if (nrhs > 4 || nrhs < 3) {
    mexErrMsgTxt("Three or four input arguments required.");
  }
  if (nlhs > 6) {
    mexErrMsgTxt("Too many output arguments.");
  }
  
  /* Check and load input arguments */
  /* Data matrix D */
  Dorig = mxGetPr(prhs[0]);
  n = mxGetM(prhs[0]);
  m = mxGetN(prhs[0]);  

  /* rank k */
  if (!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) ||
      mxGetM(prhs[1])*mxGetN(prhs[1]) != 1)
    mexErrMsgTxt("Second input must be a real scalar.");
  k = (int)mxGetScalar(prhs[1]);


  /* threshold t */
  if (!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) ||
      mxGetM(prhs[2]) != 1)
    mexErrMsgTxt("Third input must be a real scalar or row vector.");
  t_length = mxGetN(prhs[2]);
  t = mxGetPr(prhs[2]);
  /*opti.threshold = mxGetScalar(prhs[2]);*/

  if (nrhs == 4) {
    /* bonus b */
    if (!mxIsDouble(prhs[3]) || mxIsComplex(prhs[3]) ||
	mxGetM(prhs[3]) != 1)
      mexErrMsgTxt("Fourth input must be a real scalar or row vector.");
    b_length = mxGetN(prhs[3]); 
    b = mxGetPr(prhs[3]);
  } else { 
    b_length = 1;
    b = &DEFAULT_B; 
  }


  /* Convert Dorig to proper form */
  
  /* allocate memory and fill the matrix*/

  D = (matrix)mxMalloc(n * sizeof(vector));
  for (i=0; i<n; i++) {
    D[i] = (vector)mxMalloc(m * sizeof(char));
    memset(D[i], 0, m);
    for (j=0; j<m; j++) {
      if (Dorig[j*n + i] == 1)
	D[i][j] = 1;
      else if ((int)Dorig[j*n + i] != 0)
	mxErrMsgTxt("Input matrix must be 0/1.");
    }
  }

  /* Allocate memory for output arguments */

  B = (matrix)mxMalloc(k * sizeof(vector));
  for (i=0; i<k; i++) {
    B[i] = (vector)mxMalloc(m * sizeof(char));
    memset(B[i], 0, m);
  }

  S = (matrix)mxMalloc(n * sizeof(vector));
  for (i=0; i<n; i++) {
    S[i] = (vector)mxMalloc(k * sizeof(char));
    memset(S[i], 0, k);
  }

  bestB = (matrix)mxMalloc(k * sizeof(vector));
  for (i=0; i<k; i++) {
    bestB[i] = (vector)mxMalloc(m * sizeof(char));
    memset(bestB[i], 0, m);
  }

  bestS = (matrix)mxMalloc(n * sizeof(vector));
  for (i=0; i<n; i++) {
    bestS[i] = (vector)mxMalloc(k * sizeof(char));
    memset(bestS[i], 0, k);
  }

  /* make bestError big enough */
  bestError = ULONG_MAX;
  bestThreshold = -1;
  bestBonusCovered = -1;
  bestPenaltyOvercovered = -1;

  /* Iterate over all values of t and bonus */
  for (ti=0; ti<t_length; ti++) {
    /* set threshold */
    opti.threshold = t[ti];
    for (bi=0; bi<b_length; bi++) {

      /* Divide b to positive and negative bonus, b = bp / bn */
      /*bn = modf(b[bi], &bp);*/

      if (b[bi] >= 0) {bp = b[bi]; bn = 1;}
      else {bp = 1; bn = -1*b[bi];}      

      /* DEBUG */
      #ifdef DEBUG
      mexPrintf("b[bi] = %f, bn = %f, bp = %f, 1/bn=%f\n", b[bi], bn, bp, 1/bn);
      #endif
      opti.bonus_covered = (bp < 1)?1:(int)bp;
      opti.penalty_overcovered = (bn < 1)?1:(int)bn;
      /*
      if (bn > 0)
	opti.penalty_overcovered = lround(1.0/bn);
      else
	opti.penalty_overcovered = 1;
      */
      /* DEBUG */
      #ifdef DEBUG
      mexPrintf("t = %f, b_covered = %i, b_overcovered = %i\n", opti.threshold,\
		opti.bonus_covered, opti.penalty_overcovered);
      #endif

      if (approximate(D, n, m, B, k, S, &opti) != 1) 
	mexErrMsgTxt("Asso failed.");

      error = sab(D, S, B, n, m, k);

      /* DEBUG */
      #ifdef DEBUG
      mexPrintf("bestError = %lu, error = %lu\n", bestError, error);
      #endif

      if (error < bestError) {
	bestError = error;
	bestThreshold = opti.threshold;
	bestBonusCovered = opti.bonus_covered;
	bestPenaltyOvercovered = opti.penalty_overcovered;
	for (i=0; i<k; i++) {
	  memcpy(bestB[i], B[i], m);
	}
	for (i=0; i<n; i++) {
	  memcpy(bestS[i], S[i], k);
	}
	/* DEBUG */
	#ifdef DEBUG
	mexPrintf("bestError = %lu was bigger than error = %lu\n", \
		  bestError, error);
	#endif
	
      }

      /* Clear B and S */
      for (i=0; i<k; i++) memset(B[i], 0, m);
      for (i=0; i<n; i++) memset(S[i], 0, k);
    }
  }
	

  /* Initialize output, by default return B */

  plhs[0] = mxCreateDoubleMatrix(k, m, mxREAL);
  Bfinal = mxGetPr(plhs[0]);
  memset(Bfinal, 0, k*m);
  for (i=0; i<k; i++) {
    for (j=0; j<m; j++) {
      if (bestB[i][j]) Bfinal[k*j + i] = 1.0;
    }
  }

  /* if S is also asked... */
  if (nlhs >= 2) {
    plhs[1] = mxCreateDoubleMatrix(n, k, mxREAL);
    Sfinal = mxGetPr(plhs[1]);
    memset(Sfinal, 0, n*k);
    for (i=0; i<n; i++) {
      for (j=0; j<k; j++) {
	if (bestS[i][j]) Sfinal[n*j + i] = 1.0;
      }
    }
  }

  /* if also error is asked ... */
  if (nlhs >= 3) {
    plhs[2] = mxCreateDoubleScalar((double)bestError);
  }

  /* if also best threshold is asked ... */
  if (nlhs >= 4) {
    plhs[3] = mxCreateDoubleScalar(bestThreshold);
  }

  /* if also best bonus 1 is asked ... */
  if (nlhs >= 5) {
    plhs[4] = mxCreateDoubleScalar((double)bestBonusCovered);
  }

  /* if also best penalty 1 is asked ... */
  if (nlhs == 6) {
    plhs[5] = mxCreateDoubleScalar((double)bestPenaltyOvercovered);
  }
  
  /* And that's all folks */
}


unsigned long int
sab(matrix A, matrix B, matrix C, int n, int m, int k)
{
  int i, j, l, set;
  unsigned long int error = 0;
  
  for (i=0; i<n; i++) {
    for (j=0; j<m; j++) {
      set = 0;
      for (l=0; l<k; l++) {
	if (B[i][l] == 1 && C[l][j] == 1) {
	  if (!A[i][j]) error++; /* A[i,j]=0, (BoC)[i,j] = 1 */
	  set = 1;
	  break;
	}
      }
      if (!set && A[i][j]) error++; /* A[i,j] = 1; (BoC)[i,j] = 0 */
    }
  }
  return error;
}
