#include <mex.h>
#include <matrix.h>
#include <lapack.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    ptrdiff_t n,p;      /* matrix dimensions */ 
  
 	/* Check for proper arguments. */
    if ( nrhs != 2 || mxIsSingle(prhs[0]) != mxIsSingle(prhs[1]) ) {
        mexErrMsgIdAndTxt("MATLAB:chosolve:rhs",
            "This function requires 2 input matrices of same precision.");
    }

    /* dimensions of input matrices */
    n = mxGetM(prhs[0]);
    p = mxGetN(prhs[1]);

    /* Validate input arguments */
    if (n != mxGetN(prhs[0])) {
        mexErrMsgIdAndTxt("MATLAB:chosolve:square",
            "LAPACK function requires input matrix 1 must be square.");
    }

    if (n != mxGetM(prhs[1])) {
        mexErrMsgIdAndTxt("MATLAB:chosolve:matchdims",
            "Input matrices should have the same number of rows.");
    }

    char uplo = 'L';
    ptrdiff_t info = 0;

    if ( mxIsSingle(prhs[0]) ) {
        float *SA = (float*) mxGetPr(prhs[0]); /* pointer to first input matrix */
        float *SB = (float*) mxGetPr(prhs[1]); /* pointer to second input matrix */
        /* Call LAPACK */
        spotrs(&uplo, &n, &p, SA, &n, SB, &n, &info);
    }
    else {
        double *DA = (double*) mxGetPr(prhs[0]); /* pointer to first input matrix */
        double *DB = (double*) mxGetPr(prhs[1]); /* pointer to second input matrix */
        /* Call LAPACK */
        dpotrs(&uplo, &n, &p, DA, &n, DB, &n, &info);
    }

    /* Check result of function call */
    if(info != 0)
    {   
        mexPrintf("Return code of function [sd]potrs: %d\n", info);
        mexErrMsgTxt("Error using LAPACK function.");
    }

}
