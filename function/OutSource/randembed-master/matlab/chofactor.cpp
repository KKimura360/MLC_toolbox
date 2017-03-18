#include <mex.h>
#include <matrix.h>
#include <lapack.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    ptrdiff_t n;     /* matrix dimension */ 
  
 	/* Check for proper number of arguments. */
    if ( nrhs != 1 ) {
        mexErrMsgIdAndTxt("MATLAB:chofactor:rhs",
            "This function requires 1 input matrix.");
    }

    /* dimension of input matrix */
    n = mxGetN(prhs[0]);

    /* Validate input arguments */
    if (n != mxGetM(prhs[0])) {
        mexErrMsgIdAndTxt("MATLAB:chofactor:square",
            "LAPACK function requires square input matrix.");
    }

    char uplo = 'L';
    ptrdiff_t info = 0;

    if ( mxIsSingle(prhs[0]) ) {
        float* S = (float*) mxGetPr(prhs[0]); /* pointer to input matrix */
        /* Call LAPACK */
        spotrf(&uplo,&n,S,&n,&info);
    }
    else {
        double* D = (double*) mxGetPr(prhs[0]); /* pointer to input matrix */ 
        /* Call LAPACK */
        dpotrf(&uplo,&n,D,&n,&info);
    }

    /* Check result of function call */
    if(info != 0)
    {   
        mexPrintf("Return code of function [sd]potrf: %d\n", info);
        mexErrMsgTxt("Error using LAPACK function.");
    }
}
