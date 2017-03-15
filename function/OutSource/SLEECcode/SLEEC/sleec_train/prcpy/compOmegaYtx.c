#include "mex.h"
#include <omp.h>
/* Computes sparse matrix (Y^T) vector (x) multiplication. 
   Sparse matrix is specified by non-zero elements Ynz and their indices I, J.
   Usage: Ytx=compOmegaYtx(m,n,Ynz,I,J,x).
   Written by: Prateek Jain (pjain@cs.utexas.edu) and Raghu Meka (raghu@cs.utexas.edu)
   Last updated: 10/29/09
*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){

  mwSize nz_size;
  int nz_idx;
  int m,n,i,j;
  double *Ynz, *Ytx, *x, *I, *J, *hisJ;
  m=*((double *)mxGetPr(prhs[0]));
  n=*((double *)mxGetPr(prhs[1]));
  nz_size=mxGetM(prhs[2]);
  Ynz=mxGetPr(prhs[2]);
  I=mxGetPr(prhs[3]);
  J=mxGetPr(prhs[4]);
  x=mxGetPr(prhs[5]);
  hisJ=mxGetPr(prhs[6]);
  plhs[0]=mxCreateDoubleMatrix(n, 1, mxREAL);
  Ytx=mxGetPr(plhs[0]);
  for(i=0;i<n;i++){
    Ytx[i]=0;
  }
  //#pragma omp parallel for
//   for(nz_idx=0;nz_idx<nz_size;nz_idx++){
//     Ytx[(int)(J[nz_idx])-1]+=Ynz[nz_idx]*x[(int)(I[nz_idx])-1];
//   }
#pragma omp parallel for
  for(nz_idx = 0; nz_idx < m; nz_idx++)
  {
    long long row_idx;
     
     long long lim, curSum;
     if(nz_idx == 0)
     {
         lim = hisJ[nz_idx];
         curSum = 0;
     }
     else
     {
         lim = hisJ[nz_idx] - hisJ[nz_idx-1];
         curSum = hisJ[nz_idx-1];
     }
     for(row_idx=0;row_idx < lim;row_idx++)
     {
       Ytx[nz_idx] += Ynz[curSum + row_idx]*x[(int)(I[row_idx + curSum])-1];
     }
  }

}
