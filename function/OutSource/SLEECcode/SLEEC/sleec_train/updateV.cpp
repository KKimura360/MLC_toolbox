#include "mex.h"
#include <omp.h>
#include <stdio.h>
#include <string.h>
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
  double *U, *V, *Omega, *b, *Omega_Val, *Vnew, *histI, *b2;
  double eta;
  int numThreads;
  long long m, n, k, Omega_size;
  long long i, j, Omega_idx, row_idx, col_idx;
  
  int l;
  
  m=mxGetM(prhs[0]);
  n=mxGetM(prhs[1]);
  k=mxGetN(prhs[0]);
  U=mxGetPr(prhs[0]);
  V=mxGetPr(prhs[1]);

  Omega_size=mxGetM(prhs[2]);
  Omega=mxGetPr(prhs[2]);
  Omega_Val=mxGetPr(prhs[3]);
  
  eta = *(mxGetPr(prhs[4]));
  
  histI = mxGetPr(prhs[5]);
  
  numThreads = *(mxGetPr(prhs[6]));
  
  plhs[0]=mxCreateDoubleMatrix(n, k, mxREAL);
  Vnew=mxGetPr(plhs[0]);
  
  plhs[1]=mxCreateDoubleMatrix(Omega_size, 1, mxREAL);
  b=mxGetPr(plhs[1]);
  
  memcpy(Vnew, V, sizeof(double)*n*k);
  
  //b = (double*)malloc(sizeof(double)*Omega_size);
  #pragma omp parallel for num_threads(numThreads)
  for(Omega_idx=0;Omega_idx<Omega_size;Omega_idx++){
    i=(long long)(((long long)Omega[Omega_idx]-1)%m);
    j=(long long)((Omega[Omega_idx]-1)/m);
    b[Omega_idx]=0;
    for(l=0;l<k;l++){
      b[Omega_idx]+=U[l*m+i]*V[l*n+j];
    }
    b[Omega_idx]-=Omega_Val[Omega_idx];
  }
  
 
  for(long long row_idx = 0; row_idx < n; row_idx ++)
  {   
      #pragma omp parallel for num_threads(numThreads)
      for(long long col_idx = 0; col_idx < k; col_idx ++)
      {
      long long lim = histI[row_idx+1];
      long long init = histI[row_idx];
        for (long l = init; l < lim; l++)
        {   
            long long j = (long long)((long long)(Omega[l]-1)%m);
            Vnew[col_idx*n+row_idx]-=(eta*U[col_idx*m+j]*b[l]);
        }
      }
  }
}
