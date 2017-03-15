#include "mex.h"
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <cstring>
#include "smat.h"
#include <algorithm>
#include <iostream>  
#include <omp.h>
#include <vector>

#define eps 0.1
#define min(a,b) a>b?b:a

using namespace std;

int transpose(const mxArray *M, mxArray **Mt) {
	mxArray *prhs[1] = {const_cast<mxArray *>(M)}, *plhs[1];
	if(mexCallMATLAB(1, plhs, 1, prhs, "transpose"))
	{
		mexPrintf("Error: cannot transpose training instance matrix\n");
		return -1;
	}
	*Mt = plhs[0];
	return 0;
}

class mxSparse_iterator_t: public entry_iterator_t {
	private:
		mxArray *Mt;
		mwIndex *ir_t, *jc_t;
		double *v_t;
		size_t	rows, cols, cur_idx, cur_row;
	public:
		mxSparse_iterator_t(const mxArray *M){
			rows = mxGetM(M); cols = mxGetN(M);
			nnz = *(mxGetJc(M) + cols); 
			transpose(M, &Mt);
			ir_t = mxGetIr(Mt); jc_t = mxGetJc(Mt); v_t = mxGetPr(Mt);
			cur_idx = cur_row = 0;
		}
		rate_t next() {
			int i = 1, j = 1;
			double v = 0;
			while (cur_idx >= jc_t[cur_row+1]) ++cur_row;
			if (nnz > 0) --nnz;
			else fprintf(stderr,"Error: no more entry to iterate !!\n");
			rate_t ret(cur_row, ir_t[cur_idx], v_t[cur_idx]);
			cur_idx++;
			return ret;
		}
		~mxSparse_iterator_t(){
			mxDestroyArray(Mt);
		}

};

smat_t mxSparse_to_smat(const mxArray *M, smat_t &R) {
	long rows = mxGetM(M), cols = mxGetN(M), nnz = *(mxGetJc(M) + cols); 
	mxSparse_iterator_t entry_it(M);
	R.load_from_iterator(rows, cols, nnz, &entry_it);
	return R;
}
struct valLocPair
{
	double val;
	int loc;
	short orig;
};
typedef struct valLocPair vlPair;

void formScoreMat(smat_t labels, double *nnMat, double *I, double *J, double *S, long long n, long long l, long long nt, long long k)
{	
	double *lbVec = (double *)malloc(sizeof(double)*l);
	long long *idxArr = (long long *)malloc(sizeof(double)*k*100);
	long long Icount = 0;
	for(long long testIter = 0; testIter < nt; testIter ++)
	{
		memset(lbVec, 0, sizeof(double)*l);
		memset(idxArr, 0, sizeof(double)*k*100);
		long long lblCount = 0;
		for(long long neighIter = 0; neighIter < k; neighIter ++)
		{	
			long long neighIdx = nnMat[testIter*k + neighIter];
			for(long long labN = labels.col_ptr[neighIdx-1]; labN < labels.col_ptr[neighIdx]; labN++)
			{
				if(lbVec[labels.row_idx[labN]] == 0)
				{	
					idxArr[lblCount] = labels.row_idx[labN];
					lblCount ++;
				}
				lbVec[labels.row_idx[labN]]++;
			}
		}
		
		for(long long Iiter = 0; Iiter < lblCount; Iiter++)
		{
			I[Icount] = testIter +1;
			J[Icount] = idxArr[Iiter] + 1;
			S[Icount] = lbVec[idxArr[Iiter]];
			Icount ++;
		}
	}
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	smat_t labels; //lxn
	double *nnMat; //kXnt
	double *I;
	double *J;
	double *S;
	long long avgLbl;
	
	long long n;
	long long l;
	long long nt;
	long long k;
	
	
	const mxArray *mxY=prhs[0];
	mxSparse_to_smat(mxY, labels);
	
	nnMat = mxGetPr(prhs[1]);
	
	l = mxGetM(prhs[0]);
	n = mxGetN(prhs[0]);
	k = mxGetM(prhs[1]);
	nt = mxGetN(prhs[1]);
	
	avgLbl = (long long)(*mxGetPr(prhs[2]));
	
	plhs[0] = mxCreateDoubleMatrix(nt*k*avgLbl, 1, mxREAL);
	I = mxGetPr(plhs[0]);
	
	plhs[1] = mxCreateDoubleMatrix(nt*k*avgLbl, 1, mxREAL);
	J = mxGetPr(plhs[1]);
	
	plhs[2] = mxCreateDoubleMatrix(nt*k*avgLbl, 1, mxREAL);
	S = mxGetPr(plhs[2]);
	
	printf("n: %lld nt: %lld l: %lld k: %lld avgLbl: %lld \n", n, nt, l, k, avgLbl);
	
	formScoreMat(labels, nnMat, I, J, S, n, l, nt, k);
}