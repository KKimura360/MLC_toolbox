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
#include <fstream>
#include <omp.h>

using namespace std;
#define eps 0.1
#define min(a,b) a>b?b:a

#define printf //printf
#define fflush //fflush
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
	long long loc;
	short orig;
};
typedef struct valLocPair vlPair;

/* Sort in descending order */
int compVLPair(const void *a, const void *b){
	if(((vlPair*)a)->val > ((vlPair*)b)->val)
		return -1;
	else
		return 1;
}

/*  Input:	Matrix A in column major form, dimensions dxl
			Vector x, dimension dx1
	Output:	Vector b = ATx, dimension lx1
 */
void dmat_x_dvec(double *A, double *x, long long d, long long l, double *b)
{
	long long rCount, cCount;
	for(rCount = 0; rCount < l; rCount++)
	{	
		double score = 0;
		for(cCount = 0; cCount < d; cCount++)
		{
			score += A[rCount*d + cCount]*x[cCount];
		}
		b[rCount] = score;
	}
}

void computeknn(double *Z, double *Zt, double *neighIdx, double *neighVal, long long numneighbors, long long n, long long k, long long nt)
{
	for (long long testIter = 0; testIter < nt; testIter ++)
	{	
        vlPair *vl = (vlPair *)malloc(sizeof(vlPair)*numneighbors);
        double curminval = 99999;
        long curminloc = -1;
        double score;
		for (long long trainIter = 0; trainIter < n; trainIter++)
		{
			score = 0;
			for(int dimIter = 0; dimIter < k; dimIter++)
			{
				score += Zt[testIter*k + dimIter]*Z[trainIter*k+dimIter];
			}
			
			if(trainIter<numneighbors)
			{
				vl[trainIter].loc = trainIter;
				vl[trainIter].val = score;
				if(score < curminval)
				{
					curminval = score;
					curminloc = trainIter;
					
				}
			}
			else if(score >= curminval)
			{	
				vl[curminloc].loc = trainIter;
				vl[curminloc].val = score;
				//Update the curminloc and curminval
				curminval = 99999;
				curminloc = -1;
				for(long i = 0; i < numneighbors; i++)
				{	
					
					if(vl[i].val < curminval)
					{
						curminval = vl[i].val;
						curminloc = i;
					}
				}
			}
		}
		qsort(vl, numneighbors, sizeof(vlPair), compVLPair);
		for(long neighIter =0; neighIter < numneighbors; neighIter++)
		{
			neighIdx[testIter*numneighbors + neighIter] = vl[neighIter].loc + 1;
			neighVal[testIter*numneighbors + neighIter] = vl[neighIter].val;
		}
        free(vl);
	}
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{	
	double *Z;			
	double *Zt;
	long long numneighbors;	
	
	long long n;			
	long long k;			\
	long long nt;
	
	double *neighIdx;		
	double *neighVal;
	
	double *temp;
	
	#define ZINP		prhs[0]
	#define ZTINP		prhs[1]
	#define NUMNEIGH	prhs[2]
	
	#define NEIGHIDX	plhs[0]
	#define NEIGHVAL	plhs[1]
	
	Z = mxGetPr(ZINP);
	Zt = mxGetPr(ZTINP);
	
	temp = mxGetPr(NUMNEIGH);
	numneighbors = (long long)(*temp);
	
	n = mxGetN(prhs[0]);
	k = mxGetM(prhs[0]);
	nt = mxGetN(prhs[1]);
	
	NEIGHIDX = mxCreateDoubleMatrix(numneighbors, nt, mxREAL);
	neighIdx = mxGetPr(NEIGHIDX);
	
	NEIGHVAL = mxCreateDoubleMatrix(numneighbors, nt, mxREAL);
	neighVal = mxGetPr(NEIGHVAL);
	
	printf("%ld %ld %ld\n", n, k, nt);
	computeknn(Z, Zt, neighIdx, neighVal, numneighbors, n, k, nt);
}