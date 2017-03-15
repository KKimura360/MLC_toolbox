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

#define printf //printf
#define fflush //fflush
#define eps 0.1
#define min(a,b) a>b?b:a

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

int compVLPair(const void *a, const void *b){
	if(((vlPair*)a)->val > ((vlPair*)b)->val)
		return -1;
	else
		return 1;
}

void eval_prec_k_mult(smat_t Y, smat_t Yt, double *KNN, double *result, long long n, long long l, long long nt, long long numneighbors, double *pred_val, int numThreads)
{
    
   #pragma omp parallel for num_threads(numThreads)
	for(long long dataIter = 0; dataIter < nt; dataIter ++)
	{   
        double *prec = (double *)malloc(sizeof(double)*5);
        memset(prec, 0, 5*sizeof(double));
		unsigned int numLabelsActive = 0;
		for(long neighborIter = 0; neighborIter < numneighbors; neighborIter ++)
		{
			long long neighIdx = KNN[dataIter*numneighbors + neighborIter];
			numLabelsActive += (Y.col_ptr[neighIdx] - Y.col_ptr[neighIdx-1]);
		}
        
		 vlPair *vl = (vlPair *)malloc(sizeof(vlPair)*numLabelsActive);
		memset(vl, 0, sizeof(vlPair)*numLabelsActive);
		long count = 0;
		
		for (int neighborIter = 0; neighborIter < numneighbors; neighborIter++)
		{   
			long neighIdx = KNN[dataIter*numneighbors + neighborIter];
			for(long idx = Y.col_ptr[neighIdx-1]; idx < Y.col_ptr[neighIdx]; idx++)
			{
				int found = 0;
				for(int i = 0; i < count; i++)
				{
					if(vl[i].loc == Y.row_idx[idx])
					{
						vl[i].val += Y.val[idx];
						found = 1;
						break;
					}
				}
				
				if(found == 0)
				{
					vl[count].loc = Y.row_idx[idx];
					vl[count].val = Y.val[idx];
					count ++;
				}
			}
		}
		qsort(vl, count, sizeof(vlPair), compVLPair);
		
		for (int i = 0; i < 5; i++)
		{
			pred_val[i + dataIter*5] = vl[i].loc;
		}
		
		for(int pIter = 0; pIter < 5; pIter ++)
		{
			if(count < pIter)
				continue;
			for(long idx = Yt.col_ptr[dataIter]; idx < Yt.col_ptr[dataIter+1]; idx++)
			{	
				if(Yt.row_idx[idx] == vl[pIter].loc)
				{
					for(int i = pIter; i <5; i++)
					{
						prec[i] += 1;
					}
				}
			}
		}
        
        for (int i = 0; i < 5; i++)
		{
			result[i + dataIter*5] = prec[i]/(i+1);
		}
        
		free(vl);
        free(prec);
	}
    
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{	
	smat_t Y;
	smat_t Yt;
	double *KNN;
	long long numneighbors;
	int numThreads;
    
	long long n;			
	long long l;
	long long nt;
	
	double *temp;
	
	double *result;
	double *pred_val;
	
	#define YINP		prhs[0]
	#define YTINP 		prhs[1]
	#define KNNI		prhs[2]
	#define NUMNEIGH	prhs[3]
	
	#define RESULT		plhs[0]
	#define PRED		plhs[1]
	
	const mxArray *mxY=YINP;
	mxSparse_to_smat(mxY, Y);
	
	const mxArray *mxYt=YTINP;
	mxSparse_to_smat(mxYt, Yt);
	
	temp = mxGetPr(NUMNEIGH);
	numneighbors = (long long)(*temp);
	
	KNN = mxGetPr(KNNI);
	numThreads = *(mxGetPr(prhs[4]));
    
	n = mxGetN(prhs[0]);
	l = mxGetM(prhs[0]);
	nt = mxGetN(prhs[1]);
	
	RESULT = mxCreateDoubleMatrix(5, nt, mxREAL);
	result = mxGetPr(RESULT);
	PRED = mxCreateDoubleMatrix(5, nt, mxREAL);
	pred_val = mxGetPr(PRED);
	
	eval_prec_k_mult(Y, Yt, KNN, result, n, l, nt, numneighbors, pred_val, numThreads);
}
