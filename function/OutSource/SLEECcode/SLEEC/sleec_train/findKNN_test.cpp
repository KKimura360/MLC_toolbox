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
	int loc;
	short orig;
};
typedef struct valLocPair vlPair;

void smat_x_dvec(const smat_t &labels, long long l, long long numData, double *score, long long dataPoint, long long *neighborIdx, double *neighborVal, long long numneighbors)
{
	long long minIdx = 0;
	double minVal = 100;
	for(long long dataIter = 0; dataIter < numData; dataIter ++)
	{	
		
		double scoreVal = 0;
		long long vecIdx=labels.col_ptr[dataPoint], matIdx=labels.col_ptr[dataIter];
		while((vecIdx < labels.col_ptr[dataPoint+1]) && (matIdx < labels.col_ptr[dataIter+1]))
		{	
			if(labels.row_idx[vecIdx] == labels.row_idx[matIdx])
			{	
				scoreVal += labels.val[vecIdx]*labels.val[matIdx];
				if(vecIdx < (labels.col_ptr[dataPoint+1]))
					vecIdx++;
				if(matIdx < (labels.col_ptr[dataIter+1]))
					matIdx++;
				
			}
			else if(labels.row_idx[vecIdx] < labels.row_idx[matIdx])
			{	
				if(vecIdx < (labels.col_ptr[dataPoint+1]-1))
					vecIdx++;
				else
					matIdx++;
			}
			else if(labels.row_idx[vecIdx] > labels.row_idx[matIdx])
			{	
				if(matIdx < (labels.col_ptr[dataIter+1]-1))
					matIdx++;
				else
					vecIdx++;
			}
		}
		score[dataIter] = scoreVal;
		if(dataIter < numneighbors)
		{
			neighborIdx[dataIter] = dataIter;
			neighborVal[dataIter] = scoreVal;
			if(scoreVal < minVal)
			{
				minVal = scoreVal;
				minIdx = dataIter;
			}
		}
		else if(scoreVal > minVal)
		{
			neighborIdx[minIdx] = dataIter;
			neighborVal[minIdx] = scoreVal;
			minVal = 1000;
			minIdx = -1;
			for(long long i = 0; i<numneighbors; i++)
			{
				if(neighborVal[i] < minVal)
				{
					minVal = neighborVal[i];
					minIdx = i;
				}
			}
		}
	}
}

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


/* Frobenius norm regularization for the obtained w matrix */
void normalizeW(double *w, long long d, long long l, double r)
{	
	long long iter;
	double norm = 0;
	for(iter = 0; iter < l*d; iter++)
		norm += (w[iter]*w[iter]);
	norm = sqrt(norm);
	if(norm > 0)
	{
		for(iter = 0; iter < l*d; iter++)
			w[iter] = r*(w[iter]/norm);
	}
}

void computeknn(smat_t labels, long long l, long long numData, long long numneighbors, double *om, double *omval, double *neighIdx, int numThreads)
{
	
	
	#pragma omp parallel for num_threads(numThreads)
	for (long long dataIter = 0; dataIter < numData; dataIter ++)
	{	
		double *score = (double *)malloc(sizeof(double)*numData);
		long long *neighborIdx = (long long *)malloc(sizeof(long long)*numneighbors);
		double *neighborVal = (double *)malloc(sizeof(double)*numneighbors);
		smat_x_dvec(labels, l, numData, score, dataIter, neighborIdx, neighborVal, numneighbors);
		
		//#pragma omp parallel for
		for(long long nIter = 0; nIter < numneighbors; nIter++)
		{
			om[dataIter*numneighbors + nIter] = numData*(neighborIdx[nIter]) + dataIter+1;
			omval[dataIter*numneighbors + nIter] = neighborVal[nIter] + 1e-10;
			neighIdx[dataIter*numneighbors + nIter] = (neighborIdx[nIter] + 1);
		}
		free(score);
		free(neighborIdx);
		free(neighborVal);
	}
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{	
	smat_t labels;			/* Input label vectors */
	long long numneighbors;	/* Number of nearest neighbours */
	
	long long numData;			/* Number of data points */
	long long l;					/* Label dimensionality */
	int numThreads;
    
	double *om;				/* Indices of nearest neighbours */
	double *omval;			/* Values of nearest neighbour dot products */
	double *neighIdx;		/* Matrix containing neighbor indices */
	double *temp;
	
	#define LABELS		prhs[0]
	#define NUMNEIGH	prhs[1]
	
	#define OM			plhs[0]
	#define OMVAL		plhs[1]
	#define NIDX		plhs[2]
	/* Assinging input & output pointers */
	const mxArray *mxY=LABELS;
	mxSparse_to_smat(mxY, labels);
	
	temp = mxGetPr(NUMNEIGH);
	numneighbors = (long long)(*temp);
	
	numData = mxGetN(prhs[0]);
	l = mxGetM(prhs[0]);
	
	OM = mxCreateDoubleMatrix(numneighbors, numData, mxREAL);
	om = mxGetPr(OM);
	
	OMVAL = mxCreateDoubleMatrix(numneighbors, numData, mxREAL);
	omval = mxGetPr(OMVAL);
	
	OMVAL = mxCreateDoubleMatrix(numneighbors, numData, mxREAL);
	omval = mxGetPr(OMVAL);
	
	NIDX = mxCreateDoubleMatrix(numneighbors, numData, mxREAL);
	neighIdx = mxGetPr(NIDX);
	
    numThreads = *(mxGetPr(prhs[2]));
	computeknn(labels, l, numData, numneighbors, om, omval, neighIdx, numThreads);
}