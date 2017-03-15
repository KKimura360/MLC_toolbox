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

/* Sort in descending order */
int compVLPair(const void *a, const void *b){
	if(((vlPair*)a)->val > ((vlPair*)b)->val)
		return -1;
	else
		return 1;
}

void identifyCluster(smat_t datapoints, smat_t clustercenters, long k, long long n, long long l, double *clusterassign)
{
    double *score = (double *)malloc(sizeof(double)*n);
    double *cc = (double *)malloc(sizeof(double)*l);
    memset(cc, 0, sizeof(double)*l);
    double scoreval = 0;
    
    for(long long clusIter = 0; clusIter < k; clusIter++)
    {
        for (long long ccptr = clustercenters.col_ptr[clusIter]; ccptr < clustercenters.col_ptr[clusIter+1]; ccptr++)
            cc[clustercenters.row_idx[ccptr]] = clustercenters.val[ccptr];
        
        for (long long dataIter = 0; dataIter < n; dataIter ++)
        {
            scoreval = 0;
            for(long long dptr =datapoints.col_ptr[dataIter]; dptr < datapoints.col_ptr[dataIter+1]; dptr ++)
                scoreval += (cc[datapoints.row_idx[dptr]]*datapoints.val[dptr]);
            
            if(clusIter == 0)
            {
                clusterassign[dataIter] = clusIter;
                score[dataIter] = scoreval;
            }
            else if(scoreval > score[dataIter])
            {
                clusterassign[dataIter] = clusIter;
                score[dataIter] = scoreval;
            }
        }
        
        for (long long ccptr = clustercenters.col_ptr[clusIter]; ccptr < clustercenters.col_ptr[clusIter+1]; ccptr++)
            cc[clustercenters.row_idx[ccptr]] = 0;
    }
}
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{	
	smat_t datapoints;		/* Datapoints to be clustered, dimension lxn*/
	smat_t clustercenters;	/* Initial Cluster Centers, dimension lxk*/
	long k;					/* Number of clusters */
	
	double *clusterassign;	/* Clusters Assigned to datapoints */
	
	long long n;			/* Number of datapoints in stream */
	long long l;			/* Label dimensionality of the data */
	
	double *temp;
	
	#define DATA		prhs[0]
	#define CLUSTCENT	prhs[1]
	#define	K			prhs[2]
	
	#define CLUSTASSGN	plhs[0]
	/* Assigning input & output pointers */
	const mxArray *mxY=DATA;
	mxSparse_to_smat(mxY, datapoints);
	
    const mxArray *mxCC = CLUSTCENT;
	mxSparse_to_smat(mxCC, clustercenters);
	
	temp = mxGetPr(K);
	k = (long)*temp;
	
	n = datapoints.cols;
	l = datapoints.rows;
	
	CLUSTASSGN = mxCreateDoubleMatrix(n, 1, mxREAL);
	clusterassign = mxGetPr(CLUSTASSGN);
	
	identifyCluster(datapoints, clustercenters, k, n, l, clusterassign);
}