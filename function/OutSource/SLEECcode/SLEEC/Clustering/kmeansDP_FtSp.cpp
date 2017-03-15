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

void doKMeans(smat_t datapoints, double *clustercenters, long k, double numiters, long long n, long long l, double *clusterassign, int numThreads)
{
	
	for(long iter = 0; iter < numiters; iter++)
	{	
		long long c1=0, c2=0, c3=0;
		//For each datapoint, assign the relevant cluster center, if tie select randomly
        #pragma omp parallel for num_threads(numThreads)
		for(long long dataIter = 0; dataIter < n; dataIter ++)
		{   
            double curmin;
            double *censel = (double *)malloc(sizeof(double)*k);
            long numcen;
            
			numcen = 0;
			curmin = -9999;
            memset(censel, 0, sizeof(double)*k);
			for(long long clusIter = 0; clusIter < k; clusIter++)
			{
				double curIntersect = 0;
				for(long labIter = datapoints.col_ptr[dataIter]; labIter < datapoints.col_ptr[dataIter + 1]; labIter ++)
				{
					curIntersect += (clustercenters[clusIter*l + datapoints.row_idx[labIter]]*datapoints.val[labIter]);
				}
				if((curIntersect - curmin) < 1e-12 && (curmin - curIntersect) < 1e-12 )
				{
					censel[numcen] = clusIter;
					numcen ++;
				}
				else if(curIntersect > curmin + 1e-12)
				{
					censel[0] = clusIter;
					numcen = 1;
					curmin = curIntersect;
				}
			}
			if(numcen == 1)
			{
				clusterassign[dataIter] = censel[0];
				c1++;
			}
			else if(numcen > 1)
			{
				clusterassign[dataIter] = censel[rand()%numcen];
				c2++;
			}
			else if(numcen == 0)
			{
				clusterassign[dataIter] = rand()%k;
				c3++;
			}
            free(censel);
		}
		long long *cluscount = (long long *)malloc(sizeof(long long)*k);

        memset(cluscount, 0, sizeof(long long)*k);
        memset(clustercenters, 0, sizeof(double)*l*k);
		//Update cluster centers
		for (long long dataIter = 0; dataIter < n; dataIter ++)
		{	
			long curcluster = clusterassign[dataIter];
			cluscount[curcluster] ++;
			for(long labIter = datapoints.col_ptr[dataIter]; labIter < datapoints.col_ptr[dataIter + 1]; labIter ++)
			{
				clustercenters[curcluster*l + datapoints.row_idx[labIter]] += datapoints.val[labIter];
			}
		}
		//Normalize cluster centers
        #pragma omp parallel for
		for(long clusIter = 0; clusIter < k; clusIter ++)
		{
			for(long long labIter = 0; labIter < l; labIter++)
			{
				clustercenters[clusIter*l + labIter] = clustercenters[clusIter*l + labIter]/cluscount[clusIter];
			}
		}
	}
	
}
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{	
	smat_t datapoints;		/* Datapoints to be clustered, dimension lxn*/
	double *clustercenters;	/* Initial Cluster Centers, dimension lxk*/
	long k;					/* Number of clusters */
	double numiters;		/* Number of iterations */
    int numThreads;
	
	double *clusterassign;	/* Clusters Assigned to datapoints */
	
	long long n;			/* Number of datapoints in stream */
	long long l;			/* Label dimensionality of the data */
	
	double *temp;
	
	#define DATA		prhs[0]
	#define CLUSTCENT	prhs[1]
	#define	K			prhs[2]
	#define NUMITERS	prhs[3]
	#define NUMTHREADS  prhs[4]
    
	#define CLUSTASSGN	plhs[0]
	/* Assigning input & output pointers */
	const mxArray *mxY=DATA;
	mxSparse_to_smat(mxY, datapoints);
	
	clustercenters = mxGetPr(CLUSTCENT);
	
	temp = mxGetPr(K);
	k = (long)*temp;
	
	temp = mxGetPr(NUMITERS);
	numiters = *temp;
	
    temp = mxGetPr(NUMTHREADS);
    numThreads = *temp;
    
	n = datapoints.cols;
	l = datapoints.rows;
	
	CLUSTASSGN = mxCreateDoubleMatrix(n, 1, mxREAL);
	clusterassign = mxGetPr(CLUSTASSGN);
	
	doKMeans(datapoints, clustercenters, k, numiters, n, l, clusterassign, numThreads);
}