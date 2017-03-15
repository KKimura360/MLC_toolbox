

#include "mex.h"
#include "../smat.h"
#include <cstring>

#ifdef MX_API_VER
#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif
#endif

#define CMD_LEN 2048
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL


void exit_with_help()
{
	mexPrintf(
	"Usage: smat_write(M, 'filename')\n"
	);
}


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

// convert matlab sparse matrix to C smat fmt

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

/*
blocks_t mxSparse_to_blocks(const mxArray *M, int num_blocks, blocks_t &R) {
	R = blocks_t(num_blocks);
	unsigned long rows, cols, nnz;
	mwIndex *ir, *jc;
	double *v;
	ir = mxGetIr(M); jc = mxGetJc(M); v = mxGetPr(M);
	rows = mxGetM(M); cols = mxGetN(M); nnz = jc[cols];
	R.from_matlab(rows, cols, nnz);
	for(unsigned long c = 0; c < cols; c++) {
		for(unsigned long idx = jc[c]; idx < jc[c+1]; ++idx){
			R.insert_rate(idx, rate_t(ir[idx], c, v[idx]));
			++R.nnz_row[ir[idx]];
			++R.nnz_col[c];
		}
	}
	R.compressed_space(); // Need to call sort later.
	sort(R.allrates.begin(), R.allrates.end(), RateComp(&R));
	return R;
}

// convert matab dense matrix to column fmt
int mxDense_to_matCol(const mxArray *mxM, mat_t &M) {
	unsigned long rows = mxGetM(mxM), cols = mxGetN(mxM);
	double *val = mxGetPr(mxM);
	M = mat_t(cols, vec_t(rows,0));
	for(unsigned long c = 0, idx = 0; c < cols; ++c) 
		for(unsigned long r = 0; r < rows; ++r)
			M[c][r] = val[idx++];
	return 0;
}

int matCol_to_mxDense(const mat_t &M, mxArray *mxM) {
	unsigned long cols = M.size(), rows = M[0].size();
	double *val = mxGetPr(mxM);
	if(cols != mxGetN(mxM) || rows != mxGetM(mxM)) {
		mexPrintf("matCol_to_mxDense fails (dimensions do not match)\n");
		return -1;
	}

	for(unsigned long c = 0, idx = 0; c < cols; ++c)
		for(unsigned long r = 0; r < rows; r++) 
			val[idx++] = M[c][r];
	return 0;
}

// convert matab dense matrix to row fmt
int mxDense_to_matRow(const mxArray *mxM, mat_t &M) {
	unsigned long rows = mxGetM(mxM), cols = mxGetN(mxM);
	double *val = mxGetPr(mxM);
	M = mat_t(rows, vec_t(cols,0));
	for(unsigned long c = 0, idx = 0; c < cols; ++c) 
		for(unsigned long r = 0; r < rows; ++r)
			M[r][c] = val[idx++];
	return 0;
}

int matRow_to_mxDense(const mat_t &M, mxArray *mxM) {
	unsigned long rows = M.size(), cols = M[0].size();
	double *val = mxGetPr(mxM);
	if(cols != mxGetN(mxM) || rows != mxGetM(mxM)) {
		mexPrintf("matRow_to_mxDense fails (dimensions do not match)\n");
		return -1;
	}

	for(unsigned long c = 0, idx = 0; c < cols; ++c)
		for(unsigned long r = 0; r < rows; r++) 
			val[idx++] = M[r][c];
	return 0;
}
*/

bool isDoubleSparse(const mxArray *mxM) {
		if(!mxIsDouble(mxM)) {
			mexPrintf("Error: matrix must be double\n");
			return false;
		}

		if(!mxIsSparse(mxM)) {
			mexPrintf("matrix must be sparse; "
					"use sparse(matrix) first\n");
			return false;
		}
		return true;
}
bool isDoubleDense(const mxArray *mxM) {
		if(!mxIsDouble(mxM)) {
			mexPrintf("Error: matrix must be double\n");
			return false;
		}

		if(mxIsSparse(mxM)) {
			mexPrintf("matrix must be dense; "
					"use full(matrix) first\n");
			return false;
		}
		return true;
}
// Interface function of matlab
// now assume prhs[0]: A, prhs[1]: W, prhs[0]
void mexFunction( int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[] )
{
	if(nrhs == 2)
	{
		char filename[256];
		if(isDoubleSparse(prhs[0])==false)
		{
			mexPrintf("Error: This must be both sparse and double\n");			
			return;
		}

		mxGetString(prhs[1], filename, mxGetN(prhs[1])+1);		
		smat_t M;
		mxSparse_to_smat(prhs[0], M);
		printf("nnz %ld rows %ld cols %ld\n", M.nnz, M.rows, M.cols);
		M.save_binary_to_file(filename);
	}
	else
	{
		exit_with_help();		
		return;
	}
}


