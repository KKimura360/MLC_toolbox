#include <iostream>
#include <string>

#include <cstdio>
#include <cstring>
#include <cstdlib>

#include <mex.h>

#define NAME_LEN 100000

using namespace std;

void mexFunction(int nlhs, mxArray* plhs[], int rlhs, const mxArray* prhs[])
{
	mxAssert(nrhs==3,"Required and allowed arguments: mex_feature_mat, mex_label_mat, text_data_file_name");

	mxArray* ft_mat = (mxArray*) prhs[0];
	mxArray* lbl_mat = (mxArray*) prhs[1];
	char file_name[NAME_LEN];
	mxGetString(prhs[2],file_name,NAME_LEN);

	FILE* fout = fopen(file_name,"w");

	int num_inst = mxGetN(ft_mat);
	int num_ft = mxGetM(ft_mat);
	int num_lbl = mxGetM(lbl_mat);
	fprintf(fout,"%d %d %d\n",num_inst,num_ft,num_lbl);

	mwIndex* ft_Ir = mxGetIr(ft_mat);
	mwIndex* ft_Jc = mxGetJc(ft_mat);
	double* ft_Pr = mxGetPr(ft_mat);

	mwIndex* lbl_Ir = mxGetIr(lbl_mat);
	mwIndex* lbl_Jc = mxGetJc(lbl_mat);
	double* lbl_Pr = mxGetPr(lbl_mat);

	for(int i=0; i<num_inst; i++)
	{
		for(int j=lbl_Jc[i]; j<lbl_Jc[i+1]; j++)
		{
			int lbl = lbl_Ir[j];
			if(j==lbl_Jc[i])
				fprintf(fout,"%d",lbl);
			else
				fprintf(fout,",%d",lbl);
		}
		for(int j=ft_Jc[i]; j<ft_Jc[i+1]; j++)
		{
			fprintf(fout," %d:%f",ft_Ir[j],ft_Pr[j]);
		}
		fprintf(fout,"\n");
	}

	fclose(fout);

}
