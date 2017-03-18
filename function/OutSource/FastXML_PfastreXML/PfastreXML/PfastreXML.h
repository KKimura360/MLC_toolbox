#pragma once

#include "fastXML.h"

class PfParam 
{
public:
	Param param;
	_bool pfswitch;
	_float gamma;
	_float alpha;

	PfParam()
	{
		param = Param();
		pfswitch = false;
		gamma = 30;
		alpha = 0.8;
	}

	PfParam(string fname)
	{
		check_valid_filename(fname,true);
		ifstream fin;
		fin.open(fname);

		fin>>param.num_ft;
		fin>>param.num_lbl;
		fin>>param.log_loss_coeff;
		fin>>param.max_leaf;
		fin>>param.lbl_per_leaf;
		fin>>param.bias;
		fin>>param.num_thread;
		fin>>param.start_tree;
		fin>>param.num_tree;
	
		fin>>pfswitch;
		fin>>gamma;
		fin>>alpha;

		fin.close();
	}

	void write(string fname)
	{
		check_valid_filename(fname,false);
		ofstream fout;
		fout.open(fname);

		fout<<param.num_ft<<endl;
		fout<<param.num_lbl<<endl;
		fout<<param.log_loss_coeff<<endl;
		fout<<param.max_leaf<<endl;
		fout<<param.lbl_per_leaf<<endl;
		fout<<param.bias<<endl;
		fout<<param.num_thread<<endl;
		fout<<param.start_tree<<endl;
		fout<<param.num_tree<<endl;

		fout<<pfswitch<<endl;
		fout<<gamma<<endl;
		fout<<alpha<<endl;

		fout.close();
	}
};

