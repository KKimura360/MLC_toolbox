#include <iostream>
#include <fstream>
#include <string>

#include "timer.h"
#include "fastXML.h"

using namespace std;

void help()
{
	cerr<<"Sample Usage :"<<endl;
	cerr<<"./fastXML_test [feature file name] [score file name] [model folder name] T 1 -s 0 -t 50 -q 1"<<endl<<endl;

	cerr<<"-T Number of threads to use. default=[value saved in trained model]"<<endl;
	cerr<<"-s Starting tree index. default=[value saved in trained model]"<<endl;
	cerr<<"-t Number of trees to be grown. default=[value saved in trained model]"<<endl;
	cerr<<"-q quiet option (0/1). default=[value saved in trained model]"<<endl;

	cerr<<"feature and score files are in sparse matrix format"<<endl;
	exit(1);
}

Param parse_param(int argc, char* argv[], string model_folder)
{
	Param param(model_folder+"/param");

	string opt;
	string sval;
	_float val;

	for(_int i=0; i<argc; i+=2)
	{
		opt = string(argv[i]);
		sval = string(argv[i+1]);
		val = stof(sval);

		if(opt=="-T")
			param.num_thread = (_int)val;
		else if(opt=="-s")
			param.start_tree = (_int)val;
		else if(opt=="-t")
			param.num_tree = (_int)val;
		else if(opt=="-q")
			param.quiet = (_bool)val;
	}

	return param;
}

int main(int argc, char* argv[])
{
	if(argc < 4)
		help();

	string ft_file = string(argv[1]);
	check_valid_filename(ft_file,true);
	SMatF* tst_ft_mat = new SMatF(ft_file);

	string score_file = string(argv[2]);
	check_valid_filename(score_file,false);

	string model_folder = string(argv[3]);
	check_valid_foldername(model_folder);

	Param param = parse_param(argc-4, argv+4, model_folder);

	if( param.quiet )
		loglvl = LOGLVL::QUIET;

	_float ram = 0;
	Timer timer;
	timer.start();

	SMatF* score_mat = test_trees(tst_ft_mat, param, model_folder, ram);

	cout << "test time: " << timer.stop() << " s" << endl;
	cout << "ram: " << ram/1000000 << " MB" << endl;

	score_mat->write(score_file);

	delete tst_ft_mat;
	delete score_mat;
}
