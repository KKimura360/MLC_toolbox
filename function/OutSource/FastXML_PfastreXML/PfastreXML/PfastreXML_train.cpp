#include <iostream>
#include <fstream>
#include <string>
#include <thread>

#include "timer.h"
#include "PfastreXML.h"

using namespace std;

void help()
{
	cerr<<"Sample Usage :"<<endl;
	cerr<<"./PfastreXML_train [feature file name] [label file name] [inverse propensity file name] [model folder name] -S 0 -T 1 -s 0 -t 50 -b 1.0 -c 1.0 -m 10 -l 10 -g 30 -a 0.8 -q 1"<<endl<<endl;

	cerr<<"-S PfastXML switch, setting this to 1 omits tail classifiers, thus leading to PfastXML algorithm. default=0"<<endl;
	cerr<<"-T Number of threads to use. default=1"<<endl;
	cerr<<"-s Starting tree index. default=0"<<endl;
	cerr<<"-t Number of trees to be grown. default=50"<<endl;
	cerr<<"-b Feature bias value, extre feature value to be appended. default=1.0"<<endl;
	cerr<<"-c SVM weight co-efficient. default=1.0"<<endl;
	cerr<<"-m Maximum allowed instances in a leaf node. Larger nodes are attempted to be split, and on failure converted to leaves. default=10"<<endl;
	cerr<<"-l Number of label-probability pairs to retain in a leaf. default=100"<<endl;
	cerr<<"-g gamma parameter appearing in tail label classifiers. default=30"<<endl;
	cerr<<"-a Trade-off parameter between PfastXML and tail label classifiers. default=0.8"<<endl;
	cerr<<"-q quiet option (0/1). default=0"<<endl;

	cerr<<"feature and label files are in sparse matrix format"<<endl;
	exit(1);
}

PfParam parse_param(_int argc, char* argv[])
{
	PfParam pfparam;

	string opt;
	string sval;
	_float val;

	for(_int i=0; i<argc; i+=2)
	{
		opt = string(argv[i]);
		sval = string(argv[i+1]);
		val = stof(sval);

		if(opt=="-m")
			pfparam.param.max_leaf = (_int)val;
		else if(opt=="-l")
			pfparam.param.lbl_per_leaf = (_int)val;
		else if(opt=="-b")
			pfparam.param.bias = (_float)val;
		else if(opt=="-c")
			pfparam.param.log_loss_coeff = (_float)val;
		else if(opt=="-T")
			pfparam.param.num_thread = (_int)val;
		else if(opt=="-s")
			pfparam.param.start_tree = (_int)val;
		else if(opt=="-t")
			pfparam.param.num_tree = (_int)val;
		else if(opt=="-S")
			pfparam.pfswitch = (_bool)val;
		else if(opt=="-g")
			pfparam.gamma = (_float)val;
		else if(opt=="-a")
			pfparam.alpha = (_float)val;
		else if(opt=="-q")
			pfparam.param.quiet = (_bool)val;
	}

	return pfparam;
}

int main(int argc, char* argv[])
{
	if(argc < 5)
		help();


	string ft_file = string(argv[1]);
	check_valid_filename(ft_file, true);
	SMatF* trn_ft_mat = new SMatF(ft_file);

	string lbl_file = string(argv[2]);
	check_valid_filename(lbl_file, true);
	SMatF* trn_lbl_mat = new SMatF(lbl_file);

	string prop_file = string(argv[3]);
	check_valid_filename(prop_file, true);
	ifstream fin;
	fin.open(prop_file);
	VecF inv_props;
	for(_int i=0; i<trn_lbl_mat->nr; i++)
	{
		_float f;
		fin>>f;
		inv_props.push_back(f);
	}
	fin.close();

	string model_folder = string(argv[4]);
	check_valid_foldername(model_folder);

	PfParam pfparam = parse_param(argc-5,argv+5);
	pfparam.param.num_ft = trn_ft_mat->nr;
	pfparam.param.num_lbl = trn_lbl_mat->nr;
	pfparam.write(model_folder+"/param");
	Param param = pfparam.param;

	if( param.quiet )
		loglvl = LOGLVL::QUIET;

	Timer timer;
	timer.start();

	USE_IDCG = false;

	/* Weighting label matrix with inverse propensity weights */
	for(_int i=0; i<trn_lbl_mat->nc; i++)
		for(_int j=0; j<trn_lbl_mat->size[i]; j++)
			trn_lbl_mat->data[i][j].second *= inv_props[trn_lbl_mat->data[i][j].first];

	/* training PfastXML trees */
	train_trees(trn_ft_mat, trn_lbl_mat, param, model_folder);

	/* if pfswitch is true, terminate here immediately after PfastXML */
	if(pfparam.pfswitch)
	{
		cout << "training time: " << timer.stop() << " s" << endl;

		delete trn_ft_mat;
		delete trn_lbl_mat;
		return 0;
	}

	/* normalize feature vectors to unit norm */
	trn_ft_mat->unit_normalize_columns();

	/*--- calculating model parameters saved in w ---*/

	SMatF* tmat = trn_lbl_mat->transpose();

	for(int i=0; i<tmat->nc; i++)
	{
		_float a = 1.0/(tmat->size[i]);
		for(int j=0; j<tmat->size[i]; j++)
			tmat->data[i][j].second = a;
	}

	SMatF* w = trn_ft_mat->prod(tmat);

	cout << "training time: " << timer.stop() << " s" << endl;


	w->write(model_folder+"/w");

	/* free allocated resources */
	delete tmat;
	delete w;
	delete trn_ft_mat;
	delete trn_lbl_mat;
}
