PfastreXML: Propensity weighted Reranked FastXML

Authors: 
========
Himanshu Jain
Yashoteja Prabhu
Manik Varma

Usage
=====

Linux/Windows makefiles for compiling PfastreXML have been provided with the source code. To compile, run "make" (Linux) or "nmake -f Makefile.win" (Windows) in the topmost folder. Run the following commands from inside PfastreXML folder for training and testing.

Training
--------

C++:

	./PfastreXML_train [feature file name] [label file name] [inverse propensity file name] [model folder name] -S 0 -T 1 -s 0 -t 50 -b 1.0 -c 1.0 -m 10 -l 10 -g 30 -a 0.8 

Matlab:

	PfastreXML_train([feature matrix], [label matrix], [inverse propensity vector], param, [model folder name])

where:

	-S ≡ param.pfswitch		: PfastXML switch, setting this to 1 omits tail classifiers, leading to PfastXML algorithm		default=0
	-T ≡ param.num_thread		: Number of threads to use										default=1
	-s ≡ param.start_tree		: Starting tree index											default=0
	-t ≡ param.num_tree		: Number of trees to be grown										default=50
	-b ≡ param.bias			: Feature bias value, extre feature value to be appended						default=1.0
	-c ≡ param.log_loss_coeff	: SVM weight co-efficient										default=1.0
	-l ≡ param.lbl_per_leaf		: Number of label-probability pairs to retain in a leaf							default=100
	-g ≡ param.gamma		: gamma parameter appearing in tail label classifiers							default=30
	-a ≡ param.alpha		: Trade off parameter between PfastXML and tail classifier scores					default=0.8
	-m ≡ param.max_leaf		: Maximum allowed instances in a leaf node. Larger nodes are attempted to be split, and on failure converted to leaves		default=10

Testing
-------

C++:

	./PfastreXML_test [feature file name] [score file name] [model folder name] -S 0 -T 1 -s 0 -t 50 -n 1000 

Matlab:

	[score matrix] = PfastreXML_train([feature matrix], param, [model folder name])

where:

	-S ≡ param.pfswitch		: same as in training											default=[value saved in trained model]
	-T ≡ param.num_thread		: same as in training											default=[value saved in trained model]
	-s ≡ param.start_tree		: same as in training											default=[value saved in trained model]
	-t ≡ param.num_tree		: same as in training											default=[value saved in trained model]
	-n ≡ param.actlbl		: Number of predicted scores per test instance. Lower value means quicker prediction			default=0.8

Performance Evaluation
----------------------

Scripts for performance evaluation are only available in Matlab. To compile these scripts, execute "make" in the topmost folder from the Matlab terminal.
Following command is executed from Tools/metrics folder and outputs a struct containing all the metrics:

	[metrics] = get_all_metrics([test score matrix], [test label matrix], [inverse label propenbsity vector])


Miscellaneous
-------------

* The data format required by PfastreXML for feature and label input files is different from the format used in the repository datasets. To convert from the repository format to PfastreXML format, run the following command from 'Tools' folder:

    	perl convert_format.pl [repository data file] [output feature file name] [output label file name] 

* Scripts are provided in the 'Tools' folder for sparse matrix inter conversion between Matlab .mat format and text format.
    To read a text matrix into Matlab:

    	[matrix] = read_text_mat([text matrix name]); 

    To write a Matlab matrix into text format:

    	write_text_mat([Matlab sparse matrix], [text matrix name to be written to]);

* To generate inverse label propensity weights, run the following command inside 'Tools/metrics' folder on Matlab terminal:

    	[weights vector] = inv_propensity([training label matrix],A,B); 

    A,B are the parameters of the inverse propensity model. Following values are to be used over the benchmark datasets:

    	Wikipedia-LSHTC: A=0.5,  B=0.4
    	Amazon:          A=0.6,  B=2.6
    	Other:		 A=0.55, B=1.5


Toy Example
===========

The zip file containing the source code also includes the EUR-Lex dataset as a toy example.
To run PfastreXML on the EUR-Lex dataset, execute "bash sample_run.sh" (Linux) or "sample_run" (Windows) in the PfastreXML folder.
Read the comments provided in the above scripts for better understanding.

References
==========

1   H. Jain, Y. Prabhu, and M. Varma, Extreme Multi-label Loss Functions for Recommendation, Tagging, Ranking & Other Missing Label Applications, in KDD 2016.

2   Y. Prabhu, and M. Varma, FastXML: A Fast, Accurate and Stable Tree-classifier for eXtreme Multi-label Learning, in KDD 2014.

2   K. Bhatia, H. Jain, P. Kar, M. Varma and P. Jain, Sparse Local Embeddings for Extreme Multi-label Classification, in NIPS 2015.

