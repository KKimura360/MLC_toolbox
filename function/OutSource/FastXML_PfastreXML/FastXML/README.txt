FastXML: A Fast, Accurate and Stable Tree-classifier for eXtreme Multi-label Learning

Authors: 
========
Yashoteja Prabhu
Manik Varma

Download FastXML
================

Download FastXML source code in C++ and Matlab

Please make sure that you have read the license agreement in LICENSE.doc/pdf. Please do not install or use FastXML unless you agree to the terms of the license.

The code for FastXML is written in C++ and should compile on 32/64 bit Windows/Linux machines. Simple Matlab wrappers have also been provided with the code. Installation and usage instructions are provided below and in Readme.txt. This code is made available as is for non-commercial research purposes. Default parameters provided below work well on the benchmark datasets . The code requires C++11 enabled compilers.

Please contact Yashoteja Prabhu and Manik Varma if you have any questions or feedback.
Experimental Results and Datasets
Please visit the Extreme Classification Repository to download the benchmark datasets and compare FastXML's performance to baseline algorithms.

Usage
=====

Linux/Windows makefiles for compiling FastXML have been provided with the source code. To compile, run "make" (Linux) or "nmake -f Makefile.win" (Windows) in the topmost folder. Run the following commands from inside FastXML folder for training and testing.

Training
--------

C++:

	./fastXML_train [feature file name] [label file name] [model folder name] -S 0 -T 1 -s 0 -t 50 -b 1.0 -c 1.0 -m 10 -l 10 -g 30 -a 0.8 

Matlab:

	fastXML_train([feature matrix], [label matrix], param, [model folder name])

where:

	-T ≡ param.num_thread		: Number of threads to use										default=1
	-s ≡ param.start_tree		: Starting tree index											default=0
	-t ≡ param.num_tree		: Number of trees to be grown										default=50
	-b ≡ param.bias			: Feature bias value, extre feature value to be appended						default=1.0
	-c ≡ param.log_loss_coeff	: SVM weight co-efficient										default=1.0
	-l ≡ param.lbl_per_leaf		: Number of label-probability pairs to retain in a leaf							default=100
	-m ≡ param.max_leaf		: Maximum allowed instances in a leaf node. Larger nodes are attempted to be split, and on failure converted to leaves		default=10

Testing
-------

C++:

	./fastXML_test [feature file name] [score file name] [model folder name] T 1 -s 0 -t 50

Matlab:

	[score matrix] = fastXML_train([feature matrix], param, [model folder name])

where:

	-T ≡ param.num_thread		: same as in training											default=[value saved in trained model]
	-s ≡ param.start_tree		: same as in training											default=[value saved in trained model]
	-t ≡ param.num_tree		: same as in training											default=[value saved in trained model]

Performance Evaluation
----------------------

Scripts for performance evaluation are only available in Matlab. To compile these scripts, execute "make" in the topmost folder from the Matlab terminal.
Following command is executed from Tools/metrics folder and outputs a struct containing all the metrics:

	[metrics] = get_all_metrics([test score matrix], [test label matrix], [inverse label propensity vector])

Miscellaneous
-------------

* The data format required by FastXML/PfastreXML for feature and label input files is different from the format used in the repository datasets. To convert from the repository format to PfastreXML format, run the following command from 'Tools' folder:

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
To run FastXML on the EUR-Lex dataset, execute "bash sample_run.sh" (Linux) or "sample_run" (Windows) in the FastXML folder.
Read the comments provided in the above scripts for better understanding.

References
==========

1   Y. Prabhu, and M. Varma, FastXML: A Fast, Accurate and Stable Tree-classifier for eXtreme Multi-label Learning, in KDD 2014.

2   R. Agrawal, A. Gupta, Y. Prabhu, and M. Varma, Multi-label learning with millions of labels: Recommending advertiser bid phrases for web pages, in WWW 2013.

3   J. Weston, A. Makadia, and H. Yee, Label partitioning for sublinear ranking, in ICML 2013.

