mulan
==========
Reproduce [mulan](http://mulan.sourceforge.net/datasets.html) dataset results from [this paper](http://arxiv.org/abs/1502.02710).  

If you lack a poissrnd function, you can try 
> make poissrnd.m

Otherwise, at a matlab prompt:
> &gt;&gt; traintestall  
> ... (lots of hyperparameter tuning) ...   

will eventually report Hamming loss, micro-F1, and macro-F1 for bibtex, corel5k, mediamill, and yeast.  It takes a while so you should go get a sandwich.

If you wish to reproduce and/or modify the conversion of raw data into matlab format, you may find the other [Makefile](Makefile) targets useful.  However the mat files have been checked in so this is not strictly necessary to compute the results. 
> make rebuildmat   
> ... (lots of downloading, munging, and potential for error) ...

You really need matlab to rebuild the data files.  Octave and txt2mat do not get along.

Notice
----------
txt2mat is from [matlab central](http://www.mathworks.com/matlabcentral/fileexchange/18430-txt2mat) and is covered by the (BSD) license file [txt2mat.license](txt2mat.license).
