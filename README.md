## MLC_toolbox
A MATLAB/OCTAVE library for Multi-Label Classification

## Current available functions: 
Clustering-based method, CBMLC, HOMER, CLMLC  
Ensemble-based method,   ECC, RAkEL, RAkEL-d fRAkEL, TREMLC, MLCEnsemble,COCOA  
Feature Space Dimension Reduction (FSDR)  
FSDR-unsupervised method (confirmed), PCA, NMF, RFS     
FSDR-supervised method (confirmed), MLSI, MFFM, OPLS, MLHSL, FScore, MLJMI, MLMIM, MLMRMR  
FSDR-supervised method (uncofirmed) READER  
Label Space Dimension Reduction (LSDR), CSSP,PLST,CPLST,FaIE  
Process methods, CC, Meta-Label CC, PS, triClass    
MLC-base classifiers (confirmed), BR, LP  
MLC-base classifiers (unconfirmed), BPMLL, CLR, rankSVM  

Base Classifiers
LIBLINEAR, LIBSVM, rigde regression, k-NN


## How to run Sample Code (Sample.m)
# Dataset     
`dataname='{datasetname}'`  dataset can be found dataset/matfile/  
`numCV = 3 or 5 or 10`      3-CV or 5-CV or 10-CV we already splited training/test instance, indices can be found dataset/index/n-fold/  

---

# Method  
`method.name{'{meethodname1}','{methodname2}',...}`  
In this library we can combine any problem transformation methods.  
For example, when we want to conduct PCA for the feature selection first, and then conduct k-means to divide instances, at last, random Classifier Chain use for each cluster,  
  `method.name={'PCA',CBMLC','rCC'}`  
methods are conducted on this order. So if you want to conduct k-means first and then conduct PCA for each cluster,   
`method.name{'CBMLC','PCA','rCC'}`  
NOTE: CBMLC is Clustering Based Multi-Label Classification method.  

---

# Parameters for each method  
Many methods may have several parameters with different name, so we gave up to implement CLI. We use file for setting parameters.(you can also add the code on Sample.m)   
Set{methodname}Parameter.m is a file to set parameter. see function/{category}/{methodname}/Set{methodname}Parameter.m  
For example  
```SetPCAParameter.m  
function[param]=SetPCAParameter(~)  
%setPCAParameter  
%Dimensionality of the feature subspace  
param.dim=300;  
```  
On some parameters, you can define values depends on dataset information by setting with string from like,  
 
```SetPCAParameter.m  
function[param]=SetPCAParameter(~)  
%setPCAParameter  
%Dimensionality of the feature subspace  
param.dim= 'numF*0.5';  
```  
The method command the string line and substitute the value of the result  
In this library,   
`numF` is the number of features  
`numN` is the number of instances  
`numL` is the number of labels 

---
# Base Line Classifier 
Most methods based on traditional binary/multi-class classifier to solve MLC 
`method.base.name='{base classifier name}`   
Now we support `liinear_svm` from LIBLINEAR,`svm` from LIBSVM, `ridge` (ridge regression)  and 'knn' (k-nearest neighbor)  

---

# Threshold   
Some methods returns not discrete 0-1 classification results but scores for labels  
To obtain <b>classification result</b>, threshold is needed.   
`method.th.type='Scut' or 'Rcut' or 'Pcut' `   
`method.th.param=parameter` 
Now, we support, Scut, Rcut, Pcut. 

---

# Results
`res.{criteria}` contains result   



## Contributors
Keigo Kimura(KKimura360) and Lu Sun(futuresun912).  


## Contacts   
keikim360{at}gmail.com