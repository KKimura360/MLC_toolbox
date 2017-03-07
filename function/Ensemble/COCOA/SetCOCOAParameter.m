function[param]=SetCOCOAParameter(pseudo)

param.BR.param='';
param.BR.base.name='ridge';
param.BR.base.param.lambda=10;
param.BR.th=''; % if you need 

param.Tri.name={'LP'};
param.Tri.param.numK=3;
param.Tri.base.name='linear_svm';
param.Tri.base.param.svmparam='-s 2 -q -c 1';
param.Tri.th=''; % if you need 




