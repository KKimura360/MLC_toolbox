function[param]=SetfRAkELParameter(pseudo)
%SetfRAkELParameter
%number of sampled labels
param.numK=3;
param.numM='2*numL';
% not RAkEL-d
param.type='normal';

%classifier for the 1st layer
param.MLC.name={'BR'};
%param.MLC.base.svmparam='-s 2 -q';
param.MLC.base.name='ridge';
param.MLC.base.param.lambda=20;
param.MLC.th.type='Scut';
param.MLC.th.param=0.3;


