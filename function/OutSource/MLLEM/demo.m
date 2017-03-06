
%% demo   
%% how to use matlab < demo.m or demo  on matlab console
addpath('Eval');
addpath('Function')
params.data='CAL500';
%
load(['dataset/',params.data,'.mat']);

%% paramters for training
params.k1=30;
params.k2=30;
params.opsWI=1;
params.opsWL=1;
params.dim=20;
params.alpha=5;
params.beta=5;

%% parameters for testing 
params.lambda=1;
params.k3=100;

[nN nF]=size(Xtr);
[nN nL]=size(Ytr);

% avoiding error
params.k1=min(params.k1,nN);
params.k2=min(params.k2,nL);
params.k3=min(params.k3,nN);


%% Training
[G H V] = train_MLLEM(Xtr,Ytr,params);


%% Testing with linear embedding
params.method='L';
[Results]= test_MLLEM(Xtr,Ytr,Xts,Yts,G,H,params);

fprintf('Linear embedding :  AUC %1.3f Top-1 %1.3f Top-3 %1.3f Top-5 %1.3f\n', Results.auc, Results.top1,Results.top3,Results.top5);

%% Testing with nonlinear embedding
params.method='NL';
[Results]= test_MLLEM(Xtr,Ytr,Xts,Yts,G,H,params);
   
fprintf('Nonlinear embedding :  AUC %1.3f Top-1 %1.3f Top-3 %1.3f Top-5 %1.3f\n', Results.auc, Results.top1,Results.top3,Results.top5);





