function[model,time]=BPMLL_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
% see the detail/OutSource/
%method.param{x}.hiden_neuron;
%method.param{x}.alpha;
%recommend to use a defult setting
%method.param{x}.epochs=100;
%method.param{x}.intype=2;
%method.param{x}.outtype=2;
%method.param{x}.Cost=0.1;
%method.param{x}.min_max=[zeros(N,1),zeros(N,1)];
%% Output
%model{1}: nets (cell(epoch,1))
%model{2}: erros(cell(epoch,1))
%% Reference
%Zhang, M. L., & Zhou, Z. H. (2006). Multilabel neural networks with applications to functional genomics and text categorization. IEEE transactions on Knowledge and Data Engineering, 18(10), 1338-1351.
%http://cse.seu.edu.cn/people/zhangml/Resources.htm

%% ----- NOTE ----- %%
% Since I don't have Neural Network toolbox, I cannot run this code.
% must be checked
%% ---------------- %%


%error check

if ~isfield(method.param{1},'intype')
    warning('method.param.intype is not set, we set 2');
    method.param{1}.intype=2;
end
if ~isfield(method.param{1},'outtype')
    warning('method.param.intype is not set, we set 2');
    method.param{1}.outtype=2;
end
if ~isfield(method.param{1},'epochs')
    warning('method.param.epochs is not set, we set 100');
    method.param{1}.epochs=100;
end
if ~isfield(method.param{1},'Cost')
    warning('method.param.Cost is not set, we set 100');
    method.param{1}.Cost=0.1;
end
if ~isfield(method.param{1},'min_max')
    warning('method.param.epochs is not set, we set 100');
    method.param{1}.min_max=zeros(numN,2);
end

%% Initialization
[numN numF]=size(X);
[numNL,numL]=size(Y);
hidden_neuron=method.param{1}.hidden_neuron;
alpha=method.param{1}.alpha;
Cost=method.param{1}.Cost;
epochs=method.param{1}.epochs;
intype=method.param{1}.intype;
outtype=method.param{1}.outtype;
min_max=method.param{1}.min_max;

%Learning model
time=cputime;
[model{1},model{2}]=BPMLL_train_raw(X,Y', ... 
    hidden_neuron,alpha,epochs,intype,outtype,Cost,min_max);
time=cputime-time;


