function[conf,time]=Topk_test(X,Y,Xt,model,method)
%X: Feature matrix (NxF)
%Y: Label matrix (NxL)
%method: method.base.param.k is number of nerarest neighbor is needed
%model: not used 
%% Output
%conf: confidence value of test instances for the label (Nt x1 real-value vector)

%time: computation time for the prediction
[numN,~]=size(X);
[numNt,~]=size(Xt);
[~,numL]=size(Y);
numk=method.param{1}.numk;
numk=min(numL,numk);
conf=zeros(numNt,numL);
time=cputime;

labelRank=model;

for i=1:numk
    conf(:,labelRank(i))=1-(0.01)*i;
end
for i=(numk+1):numL-numk
    conf(labelRank(i))=0.5-(0.01)*i;
end
time=cputime-time;
