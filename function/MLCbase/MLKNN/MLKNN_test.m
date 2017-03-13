function[conf,time]=MLKNN_test(X,Y,Xt,model,method)
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
numk=min(numN,numk);
conf=zeros(numNt,numL);
type=method.param{1}.type;
time=cputime;

switch type
    case 1 
        % ML-Zhang KNN
        [Outputs]=MLKNN_test_raw(X,Y',Xt,numk,model{1},model{2},model{3},model{4});
        conf=Outputs';
    case 2 
        %SLEEC based simple knn
        time=cputime;
        % calc L2 distances between training and test instances 
        W=L2_distance(X',Xt'); % W is N x Nt matrix
        %sort with ascending order 
        [~,Ind]=sort(W); 
        for i= 1:numNt
            index=Ind(1:numk,i);
            conf(i,:)=sum(Y(index,:));
        end
        % divided by the number of nearest negihbors
        conf = conf ./ numk;
        time=cputime-time;
    otherwise
        error('type is wrong');
end
        time=cputime-time;
