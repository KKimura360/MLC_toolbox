function[conf,time]=MLLEM_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt: Feature Matrix (NtxF) for test data
%model learned by BR_train
%% Output
%conf: confidence values (Nt x L);
%time: computation time 
%% Reference
% Kimura, K., Kudo, M., & Sun, L. (2016, November). Simultaneous Nonlinear Label-Instance Embedding for Multi-label Classification. In Joint IAPR International Workshops on Statistical Techniques in Pattern Recognition (SPR) and Structural and Syntactic Pattern Recognition (SSPR) (pp. 15-25). Springer International Publishing.

%% Initialization
[numN,numF]=size(X);
[~,numL]=size(Y);
[numNt,~]=size(Xt);
time=cputime;
type=method.param{1}.type;
G=model{2};
H=model{3};

switch type
    case {'L','linear','Linear'}
        W=model{1};
        Gt= [ones(numNt,1), Xt] * W;
    case {'NL','nonlinear','Nonlinear','NonLinear'}
        k3=method.param{1}.k3;
        tmp=L2_distance(Xt',X');
        [val Ind]=sort(tmp,2);
        Gt=zeros(size(tmp,1),dim); 
        for i= 1:size(tmp,1);
            Gt(i,:)=squeeze(mean(G(Ind(i,1:k3),:),1)); %Mean of k-nearest of neighbors
        end
    otherwise
        error('type is not surpported')
end
% classify for each label
conf=L2_distance(Gt',H');
conf=max(max(conf))-conf; % reversing scores
time=time-cputime;
