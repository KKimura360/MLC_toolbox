function [M C]=PairwiseConst(Y,type);
%%Input
%Y: label matrix Instance x Label
%type: how to construct 
%    :'Exact'
%    :'SimDis'

%Initialization
[N,L]=size(Y);
M=zeros(size(Y,1));
C=M;


if strcmp(type,'Exact');
    nY=zeros(N,L);
    for n=1:N
        nY(n,:)=Y(n,:) ./ norm(Y,2);
    end
    nY= nY*nY';
    M(nY==1)=1;
    C(nY==0)=1;
else %strcmp Dissim
    % Here using Jaccard Sim 
    C=squareform(pdist(Y,'jaccard'));
    M=1-C;
end

    
