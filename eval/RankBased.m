function [Conf,lcell,nlcell,lSize] = RankBased(Conf,Yt)
%RankBased: computing the intermediate values for Rank-based metrics
%   Conf: confidence matrix in L x Nt
%   Yt:   ground truth target matrix in L x Nt

numL=size(Conf,1);
id = find(sum(Yt,2)==numL);
Yt(:,id) = [];
Conf(:,id) = [];

% Save the label set in lcell
numNt=size(Conf,2);
lcell=cell(numNt,1);
nlcell=cell(numNt,1);
lSize=zeros(numNt,1);
for i=1:numNt
    temp=Yt(:,i);
    tmpSize = sum(temp==ones(numL,1));
    if tmpSize == 0
        continue;
    else
        lSize(i,1)=tmpSize;
    end
    for j=1:numL
        if(temp(j)==1)
            lcell{i,1}=[lcell{i,1},j];
        else
            nlcell{i,1}=[nlcell{i,1},j];
        end
    end
end

end