function Average_Precision=Average_precision(Conf,lcell,lSize)
%Average_Precision: computing the average precision
%   Conf: confidence matrix in L x Nt
%   Yt:   ground truth target matrix in L x Nt

[numL,numNt] = size(Conf);
aveprec=0;
for i=1:numNt
    if lSize(i) == 0
        continue;
    end
    temp=Conf(:,i);
    [~,index]=sort(temp);
    indicator=zeros(1,numL);
    for m=1:lSize(i)
        [~,loc]=ismember(lcell{i,1}(m),index);
        indicator(1,loc)=1;
    end
    count=0;
    for m=1:lSize(i)
        [~,loc]=ismember(lcell{i,1}(m),index);
        count=count+sum(indicator(loc:numL))/(numL-loc+1);
    end
    aveprec=aveprec+count/lSize(i);
end
Average_Precision=aveprec/nnz(lSize);

end