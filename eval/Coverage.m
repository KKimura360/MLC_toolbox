function Coverage=Coverage(Conf,Yt)
%Coverage: computing the one error
%   Conf: confidence matrix in L x Nt
%   Yt:   ground truth target matrix in L x Nt

[numL,numNt]=size(Conf);
Label=cell(numNt,1);
notLabel=cell(numNt,1);
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
            Label{i,1}=[Label{i,1},j];
        else
            notLabel{i,1}=[notLabel{i,1},j];
        end
    end
end

cover=0;
for i=1:numNt
    temp=Conf(:,i);
    [~,index]=sort(temp);
    tmpmin=numL+1;
    for m=1:lSize(i)
        [~,loc]=ismember(Label{i,1}(m),index);
        if(loc<tmpmin)
            tmpmin=loc;
        end
    end
    cover=cover+(numL-tmpmin+1);
end
Coverage=((cover/numNt)-1)/numL;   % Normalized by numL

end