function OneError=One_error(Conf,lcell)
%One_error: computing the one error
%   Conf: confidence matrix in L x Nt
%   Yt:   ground truth target matrix in L x Nt

[numL,numNt]=size(Conf);
oneerr=0;
for i=1:numNt
    indicator=0;
    temp=Conf(:,i);
    [maxval,~]=max(temp);
    for j=1:numL
        if(temp(j)==maxval)
            if(ismember(j,lcell{i,1}))
                indicator=1;
                break;
            end
        end
    end
    if(indicator==0)
        oneerr=oneerr+1;
    end
end
OneError=oneerr/numNt;

end