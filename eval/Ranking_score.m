function RankScore = Ranking_score(Conf,lcell,nlcell,lSize)
%RankingScore: computing the ranking score (RankScore = 1 - RankLoss)
%   Conf: confidence matrix in L x Nt
%   Yt:   ground truth target matrix in L x Nt

[numL,numNt] = size(Conf);
rankloss=0;
for i=1:numNt
    if (lSize(i) == 0)
        continue;
    end
    temp=0;
    for m=1:lSize(i)
        for n=1:(numL-lSize(i))
            if(Conf(lcell{i,1}(m),i)<=Conf(nlcell{i,1}(n),i))
                temp=temp+1;
            end
        end
    end
    rankloss=rankloss+temp/(m*n);
end
RankScore=1-rankloss/nnz(lSize);

end