function fscore = FScore(Pred,Yt)
%FScore: Compute the FScore
%
%          Pred          L x Nt predicted label matrix           
%          Yt            L x Nt groundtruth label matrix

numNt  = size(Yt,2);
fscore = 0;
count  = numNt;
for i = 1 : numNt
    numerator = sum(and(Yt(:,i),Pred(:,i)));
    denominator = sum(Yt(:,i))+sum(Pred(:,i));
    if denominator~=0
        fscore = fscore+2*numerator/denominator;
    else
        count = count - 1;
    end
end
fscore = fscore / count;

end