function HammingScore = Hamming_score(Pred,Yt)
%HAMMINGSCORE Hamming Score (1 - Hamming Loss)
%
%          Pred          L x Nt predicted label matrix           
%          Yt            L x Nt groundtruth label matrix

    [numL,numNt] = size(Pred);
    HammingScore = sum(sum(Pred==Yt)) / (numL*numNt);
    
end