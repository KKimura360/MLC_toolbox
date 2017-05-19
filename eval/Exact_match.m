function ExactMatch = Exact_match(Pred,Yt)
%EXACTMATCH Exact Match (1 - Subset 0-1 loss)
%
%          Pred          L x Nt predicted label matrix           
%          Yt            L x Nt groundtruth label matrix

results = Pred ~= Yt;
ExactMatch = nnz(sum(results)==0) / size(Pred,2);

end