function MicroF1 = Micro_F1(Yt,Pred)
%MICROF1 Micro-averaged F1 Measure
%
%          Yt         L x Nt groundtruth label matrix
%          Pred       L x Nt predicted label matrix

intSec = Yt(:)' * Pred(:);
dePre = nnz(Yt);
if dePre ~= 0
    precision = intSec / dePre;
else
    precision = 0;
end
deRec = nnz(Pred);
if deRec ~= 0
    recall = intSec / deRec;
else
    recall = 0;
end
if recall~=0 || precision~=0
    MicroF1 = 2*precision*recall / (precision+recall);
else
    MicroF1 = 0;
end

end