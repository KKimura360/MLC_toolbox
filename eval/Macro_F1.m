function MacroF1 = Macro_F1(Yt,Pred)
%MACROF1 Macro-averaged F1 Measure
%
%          Yt         L x Nt groundtruth label matrix
%          Pred       L x Nt predicted label matrix 

numL = size(Yt,1);
lFscore = 0;
for i = 1 : numL
    intSec = Yt(i,:)*Pred(i,:)';
    dePre = nnz(Pred(i,:));
    if dePre ~= 0
        precision = intSec / dePre;
    else
        precision = 0;
    end
    deRec = nnz(Yt(i,:));
    if deRec ~= 0
        recall = intSec / deRec;
    else
        recall = 0;
    end
    if recall~=0 || precision~=0
        lFscore = lFscore + 2*recall*precision/(recall+precision);
    end
end
MacroF1 = lFscore / numL;
    
end