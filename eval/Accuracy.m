function accuracy =Accuracy(Pred,Yt)
%Accuracy: Compute the classification accuracy
%
%          Pred          L x Nt predicted label matrix           
%          Yt            L x Nt groundtruth label matrix

numNt = size(Yt,2);
accuracy = 0;
count = numNt;
for i=1:numNt
    numerator = Yt(:,i)'*Pred(:,i);
    denominator = sum(or(Yt(:,i),Pred(:,i)));
    if denominator ~= 0
        accuracy = accuracy + numerator/denominator;
    else 
        count = count - 1;
    end
end
accuracy = accuracy / count;

end