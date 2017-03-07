% SizeCheck of the MLC

if numN~=numNL
    if numN==numL
        warning('the label matrix may be transposed\n');
        Y=Y';
        [numNL,numL]=size(Y);
    elseif numF==numNL
        warning('the feature matrix may be transposed\n');
        X=X';
        [numN,numF]=size(X);
    else
        size(X)
        size(Y)
        error('the label matrix and the feature matix must be a same index')
    end
end