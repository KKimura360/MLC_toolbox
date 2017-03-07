function getLabelAnalysis(Y,Yt,pred,conf)
%% Input
%Y: Label matrix for traning
%Yt: Label matrix for test
%pred: prediction result (binary)
%conf: prediction result (confidence value)
%% Output
% Figures


[numNt,numL]=size(Yt);

[rankY,rankInd]=sort(sum(Y),'descend');
[rankYt]=sum(Yt);
rankYt=rankYt(rankInd);

Y=Y(:,rankInd);
Yt=Yt(:,rankInd);
conf=conf(:,rankInd);
pred=pred(:,rankInd);


%Dataset infromation
figure;
subplot(2,1,1);
bar(rankY);
title('#instances of training');
grid on;

subplot(2,1,2);
bar(rankYt);
title('#instance of test');
grid on;

%Label-based  true/false - positive/negative result
figure;
%true positive 
TP= sum((pred==1) .* (Yt==1));
normTP= TP./ (sum(Yt==1));
%false positive
FP= sum((pred==1) .* (Yt==0));
normFP= FP ./ (sum(Yt==0));
%true negative
TN= sum((pred==0) .* (Yt==0));
normTN= FP ./ (sum(Yt==0));
%false negative
FN= sum((pred==0) .* (Yt==1));
normFN= FN ./ (sum(Yt==1));

subplot(4,1,1)
bar(TP);
title('True-Positive, (Yhat=1,Y=1)');
grid on;

subplot(4,1,2)
bar(FP);
title('False-Positive, (Yhat=0,Y=1)');
grid on;
subplot(4,1,3)
bar(TN);
title('True-Negative, (Yhat=0,Y=0)');
grid on;
subplot(4,1,4)
bar(FN);
title('False-Negative, (Yhat=0,Y=1)');
grid on;

figure;

subplot(4,1,1)
bar(normTP);
title('Ratio:True-Positive, (Yhat=1,Y=1)/ sum(Y=1)');
grid on;

subplot(4,1,2)
bar(normFP);
title('Ratio: False-Positive, (Yhat=1,Y=0)/ sum(Y=0)');
grid on;
subplot(4,1,3)
bar(normTN);
title('Ratio:True-Negative, (Yhat=0,Y=0)/ sum(Y=0)');
grid on;
subplot(4,1,4)
bar(normFN);
title('Ratio:False-Negative, (Yhat=0,Y=1)/ sum(Y=1)');
grid on;


%% label-based auc 
figure;
subplot(3,1,1);
Labelauc=zeros(numL,1);
for label=1:numL
	Labelauc(label)=auc(Yt(:,label),conf(:,label));
end
Labelauc(isnan(Labelauc))=0;
bar(Labelauc);
title('Label-based AUC');
grid on
%% top-1 contribution
subplot(3,1,2);
% top-1 predicition
[~,ind]=max(conf');
tmppred=zeros(numNt,numL);
tmppred(:,ind)=1;
tmppred= tmppred .* Yt;
cont=sum(tmppred) ./ numNt;
bar(cont')
title('Top-1 contribution')
grid on;


%% top-3 contribution
subplot(3,1,3);

[~,ind]=sort(conf,2,'descend');
tmppred=zeros(numNt,numL);
for i=1:numNt
    tmppred(i,ind(i,1:3))=1;
end
tmppred= tmppred .* Yt;
cont=sum(tmppred) ./ (numNt);
bar(cont')
title('Top-3 contribution')
grid on;
