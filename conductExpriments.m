function [res,conf,pred]=conductExpriments(method,numTrial,numCV,dataname)

[data,target,indices]=readData(dataname,numCV);

res=cell(numTrial,1);
conf=cell(numTrial,1);
pred=cell(numTrial,1);
for countTrial=1:numTrial
    index=indices(:,countTrial);
    res{countTrial}=cell(numCV,1);
    conf{countTrial}=cell(numCV,1);
    pred{countTrial}=cell(numCV,1);
    for countCV=1:numCV
        test = (index == countCV); 
        train = ~test; 
        data=sparse(data);
        X=data(train,:);
        Xt=data(test,:);
        Y=target(:,train')';
        Yt=target(:,test')';
        
         %training (will write a wrapper later)   
         [model]=MLC_train(X,Y,method);
         %testing 
         [conf{countTrial}{countCV}]=MLC_test(X,Y,Xt,model,method);
         %Thresholding
         [pred{countTrial}{countCV}]=Thresholding(conf{countTrial}{countCV},method.th);
         %Evalution
         [res{countTrial}{countCV}]=Evaluation(Yt,conf{countTrial}{countCV},pred{countTrial}{countCV});
    end
end
