function[conf,time]=CLR_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt: Feature Matrix (NtxF) for test data
%model learned by BR_train
%% Output
%conf: confidence values (Nt x L);
%linear_svm does not return confidence value since LIBLINEAR does not
%support it. 

%errorcheck
[numN,numF]=size(X);
[numNL,numL]=size(Y);

%initialization
[numNt,~]=size(Xt);
conf=zeros(numNt,numL);
allComb=model{end};
numComb=size(allComb,1);
time=cputime; 
for i =1:numNt 
    tmpXt=Xt(i,:);
    %initial pair-wise comparison
    [tmpconf]=feval([method.base.name,'_test'],X,Y,tmpXt,model{1},method);
    %classifier compare (y1,y2) and y1=1,y2=0
    %thus, 
    posLabel=allComb(1,1);
    negLabel=allComb(1,2);
    % if classifier returns over 0.5, paiw-wise rank is y1 >y2
    % else y2 > y1 
    if tmpconf >=0.5
        labelRank=[posLabel,negLabel];
    else
        labelRank=[posLabel,negLabel];
    end
    %label list includes calibrate label
    labelList=1:(numL+1);
    % delete already compared labels
    labelList(labelList==allComb(1,1))=[];
    labelList(labelList==allComb(1,2))=[];
    % We use insert sort and compare from right (lowest label) to left
    % (highest rank)
    for j=1:length(labelList)
        %new label to add ranking
        targetLabel=labelList(j);
        %to find models, which has the targetLabel to compare 
        targetInd= logical(sum((allComb==targetLabel),2));
        for k=1:length(labelRank)
            %from right to left, we choose considered label
            rankInd=length(labelRank)-k+1;
            lowLabel=labelRank(rankInd);
            %to find models, which has the lowLabel to compare
            lowInd=logical(sum((allComb==lowLabel),2));
            % to find the model, which has both targetLabel and lowLabel
            modelInd=find((targetInd .*lowInd)>0);
            %compare the ranking targetLabel vs. lowLabel
            [tmpconf]=feval([method.base.name,'_test'],X,Y,tmpXt,model{modelInd},method);
            % if the applied model is (targetLabel,lowLabel)
            if targetLabel==allComb(modelInd,1)
                if tmpconf >=0.5
                    %if targetLabel won all the other labels in labelRank
                    if k==length(labelRank)
                        labelRank=[targetLabel,labelRank];
                    end
                    %targetLabel won lowLabel, move to left
                    continue;
                else
                    % targetLabel lost, insert targetLabel here
                    if k==1
                        % if targetLabel lost the lowest label
                        labelRank=[labelRank,targetLabel];
                    else
                        % if targetLabel lost the middle label
                        % insert rifght of rankInd and break
                        labelRank=[labelRank(1:(rankInd)),targetLabel,labelRank(rankInd+1:end)];
                        break;
                    end
                end
            % if the applied model is (lowLabel,targetLabel)    
            else
                % the sign is reversed 
                if tmpconf <0.5
                     if k==length(labelRank)
                        labelRank=[targetLabel,labelRank];
                    end
                    continue;
                else
                    if k==1
                        labelRank=[labelRank,targetLabel];
                    else
                        labelRank=[labelRank(1:(rankInd)),targetLabel,labelRank(rankInd+1:end)];
                        break;
                    end
                end
            end
        end
    end
    % After obtain ranking, we delete calibrated label
    calInd=find(labelRank==(numL+1));
    posLabels=labelRank(1:calInd);
    negLabels=labelRank(calInd:end);
    posLabels(posLabels==(numL+1))=[];
    negLabels(negLabels==(numL+1))=[];
    % to keep ranking result in the confidence value
    conf(i,posLabels)=0.5+ 0.1 *(1:length(posLabels));
    conf(i,negLabels)=0.5- 0.1 *(1:length(negLabels));

end
time=cputime-time;
            
            

