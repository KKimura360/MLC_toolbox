function[model,time]=PS_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%% Output
%model: A learned model (cell(method.param{x}.numM+1,1))
%model{1:numM}: classifiers (depends on called method)
%model{numM+1}: cell(numM,1), information of sampled labels
%% Option
%method.param{x}.type: 'disjoint', it calls RAkELd
%% Reference (APA style from google scholar)
%Read, J., Pfahringer, B., & Holmes, G. (2008, December). Multi-label classification using ensembles of pruned sets. In Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on (pp. 995-1000). IEEE.
%%% Method
%% Initialization
[numN,numF]=size(X);
[numNL,numL]=size(Y);
%error check 
%size check
sizeCheck;

%Initiliazation
[labelSet,~,newY]=unique(Y,'rows');
labelCount=histc(newY,1:max(newY));
[sortedCount,sortedId]=sort(labelCount,'descend');
type=method.param{1}.type;
numClass=method.param{1}.numClass;
time=cell(2,1);
tmptime=cputime;


switch type
    case 'A'
        if numClass > length(sortedId)
            fprintf('the number of unique labelsets are smaller than user parmeters\n');
            keepClass=sortedId;
            prunedClass=[];
        end
        keepClass=sortedId(1:numClass);
        prunedClass=sortedId(numClass+1:end);
    case 'B'
        keepClass=sortedId(sortedCount>=numClass);
        prunedClass=sortedId(sortedCount<numClass);
    otherwise
        error('wrong param.type\n');
end

%number of class (unique labelsets)
numPClass=length(prunedClass);
%kept label matrix ( #kept labelsets x Label) 
keepLabel=labelSet(keepClass,:);
addX=[];
addY=[];
for i=1:numPClass
    %find instances belongs to this pruned class
    prunedIns=zeros(numN,1);
    prunedInd=(newY==prunedClass(i));
    prunedIns(prunedInd)=1;
    %L-dimension representation of this pruned class
    prunedLabelset=labelSet(prunedClass(i),:);
    %find subsets from kept class
    %using matrix multiplication
    countVec= keepLabel * prunedLabelset';
    %if some kept class are subsets of this pruned class,
    % the matrix multiplication result(value) becomes the same to the
    % number of coressponding kept class
    % ex) a kept class 101, pruned class 111, these product returns
    % 2 which is the same number of labels the kept class has.        
    subsetVec=(sum(keepLabel,2)==countVec);
    for j=1:length(subsetVec);
        if subsetVec(j)==1
            tmpX=X(logical(prunedIns),:);
            tmpClass=ones(sum(prunedIns),1)*keepClass(j);
            addX=[addX; tmpX];
            addY=[addY; tmpClass];
        end
    end
end
    newX=[X;addX];
    newY=[newY;addY];
    newY=labelSet(newY,:);

%Output
model=cell(3,1);
model{2}=newX;
model{3}=newY;

time{end}=cputime-tmptime;

%Learning
fprintf('CALL: %s \n',method.name{2});

%Call method
[model{1},time{1}]=feval([method.name{2},'_train'],newX,newY,Popmethod(method));


