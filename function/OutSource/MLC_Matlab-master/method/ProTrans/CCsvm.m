function [Pre_Labels,Outputs] = CCsvm(train_data,train_target,test_data,test_target,svm)
%CC The Classifier Chains method for MLC

% Check the capability of input parameters
if (nargin < 4)
    error('Not enough input parameters');
elseif (nargin < 5)
    svm.type = 'Linear';
    svm.para = [];
end

% Specify the parameters of SVM
switch svm.type
    case 'RBF'
        gamma = num2str(svm.para);
        str=['-t 2 -g ',gamma,' -b 1'];
    case 'Poly'
        gamma=num2str(svm.para(1));
        coef=num2str(svm.para(2));
        degree=num2str(svm.para(3));
        str=['-t 1 ','-g ',gamma,' -r ', coef,' -d ',degree,' -b 1'];
    case 'Linear'
        str='-t 0 -b 1';
    otherwise
        error('SVM types not supported, please type "help LIFT" for more information');
end

% Training and testing
[num_class,num_test] = size(test_target);
chain = randperm(num_class);    
Pre_Labels = zeros(num_class,num_test);    
Outputs = zeros(num_class,num_test);   
pa = [];    
for i = chain
    if isempty(pa)
        model=svmtrain(train_target(i,:)',train_data,str);
        [predicted_label,~,prob_estimates]=svmpredict(test_target(i,:)',test_data,model,'-b 1');
        pos_index= model.Label==1;
        Outputs(i,:) = prob_estimates(:,pos_index)';
        Pre_Labels(i,:) = predicted_label';
    else
        model=svmtrain(train_target(i,:)',[train_data,train_target(pa,:)'],str);
        [predicted_label,~,prob_estimates]=svmpredict(test_target(i,:)',[test_data,Pre_Labels(pa,:)'],model,'-b 1');
        pos_index= model.Label==1;
        Outputs(i,:) = prob_estimates(:,pos_index)';
        Pre_Labels(i,:) = predicted_label';
    end
    pa = [pa,i];
end

end


