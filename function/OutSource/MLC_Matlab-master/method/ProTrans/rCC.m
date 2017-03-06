function [Pre_Labels,Outputs] = rCC(train_data,train_target,test_data,num_block)
%rCC randomized CC with Ridge Regression for acceleration
%   Incomplete

% Ridge parameter
lambda = 0.1;

% Randomly generate a chain order
num_label = size(train_target,1);
chain = randperm(num_label);

% Initialization
num_test = size(test_data,1);
Outputs = zeros(num_label,num_test);
Pre_Labels = zeros(num_label,num_test);

if (num_block < num_label)
    % Partition the chain into several blocks
    block_size = round(num_label/num_block);
    if (num_block * block_size < num_label)
        num_block = num_block + 1;
    end
    block = cell(num_block,1);
    for i = 1:num_block;
        if (i*block_size < num_label)
            block{i} = chain( ((i-1)*block_size+1) : (i*block_size) );
        else
            block{i} = chain( ((i-1)*block_size+1) : num_label );
        end
    end
    
    % Train classifier chains as blocks
    for i = 1:num_block
        if i == 1
            ww = ridgereg(train_target(block{i},:)',train_data,lambda);
            Outputs(block{i},:) = ( [ones(num_test,1),test_data] * ww )';
        else
            ww = ridgereg(train_target(block{i},:)',[train_data train_target(block{i-1},:)'],lambda);
            Outputs(block{i},:) = ( [ones(num_test,1),test_data,Pre_Labels(block{i-1},:)'] * ww )';
        end
        Pre_Labels(block{i},:) = round(Outputs(block{i},:));
    end
else
    % Perform CC instead
    pa = [];
    for i = chain
        if isempty(pa)
            ww = ridgereg(train_target(i,:)',train_data,lambda);
            Outputs(i,:) = [ones(num_test,1),test_data] * ww;
        else
            ww = ridgereg(train_target(i,:)',[train_data train_target(pa,:)'],lambda);
            Outputs(i,:) = [ones(num_test,1),test_data,Pre_Labels(pa,:)'] * ww;
        end
        Pre_Labels(i,:) = round(Outputs(i,:));
        pa = [pa i];
    end
end
end

