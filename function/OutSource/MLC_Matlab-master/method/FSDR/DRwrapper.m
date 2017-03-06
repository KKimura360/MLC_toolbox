function [train_data,test_data] = DRwrapper(train_data,test_data,train_target,alg)
%DRWRAPPER A wrapper of several FSDR approaches
%   ´Ë´¦ÏÔÊ¾Ï?¸ËµÃ?

% Applying a FSDR Approach
if (strcmp(alg,'HSL'))
    alg = [];
    alg.Lap_type = 'clique';
    alg.alg = '2s';
    alg.reg_2norm = 1;
    W = HSL(train_data',train_target,alg);
    train_data = train_data * W;
    test_data = test_data * W;
elseif (strcmp(alg,'OPLS'))
    alg = [];
    alg.alg = '2s';
    alg.reg_2norm = 1;
    W = OPLS(train_data',train_target,alg);
    train_data = train_data * W;
    test_data = test_data * W;
elseif (strcmp(alg,'PCA'))
    [train_data,test_data] = PCA(train_data,test_data,0.3);
elseif (strcmp(alg,'MLDA'))
    alg = [];
    alg.alg = '2s';
    alg.reg_2norm = 1;
    W=MLDA(train_data',train_target,alg);
    train_data = train_data * W;
    test_data = test_data * W;
elseif (strcmp(alg,'MDDM'))
    alg = [];
    alg.alg = 'ls';
    alg.reg_2norm = 1;
    W=MDDM(train_data',train_target,alg);
    train_data = train_data * W;
    test_data = test_data * W;
elseif (strcmp(alg,'DMDDM'))
    alg = [];
    alg.alg = 'ls';
    alg.reg_2norm = 1;
    W=DMDDM(train_data',train_target,alg);
    train_data = train_data * W;
    test_data = test_data * W;
elseif (strcmpi(alg,'ORI'))
    train_data = train_data;
    test_data = test_data;
    W=1;
elseif (strcmpi(alg,'CCA'))
    alg = [];
    alg.alg = 'ls';
    alg.reg_2norm = 1;
    W=CCA(train_data',train_target,alg);
    train_data = train_data * W;
    test_data = test_data * W;
elseif (strcmpi(alg,'NMF'))
    alg=[];
    alg.alg='tri';
    alg.reg_2norm=1;
    W=NMF(train_data',train_target,alg);
    train_data=train_data*W;
    test_data=test_data *W;
else
    disp('ERROR, unavailable DR approach');
    return;
end
fprintf('The dimension of W is %d \n',size(train_data,2)); 
end

