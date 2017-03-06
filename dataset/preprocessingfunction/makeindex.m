
Datas={'20NG','bibtex','CAL500','corel5k','corel16k1','corel16k2','delicious','emotions','enron','flags','genbase','languagelog','mediamill','medical','scene','tmc2007','tmc2007-500','yeast'};

%dataname='20NG';
for num_data=1:length(Datas)
    dataname=Datas{num_data};
    load(['../matfile/',dataname,'.mat']);
    Folds = [3,5,10];
    for num_fold=Folds
        rng('default');
        indices=[];
        for i=1:10
            ind = crossvalind('Kfold',size(data,1),num_fold);
            data=sparse(data);
            indices=[indices ind];
        end
        save(['../index/',num2str(num_fold),'-fold/',dataname,'.mat'],'indices','-mat');
        indices=[];
    end
end