function[rand_forest,method,time]=rf_train(X,y,method)
shogun
time=cputime;

features_train = RealFeatures(transpose(X));
labels_train = MulticlassLabels(transpose(y));

if ~isfield(method.base.param,'treecount')
    warning('RF tree count is not set, default setting is applied (RF)\n');
    method.base.param.treecount=100;
end

if ~isfield(method.base.param,'featurestoconsider')
    warning('Number of features to consider in RF is not set, default setting is applied (RF)\n');
    method.base.param.featurestoconsider=round(sqrt(size(X)(2)));
end

rand_forest = RandomForest(features_train, labels_train, method.base.param.treecount, method.base.param.featurestoconsider);
m_vote = MajorityVote();
rand_forest.set_combination_rule(m_vote);
rand_forest.train();
time=cputime-time;
endfunction