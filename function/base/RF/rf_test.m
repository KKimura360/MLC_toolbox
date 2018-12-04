function[output,method,time]=rf_test(X,Y,Xt,model,method)
shogun
time=cputime;
labels_predict = model.apply_multiclass( RealFeatures(transpose(Xt)) );
output = labels_predict.get_labels();
output = transpose(output);
time=cputime-time;
endfunction