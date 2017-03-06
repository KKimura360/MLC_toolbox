function [y_tr, x_tr, y_tt, x_tt] = read_dataset(DataSet)

FilePath_Ytr = [DataSet '/Y_tr'];
FilePath_Xtr = [DataSet '/X_tr'];
FilePath_Ytt = [DataSet '/Y_tt'];
FilePath_Xtt = [DataSet '/X_tt'];

y_tr = load(FilePath_Ytr);
x_tr = load(FilePath_Xtr);
y_tt = load(FilePath_Ytt);
x_tt = load(FilePath_Xtt);

