function[model,time]=FastXML_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.param: see setFastXML
%% Output
%model: A learned model of FastXML
%% Reference (APA style from google scholar)
% Prabhu, Y., & Varma, M. (2014, August). Fastxml: A fast, accurate and stable tree-classifier for extreme multi-label learning. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 263-272). ACM.
%%% Method

%% initialization
param=method.param{1};
X=sparse(X');
Y=sparse(Y');
model=cell(5,1);
time=cputime;
%% Learning model
file_count=1;
model_flag=false;
while(~model_flag)
    dicname=['tmp/',num2str(file_count)];
   if exist(dicname)>0
       file_count=file_count+1;
       continue;
   else
       %make directory
       mkdir(dicname);
       %make empty files for FastXML
       %for X
       filename=[dicname,'/tmp_X.txt'];
       X_name=filename;
       fid=fopen(filename,'W');
       fprintf(fid,'emptyfile');
       fclose(fid);
       %for Y
       filename=[dicname,'/tmp_Y.txt'];
       Y_name=filename;
       fid=fopen(filename,'W');
       fprintf(fid,'emptyfile');
       fclose(fid);
       %for Xt
       filename=[dicname,'/tmp_Xt.txt'];
       Xt_name=filename;
       fid=fopen(filename,'W');
       fprintf(fid,'emptyfile');
       fclose(fid);
       %for Yt
       filename=[dicname,'/tmp_Yt.txt'];
       Yt_name=filename;
       fid=fopen(filename,'W');
       fprintf(fid,'emptyfile');
       fclose(fid);
       %for model
       model_name=[dicname,'/model/'];
       mkdir(model_name);
       % Learn model
       fastXML_train_raw(X, Y, param,X_name,Y_name,model_name);
       model{1}=X_name;
       model{2}=Y_name;
       model{3}=Xt_name;
       model{4}=Yt_name;
       model{5}=model_name;
       model{6}=dicname;
       model_flag=true;
   end
end
time=cputime-time;