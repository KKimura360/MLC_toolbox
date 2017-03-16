function  result2TexTable(meanMat,stdMat,rowNames,colNames,filename)
%% Input
%% meanMat: mean of result matrix (#methods x #datasets)
%% stdMat : std of  matrix (#methods x #datasets)
%% rowNames: a cell of method names
%% colNames: a cell of dataset names
%% filename: (option) file name to save the table


%% Header (set you want)
Header= '\\begin{table}[t]';
Ops1=    '\\centering';
Tab=    '\\begin{tabular}{';
for i=1:(length(colNames)+1)
    Tab=[Tab, 'c'];
end
Tab=[Tab,'}'];
Ops2='\\hline';
Topline='Method';
for i=1:length(colNames)
    Topline=[Topline,blanks(1), '&', blanks(1),colNames{i}];
end
Topline=[Topline,blanks(1), '\\\\',blanks(1),'\\hline'];

%% Table
Lines=cell(length(rowNames),1);
for i=1:length(rowNames)
    Lines{i}=getTableLine(rowNames{i},meanMat(i,:),stdMat(i,:));
end

%% Footer
Lines{end}=[Lines{end}, blanks(1), '\\hline'];
Footer=['\\end{tabular}',blanks(1),'\\end{table}'];


fid=fopen(filename,'wt');
fprintf(fid,[Header, '\n']);
fprintf(fid,[Ops1, '\n']);
fprintf(fid,[Tab, '\n']);
fprintf(fid,[Ops2, '\n']);
fprintf(fid,[Topline, '\n']);
for i=1:length(rowNames)
    fprintf(fid,[Lines{i}, '\n']);
end
fprintf(fid,[Footer, '\n']);
fclose(fid);




