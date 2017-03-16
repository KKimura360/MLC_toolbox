function line=getTableLine(methodName,meanResult,stdResult)
    line=[methodName,blanks(1)];
    for i=1:length(meanResult);
        line=[line,'&', blanks(1),'$', num2str(meanResult(i),'%1.3f'), blanks(1), '\\pm', blanks(1), num2str(stdResult(i),'%1.3f'),'$',blanks(1)];
    end
    line=[line, blanks(1),'\\\\'];
end