
numMethod=length(method.name);
%%DataSet 
fprintf(['Dataset  :',dataname,' ',num2str(numCV),'-fold\n']);

%%Overview
Dispstring='';
for i=1:numMethod
    if i==numMethod
        Dispstring=[Dispstring,method.name{i}];
    else
        Dispstring=[Dispstring,method.name{i},' => '];
    end
end
Dispstring=[Dispstring,' => ',method.base.name];
fprintf(['Overview :',Dispstring,'\n']);

%%Threshold method
fprintf(['Threshold:',method.th.type,'\n']);



