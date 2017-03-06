function [method]=SetALLParams(method)
    method.param=cell(length(method.name),1);
    for countMethod= 1:length(method.name)
        [method.param{countMethod}]=feval(['Set',method.name{countMethod},'Parameter'],[]);
    end
end