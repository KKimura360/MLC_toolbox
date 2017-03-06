function [method]=paramChange(method,change,countParam)
changeInd=find(strcmpi(method.name,change.name));
eval(['method.param{changeInd}.',change.param,'=',num2str(change.value(countParam)),';']);
end