function t = make_canonical(s, C)
% MAKE_CANONICAL - converts a prefix string into a canonical form
%
% Usage: t = MAKE_CANONICAL(s, C) returns t, the canonical form of
% s in C or empty string if s is not any (or unique)
% case-insensitive prefix of strings in C

t='';

n=length(s);
i=1;

while sum(double(strncmpi(s, C, i))) > 1 && i <= n
    i = i + 1;
end

TF = strncmpi(s, C, i); %% a bit of extra work

%% No prefix match
if sum(double(TF)) == 0,
    return;
end

%% Not unambiguous
if sum(double(TF)) > 1,
    return;
end

%% Prefix match for shorter than full prefix
if ~strncmpi(s, C{TF}, n),
    return;
end

%% a match
t = C{TF};
