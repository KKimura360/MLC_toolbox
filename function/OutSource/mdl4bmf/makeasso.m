%% Selects the correct system
switch mexext
    case 'mexa64'
        makeasso_linux
    case 'mexmaci64'
        makeasso_mac
    case 'mexw64'
        makeasso_win
    otherwise
        error('Your platform is not supported; perhaps it is 32-bit. Mex extension is %s.', mexext);
end