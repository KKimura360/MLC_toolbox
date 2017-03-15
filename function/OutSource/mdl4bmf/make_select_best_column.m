%% Selects the correct system
switch mexext
    case 'mexa64'
        make_select_best_column_linux
    case 'mexmaci64'
        makea_select_best_column_mac
    case 'mexw64'
        warning('MAKE:windows', 'Windows only supports non-threaded versions')
        make_select_best_column_win
    otherwise
        error('Your platform is not supported; perhaps it is 32-bit. Mex extension is %s', mexext);
end