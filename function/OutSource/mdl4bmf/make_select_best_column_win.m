%% Compiles select_best_column[[_sparse]_par].c files
% Windows doesn't support pthreads
%mex CFLAGS="$CFLAGS -std=c99 -O2" -v select_best_column_par.c
mex CFLAGS="$CFLAGS -std=c99 -O2" -v -largeArrayDims select_best_column.c
%mex CFLAGS="$CFLAGS -std=c99 -O2" -v -largeArrayDims ...
%    select_best_column_sparse_par.c