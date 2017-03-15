
%% compile, don't link yet


% asso
mex CFLAGS="$CFLAGS -O2" -Iasso-utils/ -largeArrayDims  -v -c -DMATLAB ./asso.c

% connection point
mex CFLAGS="$CFLAGS -O2" -Iasso-utils/ -largeArrayDims -O -v -c -DMATLAB  ./matlabasso.c

%% link

mex -Iasso-utils/ -largeArrayDims -O -v -output asso -DMATLAB  ./asso.obj ./matlabasso.obj

%% remove object files

delete *.o
