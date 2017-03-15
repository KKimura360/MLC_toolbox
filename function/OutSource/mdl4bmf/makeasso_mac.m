
%% compile, don't link yet


% asso
mex CFLAGS="\$CFLAGS -O3" -Iasso-utils/ -largeArrayDims -v -lm -c -DMATLAB ./asso.c

% connection point
mex CFLAGS="\$CFLAGS -O3" -Iasso-utils/ -largeArrayDims -O -v -lm -c -DMATLAB  ./matlabasso.c

%% link

mex -Iasso-utils/ -largeArrayDims -O -v -lm -output asso -DMATLAB ...
  ./asso.o ...
  ./matlabasso.o

%% remove object files

delete *.o
