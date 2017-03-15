% This make.m is for MATLAB and OCTAVE under Windows, Mac, and Unix
% Set up mex before running this

try
    mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims read_data.cpp
    mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims write_data.cpp
catch
	fprintf('If make.m fails, please check README about detailed instructions.\n');
end
