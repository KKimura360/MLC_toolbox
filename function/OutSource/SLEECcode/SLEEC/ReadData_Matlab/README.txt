All commands are to be executed from Matlab terminal

Compiling:
==========

make


Usage:
======

Converting from text file to matlab matrices:
---------------------------------------------
[<output feature mat>,<output label mat>] = read_data(<data_file_name>);
Example: [ft_mat,lbl_mat] = read_data('./data.txt');

Converting from matlab matrices to text file:
---------------------------------------------
write_data(<input feature mat>,<input label mat>,<data_file_name>);
Example: write_data(ft_mat,lbl_mat,'./data.txt');

