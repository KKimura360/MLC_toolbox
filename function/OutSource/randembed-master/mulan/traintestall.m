more off;

if exist('poissrnd') == 0
  error('*** you are missing poissrnd: try ''make poissrnd.m'' ***')
end

fprintf('*** bibtex *** \n');
traintestone('bibtex.mat')

fprintf('*** corel5k *** \n');
traintestone('corel5k.mat')

fprintf('*** mediamill *** \n');
traintestone('mediamill.mat')

fprintf('*** yeast *** \n');
traintestone('yeast.mat')

% >> traintestall
% *** bibtex *** 
% Elapsed time is 54.680947 seconds.
% .
% iter = 1, bestham.loss = 0.027757
% iter = 1, bestmicro.loss = 0.383647
% iter = 1, bestmacro.loss = 0.410417.
% iter = 2, bestham.loss = 0.00773141
% iter = 2, bestmicro.loss = 0.698498
% iter = 2, bestmacro.loss = 0.551683........
% iter = 10, bestham.loss = 0.00552166
% iter = 10, bestmicro.loss = 0.801857
% iter = 10, bestmacro.loss = 0.721935.....
% iter = 15, bestham.loss = 0.00386835
% iter = 15, bestmicro.loss = 0.853396
% iter = 15, bestmacro.loss = 0.784076.............................
% iter = 44, bestmicro.loss = 0.854935
% iter = 44, bestmacro.loss = 0.792862...................
% iter = 63, bestmacro.loss = 0.792942......................................................................................
% iter = 149, bestham.loss = 0.00372349
% iter = 149, bestmicro.loss = 0.867665
% iter = 149, bestmacro.loss = 0.801075.......................................................................................................................................................
% iter=149 f=6033 rfac=0.254184 lambda=0.0113685 logisticiter=13 eta=1.11205 alpha=0.760568 decay=1.04386 kern=qg shrink=0.997041
% per-ex inference: (hamming) [0.013806,0.0140469,0.0145459] (micro) [0.425748,0.432951,0.439951]
% iter=149 f=6033 rfac=0.254184 lambda=0.0113685 logisticiter=13 eta=1.11205 alpha=0.760568 decay=1.04386 kern=qg shrink=0.997041
% per-ex inference: (hamming) [0.0138617,0.0141342,0.014536] (micro) [0.419229,0.427895,0.44008]
% iter=149 f=6033 rfac=0.254184 lambda=0.0113685 logisticiter=13 eta=1.11205 alpha=0.760568 decay=1.04386 kern=qg shrink=0.997041
% per-class inference: (micro) [0.414274,0.423855,0.429152] (macro) [0.341515,0.352813,0.360762]
% Elapsed time is 11177.220940 seconds.
% 
% ans = 
% 
%     calmls_embed_test_hamming: [0.0138 0.0140 0.0145]
%     calmls_embed_test_microF1: [0.4192 0.4279 0.4401]
%     calmls_embed_test_macroF1: [0.3415 0.3528 0.3608]
% 
% *** corel5k *** 
% Elapsed time is 8.694459 seconds.
% .
% iter = 1, bestham.loss = 0.0803996
% iter = 1, bestmicro.loss = 0.0690442
% iter = 1, bestmacro.loss = 0.024685.
% iter = 2, bestham.loss = 0.0108422
% iter = 2, bestmicro.loss = 0.194533
% iter = 2, bestmacro.loss = 0.0528654.
% iter = 3, bestmicro.loss = 0.232446
% iter = 3, bestmacro.loss = 0.0586099.
% iter = 4, bestmicro.loss = 0.261883
% iter = 4, bestmacro.loss = 0.088307.
% iter = 5, bestham.loss = 0.00857772
% iter = 5, bestmicro.loss = 0.305184
% iter = 5, bestmacro.loss = 0.141851...
% iter = 8, bestmicro.loss = 0.306519..
% iter = 10, bestmicro.loss = 0.312286..............
% iter = 24, bestmacro.loss = 0.154043...........
% iter = 35, bestham.loss = 0.00849316..
% iter = 37, bestmicro.loss = 0.316424.........................................................................
% iter = 110, bestmicro.loss = 0.340279
% iter = 110, bestmacro.loss = 0.177117..............................................................................................................................................................................................
% iter=35 f=2485 rfac=0.338464 lambda=0.00375866 logisticiter=3 eta=0.387659 alpha=0.741059 decay=1.01127 kern=qg shrink=0.989791
% per-ex inference: (hamming) [0.00953729,0.00966926,0.00977319] (micro) [0.19338,0.20714,0.223853]
% iter=110 f=3258 rfac=0.369619 lambda=0.0483387 logisticiter=12 eta=1.23693 alpha=0.879017 decay=0.904602 kern=qg shrink=0.995656
% per-ex inference: (hamming) [0.0101803,0.0104945,0.0107829] (micro) [0.212048,0.225301,0.243862]
% iter=110 f=3258 rfac=0.369619 lambda=0.0483387 logisticiter=12 eta=1.23693 alpha=0.879017 decay=0.904602 kern=qg shrink=0.995656
% per-class inference: (micro) [0.200555,0.208377,0.227094] (macro) [0.0662028,0.070036,0.0725056]
% Elapsed time is 13953.376338 seconds.
% 
% ans = 
% 
%     calmls_embed_test_hamming: [0.0095 0.0097 0.0098]
%     calmls_embed_test_microF1: [0.2120 0.2253 0.2439]
%     calmls_embed_test_macroF1: [0.0662 0.0700 0.0725]
% 
% *** mediamill *** 
% Elapsed time is 3.339075 seconds.
% .
% iter = 1, bestham.loss = 0.0290403
% iter = 1, bestmicro.loss = 0.586833
% iter = 1, bestmacro.loss = 0.270726.
% iter = 2, bestham.loss = 0.027716
% iter = 2, bestmicro.loss = 0.626573
% iter = 2, bestmacro.loss = 0.359596.
% iter = 3, bestmacro.loss = 0.372402.
% iter = 4, bestmicro.loss = 0.627085
% iter = 4, bestmacro.loss = 0.388492.
% iter = 5, bestmicro.loss = 0.629109...
% iter = 8, bestmicro.loss = 0.631867........
% iter = 16, bestmacro.loss = 0.39252...
% iter = 19, bestmacro.loss = 0.406736...........
% iter = 30, bestmicro.loss = 0.634464.....................
% iter = 51, bestham.loss = 0.0275912............................................................
% iter = 111, bestham.loss = 0.0275153...............................................
% iter = 158, bestmicro.loss = 0.63876...................................................................................................
% iter = 257, bestham.loss = 0.0273437...........................................
% iter=257 f=7423 rfac=0.949227 lambda=0.00232225 logisticiter=14 eta=0.538104 alpha=0.336524 decay=1.01257 kern=qg shrink=0.994138
% per-ex inference: (hamming) [0.0303407,0.0305782,0.0308735] (micro) [0.573255,0.575907,0.579891]
% iter=158 f=7988 rfac=1.30998 lambda=0.00935002 logisticiter=15 eta=0.668042 alpha=0.553822 decay=0.907618 kern=m5 shrink=0.997974
% per-ex inference: (hamming) [0.0311964,0.0313608,0.0315819] (micro) [0.573226,0.576045,0.579335]
% iter=19 f=7425 rfac=2.87166 lambda=0.0435527 logisticiter=10 eta=0.790179 alpha=0.270301 decay=0.953439 kern=m5 shrink=0.996419
% per-class inference: (micro) [0.529649,0.534308,0.538598] (macro) [0.254114,0.261659,0.267115]
% Elapsed time is 25428.961521 seconds.
% 
% ans = 
% 
%     calmls_embed_test_hamming: [0.0303 0.0306 0.0309]
%     calmls_embed_test_microF1: [0.5732 0.5760 0.5793]
%     calmls_embed_test_macroF1: [0.2541 0.2617 0.2671]
% 
% ...
% *** yeast ***
% Elapsed time is 0.211980 seconds.
% .
% iter = 1, bestham.loss = 0.285053
% iter = 1, bestmicro.loss = 0.54332
% iter = 1, bestmacro.loss = 0.408542.
% iter = 2, bestham.loss = 0.25517             
% iter = 2, bestham.loss = 0.25517
% iter = 2, bestmicro.loss = 0.576496
% iter = 2, bestmacro.loss = 0.439882.
% iter = 3, bestham.loss = 0.22184
% iter = 3, bestmicro.loss = 0.61772
% iter = 3, bestmacro.loss = 0.461663.
% iter = 4, bestham.loss = 0.203202
% iter = 4, bestmicro.loss = 0.637412
% iter = 4, bestmacro.loss = 0.498576........
% iter = 12, bestham.loss = 0.19667
% iter = 12, bestmicro.loss = 0.654269
% iter = 12, bestmacro.loss = 0.512157..........................................................
% iter = 70, bestham.loss = 0.195331
% iter = 70, bestmicro.loss = 0.662345
% iter = 70, bestmacro.loss = 0.516658...................
% iter = 89, bestham.loss = 0.190507
% iter = 89, bestmicro.loss = 0.673972..................
% iter = 107, bestmacro.loss = 0.528201........................................................................
% iter = 179, bestham.loss = 0.187321
% iter = 179, bestmicro.loss = 0.678836..........................................................................
% iter = 253, bestmacro.loss = 0.53113...............................................
% iter=179 f=1638 rfac=1.38845 lambda=0.0977787 logisticiter=6 eta=0.835026 alpha=0.52952 decay=0.936713 kern=qm5 shrink=0.982692
% per-ex inference: (hamming) [0.214141,0.222345,0.230537] (micro) [0.588802,0.603773,0.618046]
% iter=179 f=1638 rfac=1.38845 lambda=0.0977787 logisticiter=6 eta=0.835026 alpha=0.52952 decay=0.936713 kern=qm5 shrink=0.982692
% per-ex inference: (hamming) [0.213379,0.221414,0.227405] (micro) [0.588937,0.602654,0.616921]
% iter=253 f=3228 rfac=0.565755 lambda=0.0801377 logisticiter=3 eta=0.900789 alpha=0.0849182 decay=1.00775 kern=qg shrink=0.973547
% per-class inference: (micro) [0.611606,0.618844,0.627229] (macro) [0.468118,0.478474,0.485116]
% Elapsed time is 4383.160513 seconds.
% 
% ans =
% 
%     calmls_embed_test_hamming: [0.2141 0.2223 0.2305]
%     calmls_embed_test_microF1: [0.5889 0.6027 0.6169]
%     calmls_embed_test_macroF1: [0.4681 0.4785 0.4851]
