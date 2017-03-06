Multi-Label Classification with Label Space Dimension Reduction
======

The program consists of five linear label space transformation approaches.  The base learner used in all approaches is regularized linear regression with a fixed regularization parameter.  Please see the usage in demo.m .

* Binary Relevance with Random Discarding (BR), Principal Label Space Transformation (PLST) are developed in

  Farbound Tai and Hsuan-Tien Lin. Multilabel classification with principal label space transformation. Neural Computation, 24(9):2508--2542, September 2012.

* Conditional Principal Label Space Transformation (CPLST) is developed in

  Yao-Nan Chen and Hsuan-Tien Lin. Feature-aware label space dimension reduction for multi-label classification. In Advances in Neural Information Processing Systems: Proceedings of the 2012 Conference (NIPS), pages 1538--1546, December 2012.

* Feature-aware Implicit Label Space Encoding (FaIE) is developed in
  
  Zijia Lin, Guiguang Ding, Mingqing Hu, and Jianmin Wang. Multi-label Classification via Feature-aware Implicit Label Space Encoding. In Proceedings of the 31st International Conference on Machine Learning (ICML), June 2014.

* Column Subset Selection Problem (CSSP) is developed in

  Wei Bi and James Kwok. Efficient Multi-label Classification with Many Labels. In Proceedings of the 30th International Conference on Machine Learning (ICML), June 2013.

Please cite the these papers if you find corresponding parts of the program useful.

If there are any questions, please feel free to contact the corresponding author of the first two papers at

Hsuan-Tien Lin, htlin@csie.ntu.edu.tw

Hsuan-Tien Lin thanks his co-authors of the papers, especially Farbound Tai who contributed significantly to the initial layout of the program.

Hsuan-Tien Lin also thanks user rustle1314 on GitHub, who initiated the implementations of the FaIE and CSSP algorithms. The initial implementations are later polished by Hsuan-Tien Lin.
