testrembed.m
==========
This file exhibits the basic guarantee of the randomized embedding code.

Given features X and labels Y, where the SVD of X is given by 

> X = U<sub>X</sub> &Sigma;<sub>X</sub> V<sub>X</sub> 

and the SVD of (U<sub>X</sub><sup>T</sup> Y) is 

> U<sub>X</sub><sup>T</sup> Y = U<sub>E</sub> &Sigma;<sub>E</sub> V<sub>E</sub>, 
 
the k-dimensional embedding is defined as the first k columns of V<sub>E</sub>.  This definition is motivated by the optimal rank-constrained least-squares approximation of Y given X, as explained in [this paper](http://arxiv.org/abs/1412.6547).

Randomized methods provide a fast way of approximating these SVDs when the dimensionalities are large.
