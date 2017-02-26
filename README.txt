RESCAL-SGD v 0.0.1

This is an implementation of Nickel 2011 RESCAL using Stochastic Gradient Descent 
rather than the vanilla Alternating Least Squares. 
The aim is to avoid the runtime issues of vanilla RESCAL by replacing the large tensor inversion in 
RESCAL's ALS with a gradient descent-type algorithm, using the gradients derived in the original paper.

References:
Maximilian Nickel, Volker Tresp, Hans-Peter-Kriegel,
"A Three-Way Model for Collective Learning on Multi-Relational Data",
Proceedings of the 28th International Conference on Machine Learning (ICML'11), 
809--816, ACM, Bellevue, WA, USA, 2011 


CURRENTLY:
Rescal with non-stochastic gradient-descent performs comparably to rescal als, converging in an average of 24 iterations 
without using inverses or kronecker products, thus significantly reducing complexity without requiring the removal of regularization. 
Note that I have altered the loss function to reflect that in the above paper.  

Stochastic Gradient descent converges with no regularization and very small learning rate; however, it takes 1000+ iterations to 
converge, and is thus not yet very feasible. Batch-SGD, more rigorous hyper-parameter searching might, and further debugging 
might help improve convergence. 

TODO:
1. Implement smarter initialization for R -feature-relation- matrices (Currently initialized as zeros)
2. Conglomerate SGD and GD duplicate code into separate functions. 
3. Implement rigorous hyper-parameter testing
4. Implement BATCH-SGD for RESCAL
5. Implement attribute-inclusion for RESCAL
6. Publish comparisons for implemenations on KINSHIP data here to GITHUB