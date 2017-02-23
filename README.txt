RESCAL-SGD v 0.0.1

This is an implementation of Nickel 2011 RESCAL using Stochastic Gradient Descent rather than the vanilla Alternating Least Squares. The aim is to avoid the runtime issues of vanilla RESCAL by replacing the large tensor inversion in RESCAL's ALS with a gradient descent-type algorithm, using the gradients derived in the original paper. 
