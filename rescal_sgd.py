# coding: utf-8
# rescal.py - python script to compute the RESCAL tensor factorization via Stochastic Gradient descent
# Original RESCAL Scripts for Alternating Least Squared are 
# Copyright (C) 2013 Maximilian Nickel <mnick@mit.edu>
# 	
#
# Modified Version created by Bradley Baker on February 25th 2017
#	email baker@ismll.de for questions or information
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import logging
import time
import numpy as np
from numpy import dot, zeros, array, eye, kron, prod
from numpy.linalg import norm, solve, inv, svd
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import eigsh
from numpy.random import rand
import random 
__version__ = "0.4"
__all__ = ['als']

_DEF_MAXITER = 500
_DEF_INIT = 'nvecs'
_DEF_CONV = 1e-4
_DEF_LMBDA = 0
_DEF_ATTR = []
_DEF_NO_FIT = 1e9
_DEF_FIT_METHOD = None
_DEF_EXHAUST = False 
_DEF_EPOCHS = _DEF_MAXITER
_DEF_MU = 1e-5 # hyper-parameter for SGD
_log = logging.getLogger('RESCAL')

def sgd(X, rank, **kwargs):
    """
    RESCAL-SGD algorithm to compute the RESCAL tensor factorization via Stochastic Gradient Descent.
	Based of Scripts by Maximillian Nickel 
	cloned from http://github.com/mnick/rescal.py

    Parameters
    ----------
    X : list
        List of frontal slices X_k of the tensor X.
        The shape of each X_k is ('N', 'N').
        X_k's are expected to be instances of scipy.sparse.csr_matrix
    rank : int
        Rank of the factorization
    lmbdaA : float, optional
        Regularization parameter for A factor matrix. 0 by default
    lmbdaR : float, optional
        Regularization parameter for R_k factor matrices. 0 by default
    lmbdaV : float, optional
        Regularization parameter for V_l factor matrices. 0 by default
    attr : list, optional
        List of sparse ('N', 'L_l') attribute matrices. 'L_l' may be different
        for each attribute
    init : string, optional
        Initialization method of the factor matrices. 'nvecs' (default)
        initializes A based on the eigenvectors of X. 'random' initializes
        the factor matrices randomly.
    compute_fit : boolean, optional
        If true, compute the fit of the factorization compared to X.
        For large sparse tensors this should be turned of. None by default.
    maxIter : int, optional
        Maximium number of iterations of the ALS algorithm. 500 by default.
    conv : float, optional
        Stop when residual of factorization is less than conv. 1e-5 by default
	epochs : int, optional 
		Chooses a maximum number of epochs for epochal SGD 
	exhaust: boolean, optional
		Turn exhaustive SGD sampling on or off, default is on 
	mu: float, optional 
		Set the learning rate 
    Returns
    -------
    A : ndarray
        array of shape ('N', 'rank') corresponding to the factor matrix A
    R : list
        list of 'M' arrays of shape ('rank', 'rank') corresponding to the
        factor matrices R_k
    f : float
        function value of the factorization
    itr : int
        number of iterations until convergence
    exectimes : ndarray
        execution times to compute the updates in each iteration

    Examples
    --------
    >>> X1 = csr_matrix(([1,1,1], ([2,1,3], [0,2,3])), shape=(4, 4))
    >>> X2 = csr_matrix(([1,1,1,1], ([0,2,3,3], [0,1,2,3])), shape=(4, 4))
    >>> A, R, _, _, _ = rescal([X1, X2], 2)

    See
    ---
    For a full description of the algorithm see:
    .. [1] Maximilian Nickel, Volker Tresp, Hans-Peter-Kriegel,
        "A Three-Way Model for Collective Learning on Multi-Relational Data",
        ICML 2011, Bellevue, WA, USA

    .. [2] Maximilian Nickel, Volker Tresp, Hans-Peter-Kriegel,
        "Factorizing YAGO: Scalable Machine Learning for Linked Data"
        WWW 2012, Lyon, France
    """

    # ------------ init options ----------------------------------------------
    ainit = kwargs.pop('init', _DEF_INIT)
    maxIter = kwargs.pop('maxIter', _DEF_MAXITER)
    conv = kwargs.pop('conv', _DEF_CONV)
    lmbdaA = kwargs.pop('lambda_A', _DEF_LMBDA)
    lmbdaR = kwargs.pop('lambda_R', _DEF_LMBDA)
    lmbdaV = kwargs.pop('lambda_V', _DEF_LMBDA)
    compute_fit = kwargs.pop('compute_fit', _DEF_FIT_METHOD)
    P = kwargs.pop('attr', _DEF_ATTR)
    dtype = kwargs.pop('dtype', np.float)
    exhaust_sampling = kwargs.pop('exhaust',_DEF_EXHAUST)
    max_epochs = kwargs.pop('epochs',_DEF_EPOCHS)
    mu = kwargs.pop('mu',_DEF_MU)
    # ------------- check input ----------------------------------------------
    if not len(kwargs) == 0:
        raise ValueError('Unknown keywords (%s)' % (kwargs.keys()))

    # check frontal slices have same size and are matrices
    sz = X[0].shape
    for i in range(len(X)):
        if X[i].ndim != 2:
            raise ValueError('Frontal slices of X must be matrices')
        if X[i].shape != sz:
            raise ValueError('Frontal slices of X must be all of same shape')
        #if not issparse(X[i]):
            #raise ValueError('X[%d] is not a sparse matrix' % i)

    if compute_fit is None:
        if prod(X[0].shape) * len(X) > _DEF_NO_FIT:
            _log.warn('For large tensors automatic computation of fit is disabled by default\nTo compute the fit, call rescal_als with "compute_fit=True" ')
            compute_fit = False
        else:
            compute_fit = True

    n = sz[0]
    k = len(X)

    _log.debug(
        '[Config] rank: %d | maxIter: %d | conv: %7.1e | lmbda: %7.1e' %
        (rank, maxIter, conv, lmbdaA)
    )
    _log.debug('[Config] dtype: %s / %s' % (dtype, X[0].dtype))

    # ------- convert X and P to CSR ------------------------------------------
    for i in range(k):
        if issparse(X[i]):
            X[i] = X[i].tocsr()
            X[i].sort_indices()
    for i in range(len(P)):
        if issparse(P[i]):
            P[i] = P[i].tocoo().tocsr()
            P[i].sort_indices()

    # ---------- initialize A ------------------------------------------------
    _log.debug('Initializing A')
    if ainit == 'random':
        A = array(rand(n, rank), dtype=dtype)
    elif ainit == 'nvecs':
        S = csr_matrix((n, n), dtype=dtype)
        for i in range(k):
            S = S + X[i]
            S = S + X[i].T
        _, A = eigsh(csr_matrix(S, dtype=dtype, shape=(n, n)), rank)
        A = array(A, dtype=dtype)
    else:
        raise ValueError('Unknown init option ("%s")' % ainit)	
		
    # ------- initialize R and Z ---------------------------------------------
    _log.debug('Initializing R')
    R = [] #for now only random initialization
    for i in range(k):
        R.append(array(rand(rank,rank),dtype=dtype))

    #  ------ compute factorization ------------------------------------------
    fit = fitchange = fitold = f = 0
    exectimes = []
    for epochi in range(max_epochs):
		tried_indices = []
		for itr in range(maxIter):
			tic = time.time()
			fitold = fit
			n = random.randint(0,k)
			while n in tried_indices and exhaust_sampling: 
				n = random.randint(0,k)
			tried_indices.append(n)
			A = _updateA(X, n, A, R, P, lmbdaA,mu)
			R = _updateR(X, n, A, R, lmbdaR,mu)
			#Z = _updateZ(A, P, lmbdaV)

			# compute fit value
			if compute_fit:
				fit = _compute_fit(X, A, R, P, lmbdaA, lmbdaR, lmbdaV)
			else:
				fit = itr

			fitchange = abs(fitold - fit)

			toc = time.time()
			exectimes.append(toc - tic)

			_log.info('[%3d] fit: %0.5f | delta: %7.1e | secs: %.5f' % (
				itr, fit, fitchange, exectimes[-1]
			))
			if itr > 0 and fitchange < conv:
				break
    return A, R, f, itr + 1, array(exectimes)

# ------------------ Update A ------------------------------------------------
def _updateA(X,n, A, R, P, lmbdaA, mu):
    """Update step for A"""
	# n is the chosen index of the stochastic index
    _log.debug('Updating A')
    m, rank = A.shape
    F = zeros((m, rank), dtype=A.dtype)
    E = zeros((rank, rank), dtype=A.dtype)

    An = A[n].reshape((1,rank)) # I think this is the correct csr 
    AtA = dot(An.T, An) # should be a scalar

    for i in range(len(X)):
		# X in 1xM, A in Mxrank, R[i] in rank x rank,
        Xn = X[i].getrow(n)
        F += 2*Xn.T.dot(dot(An, R[i].T)) #+ (An.dot(R[i]).T).dot(Xn) # F in 1 x rank 
		# from the paper - E is the right side of the ALS update, ie. the inverse
        E += dot(R[i], dot(AtA, R[i].T)) + dot(R[i].T, dot(AtA, R[i])) # E in rank x rank 

    # regularization
    I = lmbdaA * eye(rank, dtype=A.dtype) 

    # attributes - TODO - need SGD rules for attributes?
    #for i in range(len(Z)):
    #    F += P[i].dot(Z[i].T)
    #    E += dot(Z[i], Z[i].T)

    gradientA = A.dot(E+I) - F  # gradient in M x rank 
    A = A - mu * gradientA 
    #_log.debug('Updated A lambda_A:%f, dtype:%s' % (lmbdaA, A.dtype))
    return A


# ------------------ Update R ------------------------------------------------
def _updateR(X,n,A, R,lmbdaR,mu):
    _log.debug('Updating R (SVD) lambda R: %s' % str(lmbdaR))
    rank = A.shape[1]
    An = A[n].reshape((1,rank))
    for i in range(len(R)):
        Xn = X[i].getrow(n)# the chosen stochastic instance
        Shat = An.dot(R[i]) # take the kronecker product for the stochastic gradient instance
        Shat = kron(Shat,Shat)
        gRn = (Shat.dot(Shat.T) + lmbdaR) -  (Shat * Xn.T)
        R[i] = R[i] - mu*gRn
    return R


# ------------------ Update Z ------------------------------------------------
# SHOULD NOT NEED THIS FOR SGD
'''
def _updateZ(A, P, lmbdaZ):
    Z = []
    if len(P) == 0:
        return Z
    #_log.debug('Updating Z (Norm EQ, %d)' % len(P))
    pinvAt = inv(dot(A.T, A) + lmbdaZ * eye(A.shape[1], dtype=A.dtype))
    pinvAt = dot(pinvAt, A.T).T
    for i in range(len(P)):
        if issparse(P[i]):
            Zn = P[i].tocoo().T.tocsr().dot(pinvAt).T
        else:
            Zn = dot(pinvAt.T, P[i])
        Z.append(Zn)
    return Z
'''

def _compute_fit(X, A, R, P,lmbdaA, lmbdaR, lmbdaZ):
    """Compute fit for full slices"""
    f = 0
    # precompute norms of X
    normX = [sum(M.data ** 2) for M in X]
    sumNorm = sum(normX)

    for i in range(len(X)):
        ARAt = dot(A, dot(R[i], A.T))
        f += norm(X[i] - ARAt) ** 2

    return 1 - f / sumNorm
