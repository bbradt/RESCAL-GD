# coding: utf-8
# rescal.py - python script to compute the RESCAL tensor factorization via Stochastic Gradient descent and Gradient Descent
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.	If not, see <http://www.gnu.org/licenses/>.

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
_DEF_EPOCHS = 10
_DEF_MU = 5e-4 # hyper-parameter for SGD
_DEF_DIV = 1e10
_log = logging.getLogger('RESCAL')
_DEF_ANNEAL = 0.8

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
	div = kwargs.pop('div',_DEF_DIV)
	lmbdaA = kwargs.pop('lambda_A', _DEF_LMBDA)
	lmbdaR = kwargs.pop('lambda_R', _DEF_LMBDA)
	lmbdaV = kwargs.pop('lambda_V', _DEF_LMBDA)
	compute_fit = kwargs.pop('compute_fit', _DEF_FIT_METHOD)
	P = kwargs.pop('attr', _DEF_ATTR)
	dtype = kwargs.pop('dtype', np.float)
	exhaust_sampling = kwargs.pop('exhaust',_DEF_EXHAUST)
	max_epochs = kwargs.pop('epochs',_DEF_EPOCHS)
	mu = kwargs.pop('mu',_DEF_MU)
	anneal = kwargs.pop('anneal',_DEF_ANNEAL)
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

	m = sz[0]
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
		A = array(rand(m, rank), dtype=dtype)
		A0 = array(rand(m, rank), dtype=dtype)
	elif ainit == 'nvecs':
		S = csr_matrix((m, m), dtype=dtype)
		for i in range(k):
			S = S + X[i]
			S = S + X[i].T
		_, A = eigsh(csr_matrix(S, dtype=dtype, shape=(m, m)), rank)
		A = array(A, dtype=dtype)
		A0 = array(A, dtype=dtype)
	else:
		raise ValueError('Unknown init option ("%s")' % ainit)	
		
	# ------- initialize R and Z ---------------------------------------------
	_log.debug('Initializing R')
	R = [] #for now only random initialization
	R0 = []
	for i in range(k):
		R.append(np.zeros((rank,rank),dtype=dtype))
		R0.append(np.zeros((rank,rank),dtype=dtype))
	#  ------ compute factorization ------------------------------------------
	fit = fitchange = fitold = f = 0
	exectimes = []
	diverged = False
	epochi = 0
	itr = 0
	while epochi < max_epochs:
		tried_indices = []
		while itr < maxIter:
			tic = time.time()
			fitold = fit
			n = random.randint(0,m-1)
			while n in tried_indices and exhaust_sampling: 
				n = random.randint(0,m-1)
			tried_indices.append(n)
				
			A = _sgdUpdateA(X, n, A, R, P, lmbdaA,mu)
			R = _sgdUpdateR(X, n, A, R, lmbdaR,mu)
			#Z = _updateZ(A, P, lmbdaV)

			# compute fit value
			if compute_fit:
				fit = _compute_fit(X, A, R, P, lmbdaA, lmbdaR)
			else:
				fit = itr
			#print(fit)

			fitchange = abs(fitold - fit)
			toc = time.time()
			exectimes.append(toc - tic)
			mu *= anneal
			_log.info('[%3d] fit: %0.5f | delta: %7.1e | secs: %.5f' % (
				itr, fit, fitchange, exectimes[-1]
			))
			if len(tried_indices) == m and exhaust_sampling:
				break
			if itr > 0 and fitchange < conv:
				break
			itr += 1	
		if fitchange < conv and fit != 0:
				break		
		epochi += 1		
		itr = 0
		if fitchange < conv and epochi > 0:
			return A, R, f, itr + 1, array(exectimes)
	return A, R, f, itr + 1, array(exectimes)
#Regular Non-Stochastic Gradient Descent
def gd(X, rank, **kwargs):
	"""
	RESCAL-GD algorithm to compute the RESCAL tensor factorization via Gradient Descent.
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
	div = kwargs.pop('div',_DEF_DIV)
	lmbdaA = kwargs.pop('lambda_A', _DEF_LMBDA)
	lmbdaR = kwargs.pop('lambda_R', _DEF_LMBDA)
	lmbdaV = kwargs.pop('lambda_V', _DEF_LMBDA)
	compute_fit = kwargs.pop('compute_fit', _DEF_FIT_METHOD)
	P = kwargs.pop('attr', _DEF_ATTR)
	dtype = kwargs.pop('dtype', np.float)
	mu = kwargs.pop('mu',_DEF_MU)
	anneal = kwargs.pop('anneal',_DEF_ANNEAL)
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

	m = sz[0]
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
		A = array(rand(m, rank), dtype=dtype)
		A0 = array(rand(m, rank), dtype=dtype)
	elif ainit == 'nvecs':
		S = csr_matrix((m, m), dtype=dtype)
		for i in range(k):
			S = S + X[i]
			S = S + X[i].T
		_, A = eigsh(csr_matrix(S, dtype=dtype, shape=(m, m)), rank)
		A = array(A, dtype=dtype)
		A0 = array(A, dtype=dtype)
	else:
		raise ValueError('Unknown init option ("%s")' % ainit)	
		
	# ------- initialize R and Z ---------------------------------------------
	_log.debug('Initializing R')
	R = [] #for now only random initialization
	R0 = []
	for i in range(k):
		R.append(np.zeros((rank,rank),dtype=dtype))
		R0.append(np.zeros((rank,rank),dtype=dtype))
	#  ------ compute factorization ------------------------------------------
	fit = fitchange = fitold = f = 0
	exectimes = []
	diverged = False
	epochi = 0
	itr = 0
	for itr in range(maxIter):
		tic = time.time()
		fitold = fit
			
		A = _gdUpdateA(X, A, R, P, lmbdaA,mu)
		R = _gdUpdateR(X, A, R, lmbdaR,mu)
		#Z = _updateZ(A, P, lmbdaV)

		# compute fit value
		if compute_fit:
			fit = _compute_fit(X, A, R, P, lmbdaA, lmbdaR)
		else:
			fit = itr
		fitchange = abs(fitold - fit)
		toc = time.time()
		exectimes.append(toc - tic)
		mu *= anneal
		_log.info('[%3d] fit: %0.5f | delta: %7.1e | secs: %.5f' % (
			itr, fit, fitchange, exectimes[-1]
		))
		if itr > 0 and fitchange < conv:
			break
	if fitchange < conv and itr > 0:
		return A, R, f, itr + 1, array(exectimes)	
	
# ------------------ Stochastic Gradient Update A ------------------------------------------------
def _sgdUpdateA(X,n, A, R, P, lmbdaA, mu):
	"""Update step for A"""
	# n is the chosen index of the stochastic index
	_log.debug('Updating A')
	m, rank = A.shape
	gradientA = zeros((1, rank), dtype=A.dtype)

	An = A[n].reshape((1,rank)) # The stochastic element	
	for i in range(len(X)):
		# X in 1xM, A in Mxrank, R[i] in rank x rank,
		Xn = X[i].getcol(n)
		Loss = Xn - A.dot(dot(R[i],An.T))
		chRule = - ((R[i].T).dot((A.T).dot(Loss))).T
		gradientA += chRule 
	gradientA += lmbdaA
	# attributes - TODO - need SGD rules for attributes?
	#for i in range(len(Z)):
	#	 F += P[i].dot(Z[i].T)
	#	 E += dot(Z[i], Z[i].T)

	for i in range(m):
		A[n] = A[n] - mu * gradientA
	#_log.debug('Updated A lambda_A:%f, dtype:%s' % (lmbdaA, A.dtype))
	return A

# ------------------ Stochastic Gradient Update R ------------------------------------------------
def _sgdUpdateR(X,n,A, R,lmbdaR,mu):
	"""Stochastic-Gradient Update step for R"""
	_log.debug('Updating R (SVD) lambda R: %s' % str(lmbdaR))
	rank = A.shape[1]
	An = A[n].reshape((1,rank))
	for i in range(len(R)):
		Xn = X[i].getcol(n)# the chosen stochastic instance
		Loss = Xn - A.dot(dot(R[i],An.T))
		gradientRn = -((A.T).dot(Loss)).dot(An) + lmbdaR
		R[i] = R[i] - mu*gradientRn
	return R	
	
# ------------------ Non-Stochastic Gradient Update A ------------------------------------------------	
def _gdUpdateA(X,A, R, P, lmbdaA, mu):
	"""Full-Gradient Update step for A"""
	_log.debug('Updating A')
	m, rank = A.shape
	gradientA = zeros((m, rank), dtype=A.dtype)
	
	for i in range(len(X)):
		Loss = X[i] - A.dot(dot(R[i],A.T))
		chRule = - ((R[i].T).dot((A.T).dot(Loss))).T #chain rule term 
		gradientA += chRule 
	gradientA += lmbdaA

	A = A - mu * gradientA
	#_log.debug('Updated A lambda_A:%f, dtype:%s' % (lmbdaA, A.dtype))
	return A	

# ------------------ Full Gradient Update R ------------------------------------------------
def _gdUpdateR(X,A, R,lmbdaR,mu):
	"""Full-Gradient Update step for R"""
	_log.debug('Updating R (SVD) lambda R: %s' % str(lmbdaR))
	for i in range(len(R)):
		Loss = X[i] - A.dot(dot(R[i],A.T))
		gradientRn = -((A.T).dot(Loss)).dot(A) + lmbdaR
		R[i] = R[i] - mu*gradientRn
	return R
	

def _compute_fit(X, A, R, P,lmbdaA, lmbdaR):
	"""Compute fit for full slices"""
	f = 0
	# precompute norms of X
	normX = [sum(M.data ** 2) for M in X]
	sumNorm = sum(normX)

	for i in range(len(X)):
		ARAt = dot(A, dot(R[i], A.T))
		f += norm(X[i] - ARAt) ** 2
		f += lmbdaR *norm(R[i]) #NOTE THE ADDITION OF REGULARIZATION TERMS, WHICH WERE NOT INCLUDED IN ORIGINAL CODE
	f += lmbdaA*norm(A)
	return (1-f/sumNorm)
