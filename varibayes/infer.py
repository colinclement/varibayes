"""
infer.py

author: Colin Clement
date: 2017-10-18

This module implements  black-box variational inference algorithm as
described in http://proceedings.mlr.press/v33/ranganath14.html. 

usage:
    # function signature loglikelihood(params, data)
    vb = VariationalInferenceMF(loglikelihood, args=(data,))
    vb.fit()

"""

import numpy as np
import varibayes.opt.adadelta as adadelta


class VariationalInferenceMF(object):
    """ Mean-field variational inference module

    Given a log-likelihood log p(x,z) with observations x and parameters z,
    this module finds a variational distribution q(z|lambda) over the parameters
    z which minimizes the Kullback-Leibler divergence with p and q.
    Mean-field means that q({z_i}|lambda) = \prod_i N(\mu_i, \sigma_i^2), with 
    a mean and standard deviation for each parameter z.
    """
    def __init__(self, loglikelihood, args=(), samples=12, **kwargs):
        """ Create a Variational Inference module

        Args:
            loglikelihood (function): with arguments (params, *args).

        Kwargs:
            args (tuple): of extra arguments to be given to loglikelihood
            samples (int): number of samples for gradient estimation
            kwargs are handed to the optimization routine initialization
        """
        self.loglikelihood = loglikelihood
        self.args = args
        self.params = []
        self.samples = samples
        self.opt = adadelta.Adadelta(self.evd_grad_evd_rao_blackwell, **kwargs)

    @property
    def mus(self):
        """ Means of the variational distribution """
        return self.params[:len(self.params)//2]

    @property
    def sigmas(self):
        """ Standard deviations of the variational distribution """
        return self.params[len(self.params)//2:]
    
    def sampledistn(self, n=None, params=None):
        """ 
        Draw n samples from variational distribution with params 
        Args:
            (optional)
            n (int): number of samples to draw
            params (array_like): parameters of distribution
        Returns:
            samples (array_like): shape (n or self.samples, len(params))
        """
        params = params if params is not None else self.params
        mus, sigmas = params[:len(params)//2], params[len(params)//2:]
        return np.random.normal(mus, np.abs(sigmas), 
                                size=(n or self.samples, len(mus)))

    def logdistn(self, params=None, zs=None, n=None):
        """ 
        Compute log q(z|params) with specific zs or n samples 
        Args:
            (optional)
            params (array_like): parameters of distribution
            zs (array_like): set of samples of trial distribution
            n (int): number of samples to draw
        Returns:
            log q(z|params) (array_like): one for each sample
        """
        zs = zs if zs is not None else self.sampledistn(n, params)
        params = params if params is not None else self.params
        m, s = params[:len(params)//2], params[len(params)//2:]
        r = (zs - m)/s
        return - np.sum(r * r + np.log(2 * np.pi * s * s), axis=1)/2.

    def gradlogdistn(self, params=None, zs=None, n=None):
        """ 
        grad_params log q(z|params) with specific zs or n samples 
        Args:
            (optional)
            params (array_like): parameters of distribution
            zs (array_like): set of samples of trial distribution
            n (int): number of samples to draw
        Returns:
            gradient of params for each sample drawn
            gradlog_distn (array_like): shape (n or len(zs), len(params)) 
        """
        zs = zs if zs is not None else self.sampledistn(n, params)
        params = params if params is not None else self.params
        mus, sigmas = params[:len(params)//2], params[len(params)//2:]
        r = (zs - mus)/sigmas
        return np.concatenate([r/sigmas, (r * r - 1)/sigmas], 1)

    def evidence(self, params=None, zs=None, n=None):
        """ 
        Compute terms of the objective with params, zs or n samples 
            evidence = D_KL(q(z|params)||p(x,z)) is a lower bound on
            p(x), the evidence. The actual value of this at the best-fit
            allows model selection.
        Args:
            (optional)
            params (array_like): parameters of distribution
            zs (array_like): set of samples of trial distribution
            n (int): number of samples to draw
        Returns:
            evidence terms (array_like): shape (n or len(zs))
        """
        zs = zs if zs is not None else self.sampledistn(n, params)
        logl = np.array([self.loglikelihood(z, *self.args) for z in zs])
        return logl - self.logdistn(params, zs)
    
    def gradevidence(self, params=None, zs=None, n=None):
        """
        Compute gradient of evidence objective function
        Args:
            (optional)
            params (array_like): parameters of distribution
            zs (array_like): set of samples of trial distribution
            n (int): number of samples to draw
        Returns:
            grad_evd (array_like): shape (n or len(zs), len(params))
        """
        zs = zs if zs is not None else self.sampledistn(n, params)
        evd = self.evidence(params, zs, n)
        grad = self.gradlogdistn(params, zs, n)
        return np.mean(grad * evd[:,None], axis=0)

    def evd_grad_evd(self, params, n=None):
        """
        Function for handing the evidence and the gradient of the evidence to
        an optimization routine.
        Args:
            (required)
            params (array_like): parameters of distribution
            (optional)
            n (int): number of samples for gradient estimation
        Returns:
            evidence (float), -grad_evidence (array_like)
        """
        zs = self.sampledistn(n, params)
        evd = self.evidence(params, zs, n)
        grad = self.gradlogdistn(params, zs, n)
        return np.mean(evd), - np.mean(grad * evd[:,None], axis=0) 

    def evd_grad_evd_rao_blackwell(self, params, n=None):
        """
        Function for handing the evidence and the gradient of the evidence to
        an optimization routine.
        This method applies a variate control scheme and the Rao-Blackwell
        theorem to dramatically reduce the variance of the gradient estimator
        and dramatically improve the optimization.
        Args:
            (required)
            params (array_like): parameters of distribution
            (optional)
            n (int): number of samples for gradient estimation
        Returns:
            evidence (float), -grad_evidence (array_like)
        """

        zs = self.sampledistn(n, params)
        evd = self.evidence(params, zs, n)
        grad = self.gradlogdistn(params, zs, n)

        grad_evd = grad * evd[:,None]
        mean_grad_evd = grad_evd.mean(0)
        mean_grad = grad.mean(0)
        a = np.mean((grad_evd-mean_grad_evd)*(grad-mean_grad),0)/grad.std(0)**2

        return np.mean(evd), - (mean_grad_evd - a * mean_grad)

    def fit(self, p0, **kwargs):
        """
        Find a variational distribution by optimizing the evidence using
        Adadelta
        Args:
            p0 (array_like): initial parameters
        Kwargs are handed to optimizer.optimize
        """
        self.params = self.opt.optimize(p0.copy(), **kwargs)
