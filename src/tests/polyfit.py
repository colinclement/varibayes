"""
polyfit.py

author: Colin Clement
date: 2017-10-18

Minimal working example of using VariationalInference to fit polynomials
"""

import sys
sys.path.append('../')
import numpy as np
import numexpr as nu
import varibayes

x = np.linspace(-1,1,100)

def loglikelihood(params, data, x=x):
    sigma = p[0]
    r = (np.polyval(params[1:], x) - data)/sigma
    return - np.sum(r*r + np.log(2*np.pi*sigma*sigma))/2.
    
def res(params, data, x=x):
    mod = np.polyval(params, x)
    return mod - data

if __name__=='__main__':

    # Make some interesting data
    zz = [-1, -0.4, 0, 0.4, 1.]
    pp = np.poly(zz)
    y = np.polyval(pp, x)
    ptp = y.ptp()
    y /= ptp

    sigma = 0.01  # we divide y by its ptp so sigma is 1/SNR
    p = np.hstack([sigma, pp])
    data = y + sigma*np.random.randn(len(y))

    vb = varibayes.VariationalInferenceMF(loglikelihood, args=(data,),
                                          samples=100)

    p0 = np.hstack([sigma, pp/ptp + 0.5*np.random.randn(len(pp)),
                    10*np.random.rand(len(p))])
    vb.fit(p0.copy(), iprint=100, tol=1E-10, itn=5000)
