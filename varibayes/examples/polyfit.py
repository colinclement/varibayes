"""
polyfit.py

author: Colin Clement
date: 2017-10-18

Minimal working example of using VariationalInference to fit polynomials
Note:
    Unless you know the noise of your data, you should allow it to also be a
    parameter of your model!
"""

import numpy as np
import varibayes.infer as infer

x = np.linspace(-1,1,100)
rng = np.random.RandomState(92089)

def loglikelihood(params, data, x=x):
    sigma = p[0]
    r = (np.polyval(params[1:], x) - data)/sigma
    return - np.sum(r * r + np.log(2 * np.pi * sigma**2))/2.
    
def res(params, data, x=x):
    mod = np.polyval(params, x)
    return mod - data

if __name__=='__main__':
    import matplotlib.pyplot as plt
    # Make some interesting data
    zz = [-1, -0.4, 0, 0.4, 1.]
    pp = np.poly(zz)
    y = np.polyval(pp, x)
    ptp = y.ptp()
    y /= ptp

    sigma = 0.05  # we divide y by its ptp so sigma is 1/SNR
    p = np.hstack([sigma, pp])
    data = y + sigma*rng.randn(len(y))

    vb = infer.VariationalInferenceMF(loglikelihood, args=(data,),
                                      samples=30)

    # set random initial conditions
    p0 = np.hstack([sigma, 0.1*rng.randn(len(pp)),
                    0.1*rng.rand(len(p))])

    vb.fit(p0.copy(), iprint=500, tol=5E-8, itn=5000)

    # Check to see that optimization converged to maximum evidence
    plt.plot(vb.opt.obj_list, label='Adagrad')

    # Compare the fit to the data
    plt.figure()
    plt.plot(x, data, '-o', label='Data')
    # Also show sample fits from the distribution!
    for z in vb.sampledistn(1000):
        plt.plot(x, np.polyval(z[1:len(pp)+1], x), lw=0.1, alpha=0.05, c='k')
    plt.plot(x, np.polyval(vb.mus[1:], x), label='Average model', lw=1, c='b')
    plt.show()
