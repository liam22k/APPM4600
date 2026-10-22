import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad

def driver():
    f = lambda x: math.exp(x)
    a, b = -1, 1
    w = lambda x: 1.
    n = 2
    N = 1000
    xeval = np.linspace(a, b, N+1)

    pval = np.array([eval_legendre_expansion(f, a, b, w, n, x) for x in xeval])
    fex  = np.array([f(x) for x in xeval])

    plt.figure()
    plt.plot(xeval, fex, 'ro-', label='f(x)')
    plt.plot(xeval, pval, 'bs--', label='Expansion')
    plt.legend()
    plt.show()

    err = abs(pval - fex)
    plt.semilogy(xeval, err, 'ro--', label='error')
    plt.legend()
    plt.show()

def eval_legendre_expansion(f, a, b, w, n, x):
    p = eval_legendre(x, n)
    pval = 0.0
    for j in range(n+1):
        phi_j     = lambda t: eval_legendre(t, j)[j]
        phi_j_sq  = lambda t: phi_j(t)**2 * w(t)
        norm_fac  = quad(phi_j_sq, a, b)[0]
        func_j    = lambda t: phi_j(t) * f(t) * w(t)
        aj        = quad(func_j, a, b)[0] / norm_fac
        pval     += aj * p[j]
    return pval

def eval_legendre(x, n):
    p = np.zeros(n+1)
    p[0] = 1.0
    if n >= 1:
        p[1] = x
        for k in range(1, n):
            p[k+1] = ((2*k+1)*x*p[k] - k*p[k-1]) / (k+1)
    return p

driver()