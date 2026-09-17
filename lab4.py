import numpy as np
from scipy.integrate import quad 
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.special import erf

def driver():

    # test function from Exercise 6: f(x) = e^(x^2+7x-30) - 1, root at x = 3
    f   = lambda x: np.exp(x**2 + 7*x - 30) - 1
    fp  = lambda x: (2*x + 7)*np.exp(x**2 + 7*x - 30)
    fpp = lambda x: np.exp(x**2 + 7*x - 30)*(2 + (2*x + 7)**2)


    tol  = 1e-11
    Nmax = 100

    a = 2
    b = 4.5

    # (a) bisection alone, [a,b] = [2, 4.5]
    [astar_b, ier_b, nbisect, nfeval_b] = bisection(f, a, b, tol)
    print(' (a) Bisection ')
    print('root:', astar_b, ' ier:', ier_b)
    print('iterations:', nbisect, ' f evals:', nfeval_b)
    print()

    # (b) Newton alone, p0 = 4.5
    p0 = 4.5
    [p, pstar_n, ier_n, nnewton, nfeval_n, nfpeval_n] = newton(f, fp, p0, tol, Nmax)
    print(' (b) Newton ')
    print('root:', pstar_n, ' ier:', ier_n)
    print('iterations:', nnewton, ' f evals:', nfeval_n, ' fp evals:', nfpeval_n)
    print()

    # (c) hybrid method, [a,b] = [2, 4.5]
    [pstar_h, ier_h, nbisect_h, nnewton_h, nfeval_h, nfpeval_h, nfppeval_h] = \
        hybrid_newton(f, fp, fpp, a, b, tol, Nmax)
    print(' (c) Hybrid (bisection + Newton) ')
    print('root:', pstar_h, ' ier:', ier_h)
    print('bisection iterations:', nbisect_h, ' Newton iterations:', nnewton_h)
    print('f evals:', nfeval_h, ' fp evals:', nfpeval_h, ' fpp evals:', nfppeval_h)




def bisection(f, a, b, tol):

    nfeval = 0
    fa = f(a); nfeval += 1
    fb = f(b); nfeval += 1
    if fa*fb > 0:
        return [a, 1, 0, nfeval]

    if fa == 0:
        return [a, 0, 0, nfeval]
    if fb == 0:
        return [b, 0, 0, nfeval]

    count = 0
    d = 0.5*(a+b)
    while abs(d-a) > tol:
        fd = f(d); nfeval += 1
        if fd == 0:
            return [d, 0, count, nfeval]
        if fa*fd < 0:
            b = d
        else:
            a = d
            fa = fd
        d = 0.5*(a+b)
        count += 1

    return [d, 0, count, nfeval]


def newton(f, fp, p0, tol, Nmax):

    p = np.zeros(Nmax+1)
    p[0] = p0
    nfeval = 0
    nfpeval = 0

    for it in range(Nmax):
        fval  = f(p0);  nfeval  += 1
        fpval = fp(p0); nfpeval += 1
        p1 = p0 - fval/fpval
        p[it+1] = p1
        if abs(p1-p0) < tol:
            return [p, p1, 0, it+1, nfeval, nfpeval]
        p0 = p1

    return [p, p1, 1, Nmax, nfeval, nfpeval]


def hybrid_newton(f, fp, fpp, a, b, tol, Nmax):

    nfeval = 0
    nfpeval = 0
    nfppeval = 0

    fa = f(a); nfeval += 1
    fb = f(b); nfeval += 1
    if fa*fb > 0:
        return [a, 1, 0, 0, nfeval, nfpeval, nfppeval]

    def in_basin(x):
        nonlocal nfeval, nfpeval, nfppeval
        fx = f(x);     nfeval += 1
        fpx = fp(x);   nfpeval += 1
        fppx = fpp(x); nfppeval += 1
        if fpx == 0:
            return False
        return abs(fx*fppx/fpx**2) < 1

    # Exercise 2: bisect until the midpoint is in the basin of convergence
    d = 0.5*(a+b)
    nbisect = 0
    while not in_basin(d):
        fd = f(d); nfeval += 1
        if fa*fd < 0:
            b = d
        else:
            a = d
            fa = fd
        d = 0.5*(a+b)
        nbisect += 1
        if nbisect > Nmax:
            return [d, 1, nbisect, 0, nfeval, nfpeval, nfppeval]

    # Exercise 4: hand the bisection midpoint to Newton as its starting guess
    p0 = d
    print(p0)
    for it in range(Nmax):
        fval  = f(p0);  nfeval  += 1
        fpval = fp(p0); nfpeval += 1
        p1 = p0 - fval/fpval
        if abs(p1-p0) < tol:
            return [p1, 0, nbisect, it+1, nfeval, nfpeval, nfppeval]
        p0 = p1

    return [p0, 1, nbisect, Nmax, nfeval, nfpeval, nfppeval]


driver()