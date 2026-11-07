import numpy as np
from scipy.integrate import quad 
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.special import erf
from numpy.linalg import inv, norm
from numpy.linalg import solve
from math import factorial
import matplotlib.pyplot as plt
import numpy as np
from gauss_legendre import *


def eval_composite_trap(M, a, b, f):
    """Composite trapezoid rule with M equispaced points (M-1 intervals)."""
    x = np.linspace(a, b, M)
    h = (b - a) / (M - 1)
    w = h * np.ones(M)
    w[0] = w[-1] = h / 2
    I_hat = np.sum(w * f(x))
    return I_hat, x, w
 
 
def eval_composite_simpsons(M, a, b, f):
    """Composite Simpson's rule with M equispaced points.
    Simpson's needs an even number of intervals, so even M is bumped to M+1."""
    if M % 2 == 0:
        M += 1
    x = np.linspace(a, b, M)
    h = (b - a) / (M - 1)
    w = np.ones(M)
    w[1:-1:2] = 4
    w[2:-1:2] = 2
    w *= h / 3
    I_hat = np.sum(w * f(x))
    return I_hat, x, w
 
 
def eval_gauss_quad(M, a, b, f):
    """Gauss-Legendre quadrature with M nodes on [a,b]."""
    x, w = lgwt(M, a, b)
    I_hat = np.sum(f(x) * w)
    return I_hat, x, w
 
 
def adaptive_quad(a, b, f, tol, M, method):
    """
    Adaptive numerical integrator for int_a^b f(x) dx
 
    Input:
      a,b    - interval [a,b]
      f      - function to integrate
      tol    - absolute accuracy goal
      M      - number of quadrature nodes per bisected interval
      method - function handle for integrating on a subinterval
    Output:
      I      - approximate integral
      X      - final adapted grid nodes
      nsplit - number of interval splits
    """
    maxit = 50  # 1/2^50 ~ 1e-15
    left_p = np.zeros((maxit,))
    right_p = np.zeros((maxit,))
    s = np.zeros((maxit, 1))
    left_p[0] = a
    right_p[0] = b
    s[0], x, _ = method(M, a, b, f)
    X = [x]
    j = 1
    I = 0
    nsplit = 1
    while j < maxit:
        c = 0.5 * (left_p[j-1] + right_p[j-1])
        s1, x, _ = method(M, left_p[j-1], c, f); X.append(x)
        s2, x, _ = method(M, c, right_p[j-1], f); X.append(x)
        if np.max(np.abs(s1 + s2 - s[j-1])) > tol:
            left_p[j] = left_p[j-1]
            right_p[j] = c
            s[j] = s1
            left_p[j-1] = c
            s[j-1] = s2
            j = j + 1
            nsplit = nsplit + 1
        else:
            I = I + s1 + s2
            j = j - 1
            if j == 0:
                j = maxit
    return I, np.unique(np.concatenate(X)), nsplit
 
 
def driver():
    methods = [
        (eval_composite_trap,     'Trapezoid',      'bo--'),
        (eval_composite_simpsons, "Simpson's",      'gs--'),
        (eval_gauss_quad,         'Gauss-Legendre', 'r^--'),
    ]
 
    # interval of integration [a,b]
    a = 0.; b = np.pi
    # TRYME: swap functions (adjust a, b, I_true to match)
    # f = lambda x: np.log(x)**2; I_true = 2; labl = r'$\log^2(x)$'
    # f = lambda x: 1./(np.power(x,(1./5.))); I_true = 5./4.; labl = r'$1/x^{1/5}$'
    # f = lambda x: np.exp(np.cos(x)); I_true = 2.3415748417130531; labl = r'$e^{\cos x}$'
    # f = lambda x: x**20; I_true = 1./21.; labl = r'$x^{20}$'
    # below is for a=0.1, b=2
    # f = lambda x: np.sin(1./x); I_true = 1.1455808341; labl = r'$\sin(1/x)$'
    # f = lambda x: np.exp(x) * np.cos(x); I_true = -12.070346316389632; labl = r'$e^x \cos x$'

 
    tol = 1e-10                    # absolute tolerance for adaptive quad
    eps = np.finfo(float).eps
    Ms = np.arange(2, 15); nM = len(Ms)
 
    results = {}
    for method, name, style in methods:
        err_old = np.zeros(nM)
        err_new = np.zeros(nM)
        nsplits = np.zeros(nM, dtype=int)
        mesh = None
        for iM, M in enumerate(Ms):
            I_old, _, _ = method(M, a, b, f)
            I_new, X, nsplit = adaptive_quad(a, b, f, tol, M, method)
            err_old[iM] = max(np.abs(I_old - I_true) / np.abs(I_true), eps)
            err_new[iM] = max(np.abs(np.squeeze(I_new) - I_true) / np.abs(I_true), eps)
            nsplits[iM] = nsplit
            if M == 2:
                mesh = X
        results[name] = (err_old, err_new, nsplits, mesh, style)
        print(f'{name:15s} nsplit per M: {nsplits}')
 
    # error plots: all methods on the same axes
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for name, (err_old, err_new, _, _, style) in results.items():
        ax[0].semilogy(Ms, err_old, style, label=name)
        ax[1].semilogy(Ms, err_new, style, label=name)
    for axi, title in zip(ax, ['Non-adaptive', f'Adaptive (tol = {tol:g})']):
        axi.set_ylim([1e-16, 2])
        axi.set_xlabel('$M$')
        axi.set_title(title)
        axi.grid(True, which='both', alpha=0.3)
        axi.legend()
    ax[0].set_ylabel('Relative Error')
    fig.suptitle(f'Quadrature error for {labl} on [{a:g}, {b:.4g}]')
    fig.tight_layout()
    plt.show()
 
    # adaptive meshes for M = 2, one panel per method
    xx = np.linspace(a, b, 500)
    fig, ax = plt.subplots(len(results), 1, figsize=(8, 7), sharex=True)
    for axi, (name, (_, _, nsplits, mesh, style)) in zip(ax, results.items()):
        axi.plot(xx, f(xx), 'k-', lw=1, alpha=0.5, label=labl)
        axi.plot(mesh, f(mesh), style[:2], ms=4, label=f'{name} nodes')
        axi.set_title(f'{name}: adaptive mesh, M = 2 ({nsplits[0]} splits, {len(mesh)} nodes)')
        axi.legend(loc='lower left')
    ax[-1].set_xlabel('$x$')
    fig.tight_layout()
    plt.show()


def hw10_problem1():
    f = lambda x: np.exp(x) * np.cos(x)
    a, b = 0., np.pi
    I_true = -12.070346316389632
    tol = 1e-10

    data = {}
    for method, name, n0 in [(eval_composite_trap, 'Trapezoid', 1),
                             (eval_composite_simpsons, "Simpson's", 2)]:
        n = n0
        hs, errs = [], []
        print(f'\n{name}\n{"n":>8} {"h":>11} {"I_n":>20} {"rel err":>10}')
        while True:
            I, _, _ = method(n + 1, a, b, f)          # n subintervals = n+1 points
            h = (b - a) / n
            err = abs(I - I_true)
            hs.append(h); errs.append(err)
            print(f'{n:8d} {h:11.4e} {I:20.14f} {err/abs(I_true):10.3e}')
            if err / abs(I_true) < tol:
                break
            n *= 2                                    # halve h
        print(f'-> {name} needs n = {n} (h = {h:.3e})')
        data[name] = (np.array(hs), np.array(errs))

    # Part (b): log-log error vs h
    for (name, (hs, errs)), style in zip(data.items(), ['bo-', 'gs-']):
        p = np.polyfit(np.log(hs[2:]), np.log(errs[2:]), 1)   # skip coarsest h
        plt.loglog(hs, errs, style, label=f'{name} (slope ≈ {p[0]:.2f})')
    plt.xlabel('$h$'); plt.ylabel('Absolute error')
    plt.title(r'$\int_0^\pi e^x\cos x\,dx$')
    plt.grid(True, which='both', alpha=0.3); plt.legend()
    plt.show()

    f = lambda s: 1.0 / (1.0 + s**2)
    a, b = -5.0, 5.0
    I_true = 2 * np.arctan(5)  

    n_T, n_S = 1291, 108
    T, c1, d1 = eval_composite_trap(n_T, a, b, f)
    S, c2, d2 = eval_composite_simpsons(n_S, a, b, f)
    
    # ---------- SciPy quad ----------
    # tolerance 1e-6 (what the homework calls the default) and 1e-4
    Q6, err6, info6 = quad(f, a, b, epsabs=1e-6, epsrel=1e-6, full_output=1)
    Q4, err4, info4 = quad(f, a, b, epsabs=1e-4, epsrel=1e-4, full_output=1)
    # SciPy's actual default is epsabs = epsrel = 1.49e-8
    Qd, errd, infod = quad(f, a, b, full_output=1)
    
    print(f'Exact: 2*arctan(5) = {I_true:.15f}\n')
    print(f'{"Method":<26}{"Value":>20}{"|error|":>12}{"f evals":>10}')
    rows = [
        (f'Trapezoid  (n = {n_T})', T, n_T + 1),
        (f"Simpson's  (n = {n_S})",  S, n_S + 1),
        ('quad, tol = 1e-6',          Q6, info6['neval']),
        ('quad, tol = 1e-4',          Q4, info4['neval']),
        ('quad, SciPy default',       Qd, infod['neval']),
    ]
    for name, val, nev in rows:
        print(f'{name:<26}{val:20.15f}{abs(val - I_true):12.3e}{nev:10d}')





if __name__ == '__main__':
    hw10_problem1()
    # driver()      # Lab 11 adaptive comparison
    
 
