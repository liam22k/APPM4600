import numpy as np
from scipy.integrate import quad 
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.special import erf
from numpy.linalg import inv, norm


def driver():

    # problem 1
    xfine = np.linspace(-1, 1, 1001)

    # Track max |p| as N grows to find where it hits ~100
    print(" N   max|p(x)|")
    for N in range(2, 22):
        x, y, c = solve_coeffs(N)
        print(f"{N:2d}   {np.max(np.abs(eval_poly(c, xfine))):.4e}")

    # Plot a few representative N values
    for N in [3, 5, 9, 13, 17, 19]:
        x, y, c = solve_coeffs(N)
        p = eval_poly(c, xfine)

        plt.figure()
        plt.plot(xfine, f(xfine), 'k-', label='f(x)')
        plt.plot(xfine, p, 'b--', label=f'p(x), N={N}')
        plt.plot(x, y, 'ro', label='data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Monomial interpolation, N = {N}')
        plt.legend()
        plt.grid(True)

    plt.show()

    # problem 2

    for N in [10, 20, 40, 80]:
        x = make_nodes(N)
        y = f(x)
        w = bary_weights(x)
        p = eval_bary1(x, y, w, xfine)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

        # Left: full interval, clipped so the center stays readable
        ax1.plot(xfine, f(xfine), 'k-', label='f(x)')
        ax1.plot(xfine, p, 'b--', label='p(x)')
        ax1.plot(x, y, 'ro', ms=4, label='data')
        ax1.set_ylim(-1, 2)
        ax1.set_title(f'Barycentric (1st form), N = {N}')
        ax1.set_xlabel('x'); ax1.legend(); ax1.grid(True)

        # Right: pointwise error on a log scale
        ax2.semilogy(xfine, np.abs(p - f(xfine)) + 1e-17)
        ax2.set_title(f'|p(x) - f(x)|, N = {N}')
        ax2.set_xlabel('x'); ax2.grid(True)

        print(f"N = {N:3d}   max|p| = {np.max(np.abs(p)):.3e}   "
              f"max err on |x|<0.3 = {np.max(np.abs(p - f(xfine))[np.abs(xfine) < 0.3]):.3e}")

    plt.show()

    # problem 3
    for N in [10, 20, 40, 80]:
        x = cheb_nodes(N)
        y = f(x)
        w = bary_weights(x)
        p = eval_bary1(x, y, w, xfine)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

        # Left: full interval, clipped so the center stays readable
        ax1.plot(xfine, f(xfine), 'k-', label='f(x)')
        ax1.plot(xfine, p, 'b--', label='p(x)')
        ax1.plot(x, y, 'ro', ms=4, label='data')
        ax1.set_ylim(-1, 2)
        ax1.set_title(f'Barycentric (1st form), N = {N}')
        ax1.set_xlabel('x'); ax1.legend(); ax1.grid(True)

        # Right: pointwise error on a log scale
        ax2.semilogy(xfine, np.abs(p - f(xfine)) + 1e-17)
        ax2.set_title(f'|p(x) - f(x)|, N = {N}')
        ax2.set_xlabel('x'); ax2.grid(True)

        print(f"N = {N:3d}   max|p| = {np.max(np.abs(p)):.3e}   "
              f"max err on |x|<0.3 = {np.max(np.abs(p - f(xfine))[np.abs(xfine) < 0.3]):.3e}")

    plt.show()

def make_nodes(N):
    h = 2 / (N - 1)
    i = np.arange(1, N + 1)
    return -1 + (i - 1) * h

def f(x):
    return 1 / (1 + (10 * x)**2)

def vandermonde(x):
    n = len(x)
    V = np.zeros((n, n))
    for k in range(n):
        V[:, k] = x**(n - 1 - k)
    return V

def eval_poly(c, x):
    p = np.zeros_like(x)
    for ck in c:
        p = p * x + ck
    return p

def solve_coeffs(N):
    x = make_nodes(N)
    y = f(x)
    V = vandermonde(x)
    c = np.linalg.inv(V) @ y
    return x, y, c

def bary_weights(x):
    n = len(x)
    w = np.ones(n)
    for j in range(n):
        for i in range(n):
            if i != j:
                w[j] /= (x[j] - x[i])
    return w

def eval_bary1(xnodes, ynodes, w, xeval):
    p = np.zeros_like(xeval)
    for k, xk in enumerate(xeval):
        diff = xk - xnodes
        hit = np.where(diff == 0)[0]
        if hit.size > 0:            # xk is a node: interpolant equals data there
            p[k] = ynodes[hit[0]]
            continue
        Phi = np.prod(diff)
        p[k] = Phi * np.sum(w * ynodes / diff)
    return p

def cheb_nodes(N):
    j = np.arange(1, N + 1)
    return np.cos((2 * j - 1) * np.pi / (2 * N))

driver()