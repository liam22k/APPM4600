import numpy as np
import matplotlib.pyplot as plt

def driver():
    f = lambda x: np.exp(x)
    a, b = 0, 1
    Neval = 100
    xeval = np.linspace(a, b, Neval)
    Nint = 10

    yeval = eval_lin_spline(xeval, a, b, f, Nint)
    fex = f(xeval)

    plt.plot(xeval, fex, 'ro-', label='exact function')
    plt.plot(xeval, yeval, 'bs--', label='linear spline')
    plt.legend()
    plt.show()

    err = np.abs(yeval - fex)
    plt.figure()
    plt.semilogy(xeval, err, 'ro--', label='absolute error')
    plt.legend()
    plt.show()


def eval_lin_spline(xeval, a, b, f, Nint):
    xint = np.linspace(a, b, Nint + 1)
    yeval = np.zeros(len(xeval))

    for j in range(Nint):
        a1, b1 = xint[j], xint[j + 1]
        fa1, fb1 = f(a1), f(b1)

        # indices of xeval in [a1, b1); include right endpoint on last interval
        if j < Nint - 1:
            ind = np.where((xeval >= a1) & (xeval < b1))[0]
        else:
            ind = np.where((xeval >= a1) & (xeval <= b1))[0]

        yeval[ind] = eval_line(a1, fa1, b1, fb1, xeval[ind])

    return yeval


def eval_line(a1, fa1, b1, fb1, x):
    h = b1 - a1
    if h == 0:
        raise ValueError("degenerate interval")
    return fa1 * (b1 - x) / h + fb1 * (x - a1) / h


driver()