import numpy as np
from scipy.integrate import quad 
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.special import erf
from numpy.linalg import inv, norm


def driver():
    a = 5
    # problem 1
    x0 = np.array([1.0, 1.0])   # (x0, y0) = (1, 1)
    M = np.array([[1/6, 1/18],
                [0,   1/6 ]])
    tol = 10**-8

    M = np.array([[1/6, 1/18],
              [0,   1/6 ]])
    x0 = np.array([1.0, 1.0])
    xstar, its = iterate(x0, M, 1e-10)
    print("this is the number of its: ",its," this is xstar: ", xstar)

    # check alpha
    # err = norm(xstar - xstar[-1], axis=1)[:-1]   # e_n = ||x_n - x*||, drop the last (it's 0)

    # alpha = []
    # for n in range(1, len(err) - 1):
    #     alpha.append(np.log(err[n+1] / err[n]) / np.log(err[n] / err[n-1]))
    # alpha = np.array(alpha)
    # print(alpha)

    # check the relative error
    x_true = xstar[-1]          # or np.array([0.5, np.sqrt(3)/2]) once you confirm it in 1(d)

    rel_err = norm(xstar - x_true, axis=1) / norm(x_true)

    for n, e in enumerate(rel_err):
        print(f"{n:3d}   {e:.3e}")

    # problem 1 part c newton
    xstar, ier, its = Newton(x0, 1e-10, 100)
    print(xstar[-1], ier, its)

    # print the error response
    x_true = xstar[-1]          # or np.array([0.5, np.sqrt(3)/2]) once you confirm it in 1(d)

    rel_err = norm(xstar - x_true, axis=1) / norm(x_true)

    for n, e in enumerate(rel_err):
        print(f"{n:3d}   {e:.3e}")

    # problem 3 part b
    x0 = np.array([1.0, 1.0, 1.0])
    xstar, ier, its = surface_newton(x0, 1e-12, 100)
    print(xstar[-1])
    # plot the reletive error
    rel_err = norm(xstar - xstar[-1], axis=1) / norm(xstar[-1])
    rel_err = rel_err[:-1]          # drop the last one (it's exactly 0 against itself)

    plt.semilogy(range(len(rel_err)), rel_err, 'o-')
    plt.xlabel('iteration n')
    plt.ylabel('relative error')
    plt.title('Problem 3(b): relative error')
    plt.grid(True, which='both', alpha=0.3)
    plt.show()



def evalF(x):
    F = np.zeros(2)
    F[0] = 3*x[0]**2 - x[1]**2
    F[1] = 3*x[0]*x[1]**2 - x[0]**3 - 1
    return F

def evalJ(x):
    J = np.array([[6*x[0],                 -2*x[1]],
                  [3*x[1]**2 - 3*x[0]**2,   6*x[0]*x[1]]])
    return J

def iterate(x0, M, tol, Nmax=1000):
    x_0 = x0
    xstar = [x_0]                 
    for i in range(1, Nmax + 1):
        x_1 = x_0 - M.dot(evalF(x_0))
        xstar.append(x_1)
        if norm(x_1 - x_0) < tol:  
            return [np.array(xstar), i]
        x_0 = x_1
    return [np.array(xstar), Nmax]

def Newton(x0, tol, Nmax):
    xstar = [x0]
    for its in range(Nmax):
        J = evalJ(x0)
        Jinv = inv(J)
        F = evalF(x0)

        x1 = x0 - Jinv.dot(F)
        xstar.append(x1)

        if norm(x1 - x0) < tol:
            ier = 0
            return [np.array(xstar), ier, its]

        x0 = x1

    ier = 1
    return [np.array(xstar), ier, its]

def evalf(x):
    return x[0]**2 + 4*x[1]**2 + 4*x[2]**2 - 16

def evalgrad(x):
    return np.array([2*x[0], 8*x[1], 8*x[2]])

def surface_newton(x0, tol, Nmax):
    xstar = [x0]
    for its in range(Nmax):
        f = evalf(x0)
        grad = evalgrad(x0)
        d = f / grad.dot(grad)          
        x1 = x0 - d*grad                
        xstar.append(x1)
        if norm(x1 - x0) < tol:
            return [np.array(xstar), 0, its]
        x0 = x1
    return [np.array(xstar), 1, its]




    


driver()