import numpy as np
from scipy.integrate import quad 
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.special import erf
from numpy.linalg import inv, norm


def driver():
    # problem 1
    guesses = [np.array([-0.5, 0.25]),
               np.array([-1.5, 1.25]),
               np.array([-3.0, 3.0])]
    tol = 1e-4
    Nmax = 100
 
    for x0 in guesses:
        print(f'Initial guess {x0}')
        [xfinal, ier, its, xstar] = fixedSys(x0, tol, Nmax)
        print("Fixed pt: xstar = ", xstar)
        print("Fixed pt: ier = ", ier)
        print("Fixed pt: its = ", its)
        [xfinal, ier, its, xstar] = Newton(x0, tol, Nmax)
        print("newton: xstar = ", xstar)
        print("newton: ier = ", ier)
        print("newton: its = ", its)

    # problem 2    
    guesses = [np.array([1.0, 1.0]),
    np.array([1.0, -1.0])]#,
    # np.array([0.0, 0.0])]
    tol = 1e-10
    Nmax = 100
 
    for x0 in guesses:
        print(f'Initial guess {x0}')
        [xfinal, ier, its, xstar] = Newton2(x0, tol, Nmax)
        print("newton: xstar = ", xstar)
        print("newton: ier = ", ier)
        print("newton: its = ", its)
        [xfinal, ier, its, xstar] = LazyNewton(x0, tol, Nmax)
        print("lazy newton: xstar = ", xstar)
        print("lazy newton: ier = ", ier)
        print("lazy newton: its = ", its)
        [xfinal, ier, its, xstar] = Broyden(x0, tol, Nmax)
        print("broyden: xstar = ", xstar)
        print("broyden: ier = ", ier)
        print("broyden: its = ", its)

    # problem 3
    x0 = np.array([0.5, 1.0, -1.0])
    tol = 1e-6
    Nmax = 100
 
    [xfinal, ier, its, xstar] = Newton3(x0, tol, Nmax)
    print("newton: xstar = ", xfinal)
    print("newton: ier = ", ier)
    print("newton: its = ", its)
 
    [xfinal, ier, its, xstar] = SteepestDescent(x0, tol, Nmax)
    print("steepest descent: xstar = ", xfinal)
    print("steepest descent: ier = ", ier)
    print("steepest descent: its = ", its)
 
    # hybrid: steepest descent to a loose tolerance, then Newton
    [xsd, ier1, its1, xstar1] = SteepestDescent(x0, 5e-2, Nmax)
    [xfinal, ier2, its2, xstar2] = Newton3(xsd, tol, Nmax)
    print("hybrid: xstar = ", xfinal)
    print("hybrid: ier = ", ier2)
    print("hybrid: its = ", its1, "(steepest) +", its2, "(newton)")
    print("total its = ", its1+its2)
 
 

def evalF3(x):
    F = np.zeros(3)
    F[0] = x[0] + math.cos(x[0]*x[1]*x[2]) - 1.
    F[1] = (1. - x[0])**0.25 + x[1] + 0.05*x[2]**2 - 0.15*x[2] - 1.
    F[2] = -x[0]**2 - 0.1*x[1]**2 + 0.01*x[1] + x[2] - 1.
    return F
 
 
def evalJ3(x):
    s = math.sin(x[0]*x[1]*x[2])
    J = np.array([[1. - x[1]*x[2]*s,       -x[0]*x[2]*s, -x[0]*x[1]*s],
                  [-0.25*(1. - x[0])**(-0.75), 1.,        0.1*x[2] - 0.15],
                  [-2.*x[0],               -0.2*x[1] + 0.01, 1.]])
    return J
 
 
def evalg3(x):
    # least squares function g(x) = sum f_i(x)^2
    F = evalF3(x)
    return F.dot(F)
 
 
def eval_gradg3(x):
    # grad g = 2 J^T F  (the factor of 2 is dropped, only the direction matters)
    return np.transpose(evalJ3(x)).dot(evalF3(x))
 
 

def Newton3(x0, tol, Nmax):

    xstar = [x0]
    for its in range(Nmax):
        J = evalJ3(x0)
        if abs(np.linalg.det(J)) < 1e-14:
            return [x0, 1, its, xstar]
        x1 = x0 - inv(J).dot(evalF3(x0))
        xstar.append(x1)
        if norm(x1 - x0) < tol:
            return [x1, 0, its, xstar]
        x0 = x1
    return [x1, 1, its, xstar]
 
 

def SteepestDescent(x, tol, Nmax):

    xstar = [x]
    for its in range(Nmax):
        g1 = evalg3(x)
        z = eval_gradg3(x)
        z0 = norm(z)
        if z0 == 0:
            print("zero gradient")
            return [x, 0, its, xstar]
        z = z/z0
 
        # 2. find alpha3 with g(x - alpha3 z) < g1 by halving
        alpha3 = 1.
        g3 = evalg3(x - alpha3*z)
        while g3 >= g1:
            alpha3 = alpha3/2
            g3 = evalg3(x - alpha3*z)
            if alpha3 < tol:
                print("no likely improvement")
                return [x, 0, its, xstar]
 
        # 3. fit a quadratic P(alpha) through alpha = 0, alpha2, alpha3
        alpha2 = alpha3/2
        g2 = evalg3(x - alpha2*z)
        h1 = (g2 - g1)/alpha2
        h2 = (g3 - g2)/(alpha3 - alpha2)
        h3 = (h2 - h1)/alpha3
 
        # 4. step to the minimum of the quadratic (or alpha3 if that is better)
        alpha0 = 0.5*(alpha2 - h1/h3)
        g0 = evalg3(x - alpha0*z)
        if g0 <= g3:
            alpha, gval = alpha0, g0
        else:
            alpha, gval = alpha3, g3
 
        x = x - alpha*z
        xstar.append(x)
 
        # 5. stop when g stops decreasing
        if abs(gval - g1) < tol:
            return [x, 0, its, xstar]
 
    print('max iterations exceeded')
    return [x, 1, its, xstar]

 
def evalF2(x):
    F = np.zeros(2)
    F[0] = x[0]**2 + x[1]**2 - 4
    F[1] = np.exp(x[0]) + x[1] - 1
    return F
 
 
def evalJacob(x):
    J = np.array([[2*x[0],       2*x[1]],
                  [np.exp(x[0]), 1.0]])
    return J
 
 
def Newton2(x0, tol, Nmax):
    xstar = []
    for its in range(Nmax):
        J = evalJacob(x0)
        Jinv = inv(J)
        if abs(np.linalg.det(J)) < 1e-14:      # singular Jacobian -> stop
            return [x0, 1, its, xstar]
        F = evalF2(x0)
        x1 = x0 - Jinv.dot(F)
        xstar.append(x1)
        if norm(x1 - x0) < tol:
            return [x1, 0, its, xstar]
        x0 = x1
    return [x1, 1, its, xstar]
 
 
def LazyNewton(x0, tol, Nmax):
    J = evalJacob(x0)
    Jinv = inv(J)
    if abs(np.linalg.det(J)) < 1e-14:
        return [x0, 1, 0, xstar]
    xstar = []
    for its in range(Nmax):
        F = evalF2(x0)
        x1 = x0 - Jinv.dot(F)
        xstar.append(x1)
        if not np.all(np.isfinite(x1)):
            return [x1, 1, its, xstar]
        if norm(x1 - x0) < tol:
            return [x1, 0, its, xstar]
        x0 = x1
    return [x1, 1, its, xstar]
 
 
def Broyden(x0, tol, Nmax):

    A0 = evalJacob(x0)
    v = evalF2(x0)
    A = inv(A0)
    if abs(np.linalg.det(A)) < 1e-14:
        return [x0, 1, 0, xstar]
    s = -A.dot(v)
    xk = x0 + s
    xstar = []
    for its in range(Nmax):
        xstar.append(xk)
        w = v
        v = evalF2(xk)
        y = v - w
        z = -A.dot(y)
        p = -np.dot(s, z)
        u = np.dot(s, A)
        A = A + 1./p*np.outer(s + z, u)
        s = -A.dot(v)
        xk = xk + s
        if not np.all(np.isfinite(xk)):
            return [xk, 1, its, xstar]
        if norm(s) < tol:
            return [xk, 0, its, xstar]
    return [xk, 1, its, xstar]
 
 
def evalF(x):
    F = np.zeros(2)
    F[0] = 3*x[0]**2 + 4*x[1]**2 - 1
    F[1] = x[1]**3 - 8*x[0]**3 - 1
    return F
 
 
def evalJ(x):
    J = np.array([[6*x[0],      8*x[1]],
                  [-24*x[0]**2, 3*x[1]**2]])
    return J
 
 
def evalG(x):
    M = np.array([[0.016, -0.17],
                  [0.52,  -0.26]])
    return x - M.dot(evalF(x))
 
 
def fixedSys(x0, tol, Nmax):
    xstar = []
    for its in range(Nmax):
        x1 = evalG(x0)
        xstar.append(x1)
        if not np.all(np.isfinite(x1)):
            return [x1, 1, its, xstar]
        if norm(x1 - x0) < tol:
            return [x1, 0, its, xstar]
        x0 = x1
    return [x1, 1, its, xstar]
 
 
def Newton(x0, tol, Nmax):
    xstar = []
    for its in range(Nmax):
        J = evalJ(x0)
        Jinv = inv(J)
        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)
        xstar.append(x1)
        if norm(x1 - x0) < tol:
            return [x1, 0, its, xstar]
        x0 = x1
    return [x1, 1, its, xstar]
 
 



driver()