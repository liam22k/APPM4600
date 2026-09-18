import numpy as np
from scipy.integrate import quad 
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.special import erf

def driver():
    #problem 1 
    f1 = lambda x: -16 + 6*x + 12/x          # x* = 2
    f2 = lambda x: (2/3)*x + 1/x**2          # x* = 3**(1/3)
    f3 = lambda x: 12/(1 + x)                # x* = 3
    x = [1.7, 1.5, 2.7]
    tol = 10**-10
    Nmax = 100

    # [xstar, ier, i, xitter] = fixedpt(f1, x[0], tol, Nmax)
    # xitter_new = np.trim_zeros(xitter, 'b')
    # print(xitter_new)
    # alpha1 = np.log(abs((xitter[i - 2] - xstar) / (xitter[i - 3] - xstar)))/np.log(abs((xitter[i - 3] - xstar) / (xitter[i - 4] - xstar)))
    # print("this is alpha for f1: ", alpha1)


    [xstar, ier, i, xitter] = fixedpt(f2, x[1], tol, Nmax)
    xitter = np.trim_zeros(xitter, 'b')
    print(xitter)
    alpha2 = np.log(abs((xitter[3] - xstar) / (xitter[2] - xstar)))/np.log(abs((xitter[2] - xstar) / (xitter[1] - xstar)))
    print("this is alpha for f2: ", alpha2)

    foundlambda = findlambda(xitter, 3**(1/3), alpha2)
    print("this is the lambda for part b: ", foundlambda)


    [xstar, ier, i, xitter] = fixedpt(f3, x[2], tol, Nmax)
    xitter = np.trim_zeros(xitter, 'b')
    print(xitter)
    alpha3 = np.log(abs((xitter[27] - xstar) / (xitter[26] - xstar)))/np.log(abs((xitter[26] - xstar) / (xitter[25] - xstar)))
    print("this is alpha for f3: ", alpha3)

    foundlambda = findlambda(xitter, 3, alpha3)
    print("this is the lambda for part c: ", foundlambda)

    # problem 2


    # Constants
    Ti = 20.0          # initial soil temp [C]
    Ts = -15.0         # surface temp during cold snap [C]
    alpha = 0.138e-6   # thermal conductivity [m^2/s]
    t = 60 * 86400     # 60 days in seconds
    a = 0
    tol = 10**-13

    # f(x) = T(x,t) 
    f = lambda x: Ts + (Ti - Ts) * erf(x / (2 * np.sqrt(alpha * t)))

    # f'(x)
    fp = lambda x: (Ti - Ts) * (2 / np.sqrt(np.pi)) * np.exp(-x**2 / (4 * alpha * t)) * (1 / (2 * np.sqrt(alpha * t)))



    # find an xbar where f(xbar) > 0 (trial and error)
    xbar = 1.5   # adjust until f(xbar) > 0

    x = np.linspace(0, xbar, 200)
    y = f(x)

    plt.plot(x, y)
    plt.axhline(0, color='k', linewidth=0.8)  # zero line for reference
    plt.xlabel('x [m]')
    plt.ylabel('f(x)')
    plt.title('f(x) on [0, xbar]')
    plt.savefig('hw3problem2_plot.pdf')
    plt.show()
    # part b
    [astar, ier] = bisection(f, a, xbar, tol)

    print("the approximate depth is: ", astar)

    # part c
    x0 = .01
    [p,pstar,info,it] = newton(f, fp, x0, tol, Nmax)

    print("this is your root from newton: ", pstar)
    print(it)
    # now we plug in xbar
    x0 = xbar
    [p,pstar,info,it] = newton(f, fp, x0, tol, Nmax)

    print("this is your root from newton with xbar: ", pstar)
    print(it)

    # problem 3
    g1 = lambda x: x * (1 + (7 - x**5) / x**2)**3
    g2 = lambda x: x - (x**5 - 7) / x**2
    g3 = lambda x: x - (x**5 - 7) / (5 * x**4)
    g4 = lambda x: x - (x**5 - 7) / 12

    # verify fx pt
    print(g1(7**(1/5)), g2(7**(1/5)), g3(7**(1/5)), g4(7**(1/5)))
    print(float((7**(1/5))))

    # apply fx pt it's
    tol = 10**-10
    x0 = 1
    Nmax = 1000
    root = 7**(1/5)

    # [xstar, ier, i, xitter] = fixedpt(g1, x0, tol, Nmax)
    # xitter = np.trim_zeros(xitter, 'b')
    # print("this is the fx pt it's for part a: ", xitter)


    #part b
    # [xstar, ier, i, xitter] = fixedpt(g2, x0, tol, Nmax)
    # xitter = np.trim_zeros(xitter, 'b')
    # print("this is the fx pt it's for part b: ", xitter)


    #part c
    [xstar, ier, i, xitter] = fixedpt(g3, x0, tol, Nmax)
    xitter = np.trim_zeros(xitter, 'b')
    print("this is the fx pt it's for part c: ", xitter)
    print(len(xitter))
    alpha4 = np.log(abs((xitter[len(xitter) - 1] - xstar) / (xitter[len(xitter) - 2] - xstar)))/np.log(abs((xitter[len(xitter) - 2] - xstar) / (xitter[len(xitter) - 3] - xstar)))
    print(alpha4)
    [mylambda] = findlambda(xitter, root, alpha4)
    print("this is lambda: ", mylambda)

    #part d
    # [xstar, ier, i, xitter] = fixedpt(g4, x0, tol, Nmax)
    # xitter = np.trim_zeros(xitter, 'b')
    # print("this is the fx pt it's for part d: ", xitter)
    # print(len(xitter))

def fixedpt(f, x0, tol, Nmax):
    xitter = np.zeros((Nmax,1))

    count = 0
    i = 0
    ier = 1
    while (count < Nmax):
        count = count + 1
        xitter[count] = x0
        x1 = f(x0)
        if (abs(x1 - x0) < tol * max(1, abs(x1))):   # relative-style stopping criterion

            xstar = x1
            ier = 0
            return [xstar, ier, count, xitter]
            
       

        
        x0 = x1
        i = i + 1
    ier = 1
    
    return [xstar, ier, count, xitter]

def findlambda(xitter, root, alpha):
    n = len(xitter)
    

    lambda1 = abs(xitter[n-1] - root)/(abs(xitter[n-2] - root)**alpha)

    return lambda1

def bisection(f, a, b, tol):
    fa = f(a)
    fb = f(b)
    if (fa*fb > 0):
        ier = 1
        astar = a
        return [astar, ier]

    if (fa == 0):
        astar = a
        ier = 0
        return [astar, ier]

    if (fb == 0):
        astar = b
        ier = 0
        return [astar, ier]

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a) > tol):
        fd = f(d)
        if (fd == 0):
            astar = d
            ier = 0
            return [astar, ier]
        if (fa*fd < 0):
            b = d
        else:
            a = d
            fa = fd
        d = 0.5*(a+b)
        count = count + 1

    astar = d
    ier = 0
    print('count = ', count)
    return [astar, ier]


def newton(f,fp,p0,tol,Nmax):
  """
  Newton iteration.
  
  Inputs:
    f,fp - function and derivative
    p0   - initial guess for root
    tol  - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
  Returns:
    p     - an array of the iterates
    pstar - the last iterate
    info  - success message
          - 0 if we met tol
          - 1 if we hit Nmax iterations (fail)
  """
  p = np.zeros(Nmax+1)
  p[0] = p0
  for it in range(Nmax):
      p1 = p0-f(p0)/fp(p0)
      p[it+1] = p1
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          return [p,pstar,info,it]
      p0 = p1
  pstar = p1
  info = 1
  return [p,pstar,info,it]

driver()
