import numpy as np
from scipy.integrate import quad 
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.special import erf
from numpy.linalg import inv, norm

def driver():

    x0 = np.array([2.0, .5])   # also try np.array([3.0, 5.0])
    tol = 1e-6
    Nmax = 100
    [xstar, ier, its] = Newton(x0, tol, Nmax)
    print("this is regular newton: ", xstar, ier, its)
    
    [xstar, ier, its] = LazyNewton(x0, tol, Nmax)
    print("this is lazy newton: ", xstar, ier, its)

    [xstar, ier, its] = slackerNewton(x0, tol, Nmax)
    print("this is lazy newton: ", xstar, ier, its)

    x0 = np.array([1, 0])
    [xstar, ier, its] = slackerNewtonG(x0, tol, Nmax)
    print("this is lazy newton: ", xstar, ier, its)

def evalF(x):
    F = np.zeros(2)
    F[0] = x[0]**2 + x[1]**2 - 2
    F[1] = np.exp(x[0] - 1) + x[1]**2 - 2
    return F

def evalG(x):
    G = np.zeros(2)
    G[0] = 4*x[0]**2 + x[1]**2 - 4
    G[1] = x[0] + x[1] - np.sin(x[0] - x[1])
    return G

def evalJ(x):
    J = np.array([[2*x[0],          2*x[1]],
                  [np.exp(x[0]-1),  2*x[1]]])
    return J

def evaljacob(x):
    Jacobs = np.array([[8*x[0],          2*x[1]],
                   [1-np.cos(x[0] - x[1]),  1+np.cos(x[0] - x[1])]])
    return Jacobs
def Newton(x0, tol, Nmax):
    xstar = []
    x_0 = x0
    count = 0
    ier = 1
    for i in range(Nmax):
        Jacob = evalJ(x_0)
        jinv = inv(Jacob)
        x_1 = (x_0 - jinv.dot(evalF(x_0))) 
        xstar.append(x_1)
        count =+ 1


        if norm(x_1 - x_0) < tol:
            ier = 0
            return[np.array(xstar), ier, count]
        
        x_0 = x_1
    return[xstar, ier, count]
    

def LazyNewton(x0, tol, Nmax):
    xstar = []
    x_0 = x0
    Jacob = evalJ(x0)
    count = 0
    ier = 1
    for i in range(Nmax):
        x_1 = (x_0 - inv(Jacob).dot(evalF(x_0))) 
        xstar.append(x_1)
        count =+ 1


        if norm(x_1 - x_0) < tol:
            ier = 0
            return[np.array(xstar), ier, count]
        
        x_0 = x_1
    return[xstar, ier, count]

def slackerNewton(x0, tol, Nmax):
    xstar = []
    x_0 = x0
    Jacob = evalJ(x0)
    count = 0
    ier = 1
    for i in range(Nmax):
        x_1 = (x_0 - inv(Jacob).dot(evalF(x_0))) 
        xstar.append(x_1)
        count += 1


        if norm(x_1 - x_0) < tol:
            ier = 0
            return[np.array(xstar), ier, count]
        # elif norm(x_1 - x_0)/norm(x_1) > 1:
        #     Jacob = evalJ(x_1)
        elif count % 2 == 0:
            Jacob = evalJ(x_1)
            
        
        x_0 = x_1
    return[xstar, ier, count]

def slackerNewtonG(x0, tol, Nmax):
    xstar = []
    x_0 = x0
    Jacob = evaljacob(x0)
    count = 0
    ier = 1
    for i in range(Nmax):
        x_1 = (x_0 - inv(Jacob).dot(evalG(x_0))) 
        xstar.append(x_1)
        count += 1


        if norm(x_1 - x_0) < tol:
            ier = 0
            return[np.array(xstar), ier, count]
        # elif norm(x_1 - x_0)/norm(x_1) > 1:
        #     Jacob = evalJ(x_1)
        elif count % 2 == 0:
            Jacob = evaljacob(x_1)
            
        
        x_0 = x_1
    return[xstar, ier, count]


driver()