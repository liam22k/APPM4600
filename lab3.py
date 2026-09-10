import numpy as np
from scipy.integrate import quad 
import matplotlib.pyplot as plt
import math

def driver():

    # 2.2.1 


    # 2.2 part a
    f = lambda x: math.sqrt(10/(x+4))
    tol = 10**-10
    x0 = 1.5
    Nmax = 30
    

    [xstar, ier, i, xitter] = fixedpt(f, x0, tol, Nmax)
    print("this is xstar and ier: ", xstar, ier)
    xitter_new = np.trim_zeros(xitter, 'b')
    print(xitter_new)
    print(i)

    alpha = np.log(abs((xitter[i - 2] - xstar) / (xitter[i - 3] - xstar)))/np.log(abs((xitter[i - 3] - xstar) / (xitter[i - 4] - xstar)))

    print("this is alpha: ", alpha)

    length = len(xitter)

    [p_hat, j] = aitkens(xitter, length, x0, f, tol)

    print(p_hat, j)

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

# apply aikons method
def  aitkens(xitter, length, x0, g, tol):
    count = 0
    x1 = x0
    while(count <= length):
        a = x1
        b = g(x1)
        c = g(b)
        p_n1 = a - ((b - a)**2)/(c-2*b+a)
        count = count + 1
        if (abs(p_n1 - c) < tol * max(1, abs(c))):   # relative-style stopping criterion
        
            
            return [p_n1, count]
       
        
        x1 = p_n1
    return p_n1

    
    



driver()