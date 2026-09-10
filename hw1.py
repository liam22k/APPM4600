" this is the hw1 for this class, it goes in order problem 1 to the last problem... creator: Liam Kincaid"
import numpy as np
from scipy.integrate import quad 
import matplotlib.pyplot as plt
import math

def driver():
    # problem number 1
    p_coeffs = lambda x: (x**9 - 18*x**8 + 144*x**7 - 672*x**6 + 2016*x**5
                        - 4032*x**4 + 5376*x**3 - 4608*x**2 + 2304*x - 512)
    p_factored = lambda x: (x - 2)**9

    x = np.arange(1.920, 2.080 + 0.001, 0.001)
    y1 = p_coeffs(x)
    y2 = p_factored(x)

    # plot the two functions
    plt.plot(x, y1, label = 'p_coeffs')
    plt.plot(x, y2, label = 'p_factored')
    plt.xlabel('x')
    plt.ylabel('functions')
    plt.title('p_coeffs vs. p_factored')
    plt.legend()
    plt.show()

    # problem 2 : code not needed

    # problem 3

    # make the functions
    p_2x = lambda x: 1+x-.5*x**2
    fx = lambda x: (1+x+x**3)*math.cos(x)
    fx_jerk = lambda x: (3-9*x)*math.cos(x)+(1-17*x+x**3)*math.sin(x)
    # now approx
    x0 = .5
    abs_er = abs(fx(x0) - p_2x(x0))
    print(abs_er)
    max_er = 3*(.5**3)/6
    print(abs(max_er))

    # integrate the error function
    erx = lambda x: fx_jerk(x)* ((x**3)/6)

    approx_error = quad(erx, 0, 1)
    print(approx_error)

    # do abs error for curiosity
    abs_function = lambda x: fx(x) - p_2x(x)
    abs_error = quad(abs_function, 0 , 1)
    print(abs_error)

    # problem 4
    a = 1
    b = -56
    c = 1
    poly = lambda x: a*x**2 + b*x + c

    # quadratic equation
    root1_act = (-b+math.sqrt(b**2 - 4*a*c))/(2*a)
    root2_act = (-b-math.sqrt(b**2 - 4*a*c))/(2*a)
    print(root1_act,root2_act)

    dis = math.sqrt(b**2 - 4*a*c)
    dis_round = round(dis, 3)

    root1_app = (-b+dis_round)/(2*a)
    root2_app = (-b-dis_round)/(2*a)
    print(root1_app, root2_app)

    # find the relative error
    rel_error = [abs(root1_act - root1_app)/abs(root1_act), abs(root2_act - root2_app)/abs(root2_act)]
    print(rel_error)

    r2_better = 1/root1_app
    print(r2_better)
    rel_error_better = abs(root2_act - r2_better)/abs(root2_act)
    print(rel_error_better)

    # problem 5
    # part b

    delta = np.logspace(-16, 0, 17)
    naive = lambda x: np.cos(x + delta) - np.cos(x)
    stable = lambda x: -2*np.sin(x + delta/2)*np.sin(delta/2)
    diff = lambda x: naive(x) - stable(x)

    # plot everything
    plt.plot(delta, diff(np.pi), label='x = pi')
    plt.plot(delta, diff(10**6), label='x = 10^6')
    plt.xlabel('delta')
    plt.ylabel('difference (naive - stable)')
    plt.title('x = pi vs. x = 10^6')
    plt.xscale('log')
    plt.legend()
    plt.show()


    #part c plot 
    taylor = lambda x: -delta*np.sin(x)                            # part (c) algorithm

    diff2 = lambda x: taylor(x) - stable(x)

    plt.plot(delta, diff2(np.pi), label='x = pi')
    plt.plot(delta, diff2(10**6), label='x = 10^6')
    plt.xlabel('delta')
    plt.ylabel('difference (taylor - stable)')
    plt.title('Taylor approximation vs. part (b) identity')
    plt.xscale('log')
    plt.legend()
    plt.show()

driver()