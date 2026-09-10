import numpy as np
from scipy.integrate import quad 
import matplotlib.pyplot as plt
import math

def driver():

    # problem 2


    # Exact matrix A and its (given) inverse
    eps = 1e-10
    A = 0.5 * np.array([[1, 1],
                        [1 + eps, 1 - eps]])

    A_inv = np.array([[1 - 1e10,  1e10],
                    [1 + 1e10, -1e10]])



    b = np.array([1.0, 1.0])
    x = np.array([1.0, 1.0])          # exact solution to Ax = b

    #  condition number 
    kappa = np.linalg.cond(A, p=np.inf)
    print("kappa_inf(A) =", kappa)

    #  perturbation: same value in both components 
    delta = 1e-5
    db_same = np.array([delta, delta])
    dx_same = A_inv @ db_same
    print("\nSame perturbation:")
    print("  db =", db_same)
    print("  dx =", dx_same)
    print("  relative error ||dx||/||x|| =", np.linalg.norm(dx_same, np.inf) / np.linalg.norm(x, np.inf))

    #  perturbation: different values in each component 
    db_diff = np.array([1e-5, -1e-5])
    dx_diff = A_inv @ db_diff
    print("\nDifferent perturbation:")
    print("  db =", db_diff)
    print("  dx =", dx_diff)
    print("  relative error ||dx||/||x|| =", np.linalg.norm(dx_diff, np.inf) / np.linalg.norm(x, np.inf))

    #  compare against the condition number bound 
    bound_same = kappa * np.linalg.norm(db_same, np.inf) / np.linalg.norm(b, np.inf)
    bound_diff = kappa * np.linalg.norm(db_diff, np.inf) / np.linalg.norm(b, np.inf)
    print("\nkappa * ||db||/||b|| bound (same) =", bound_same)
    print("kappa * ||db||/||b|| bound (diff) =", bound_diff)

    #  verify by directly solving the perturbed system 
    x_perturbed_same = np.linalg.solve(A, b + db_same)
    x_perturbed_diff = np.linalg.solve(A, b + db_diff)
    print("\nx + dx (same, via A_inv)      =", x + dx_same)
    print("solve(A, b+db_same) directly  =", x_perturbed_same)
    print("x + dx (diff, via A_inv)      =", x + dx_diff)
    print("solve(A, b+db_diff) directly  =", x_perturbed_diff)

    # problem 3
    fx = lambda x: math.exp(x) - 1
    fx_prime = lambda x: math.exp(x)

    # conditional number
    k_num = lambda x: abs(x * fx_prime(x) / fx(x))

    # define buns function

    def f(x):
        y = math.exp(x)
        return y-1
    # part c
    value = 9.999999995000000*10**(-10)

    a = f(value)

    print("this is the part c answer: ", a)

    # part d
    b = 10**(-16)
    n = 1
    rel_error = 1
    while rel_error > b:
        error = (math.exp(value) / math.factorial(n)) * (value**n)
        rel_error = error / value
        print(n, rel_error)
        n = n + 1

   

    true_val = math.expm1(value)   # trusted, stable reference for e^x - 1

    P = value + value**2 / 2       

    rel_error = abs(true_val - P) / abs(true_val)

    print("true value (expm1)   :", true_val)
    print("Taylor polynomial P(x):", P)
    print("relative error        :", rel_error)
    print("expm1 reference:", math.expm1(value))

    # problem 4
    f = lambda x: 2*x - 1 - np.sin(x)
    a = 0
    b = 1

    tol = 1e-8   # bisection stops when interval length < tol

    [astar, ier] = bisection(f, a, b, tol)
    print('the approximate root is', astar)
    print('the error message reads:', ier)
    print('f(astar) =', f(astar))

    # problem 5
    f2 = lambda x: x**3 + x - 4
    a = 1
    b = 4

    tol = 1e-3   # bisection stops when interval length < tol

    [astar, ier] = bisection(f2, a, b, tol)
    print('the approximate root is', astar)
    print('the error message reads:', ier)
    print('f(astar) =', f(astar))

    # prooblem 6
    x = np.linspace(-2, 8, 2000)
    f3 = x - 4*np.sin(2*x) - 3

    plt.figure()
    plt.plot(x, f3)
    plt.axhline(0, color='black', linewidth=0.8)   # reference line at y=0 to spot crossings easily
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('f(x) = x - 4 sin(2x) - 3')
    plt.grid(True)
    plt.savefig('hw2problem6_plot.pdf')
    plt.show()

    sign_changes = np.sum(np.diff(np.sign(f3)) != 0)
    print("approximate number of zero crossings:", sign_changes)

    # find all roots precisely by bracketing each sign change and bisecting
    f3_func = lambda t: t - 4*np.sin(2*t) - 3
    roots = []
    for i in range(len(x) - 1):
        if f3[i] * f3[i+1] < 0:
            r = bisection(f3_func, x[i], x[i+1], 1e-10)[0]
            roots.append(r)
    print("all roots found:", roots)

    # part b
    g = lambda x: -np.sin(2*x) + 5*x/4 - 3/4

    Nmax = 200
    tol = 1e-10   # tightened from the class default of 1e-6, for ~10 correct digits

    # try starting guesses spread across the interval from part (a), one per suspected root
    initial_guesses = [-1, 0, 1, 2, 3, 4, 5, 6, 7]

    for x0 in initial_guesses:
        [xstar, ier] = fixedpt(g, x0, tol, Nmax)
        print(f"x0 = {x0}: approx fixed point = {xstar}, ier = {ier}")

    # theoretical check: g'(x) = -2cos(2x) + 5/4 at each root found above
    g_prime = lambda t: -2*np.cos(2*t) + 5/4
    print("\nChecking |g'(x)| at each root:")
    for r in roots:
        gp = g_prime(r)
        print(f"  root = {r:.10f}, g'(root) = {gp:.6f}, |g'(root)| < 1? {abs(gp) < 1}")


def fixedpt(f, x0, tol, Nmax):
    ''' x0 = initial guess'''
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    while (count < Nmax):
        count = count + 1
        x1 = f(x0)
        if (abs(x1 - x0) < tol * max(1, abs(x1))):   # relative-style stopping criterion
            xstar = x1
            ier = 0
            return [xstar, ier]
        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier]

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




driver()