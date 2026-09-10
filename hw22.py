import numpy as np
from scipy.integrate import quad 
import matplotlib.pyplot as plt
import math

def driver():
    print("hello, this is running")
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




    driver()