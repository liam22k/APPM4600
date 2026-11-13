import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sici, gamma, gammaincc
from scipy.integrate import quad
from numpy.polynomial.laguerre import laggauss

xs = [2, 4, 6, 8, 10]
L = 50      # truncation point
h = 0.01 

def g(t):
    # transformed integrand t*cos(1/t); its limit at t = 0 is 0 since |t*cos(1/t)| <= t
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    nz = t != 0
    out[nz] = t[nz]*np.cos(1/t[nz])
    return out




def driver():
    f = lambda x: np.sin(x)
    a, b = 0, np.pi
    I_ex = 2
    nrefine = 5
 
    # Trapezoid panels: 1, 2, ..., 32.  Simpson panels: 2, 4, ..., 64 (n must be even)
    n_trap = 2**np.arange(0, nrefine+1)
    n_simp = 2**np.arange(1, nrefine+2)
 
    RomT = rombergTrap(a, b, nrefine, f)
    RomS = rombergSimp(a, b, nrefine, f)
 
    errT = abs(I_ex - np.diag(RomT))
    errS = abs(I_ex - np.diag(RomS))
 
    print('Romberg-Trapezoidal (diagonal)')
    print(f'{"n":>4} {"f evals":>8} {"approx":>22} {"abs error":>12}')
    for n, R, e in zip(n_trap, np.diag(RomT), errT):
        print(f'{n:4d} {n+1:8d} {R:22.16f} {e:12.3e}')
 
    print('\nRomberg-Simpson (diagonal)')
    print(f'{"n":>4} {"f evals":>8} {"approx":>22} {"abs error":>12}')
    for n, R, e in zip(n_simp, np.diag(RomS), errS):
        print(f'{n:4d} {n+1:8d} {R:22.16f} {e:12.3e}')
 
    # compare by function evaluations (n panels -> n+1 evaluations)
    plt.semilogy(n_trap+1, errT, 'o-', label='Romberg-Trapezoidal')
    plt.semilogy(n_simp+1, errS, 's-', label='Romberg-Simpson')
    plt.xlabel('function evaluations')
    plt.ylabel('absolute error (diagonal entry)')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.savefig('hw11_p2.png', dpi=150, bbox_inches='tight')
    plt.show()

    n = 4                      # 4 panels -> 5 nodes
    t = np.linspace(0, 1, n+1)
    print('nodes t   :', t)
    print('g(t)      :', g(t))
 
    I_simp = CompSimp(0, 1, n, g)
 
    # reference value (integrate by parts twice):
    # int_1^inf cos(x)/x^3 dx = [cos(1) - sin(1) + Ci(1)]/2,  Ci = cosine integral
    si, ci = sici(1.0)
    I_ref = 0.5*(np.cos(1) - np.sin(1) + ci)
    print(f'Simpson (5 nodes) = {I_simp:.10f}')
    print(f'reference         = {I_ref:.10f}')
    print(f'abs error         = {abs(I_simp-I_ref):.3e}')




    print(f'Truncation: L = {L}.  Relative tail Gamma(x,L)/Gamma(x):')
    for x in xs:
        print(f'   x = {x:2d}:  {gammaincc(x, L):.1e}')
 
    print(f'\n(a) trapezoid (h = {h})   vs   (b) adaptive quad, both on [0, {L}]')
    print(f'{"x":>3} {"Gamma(x)":>10} {"trap rel err":>13} {"evals":>6} {"quad rel err":>13} {"evals":>6}')
    for x in xs:
        ex = gamma(x)
        It, nt = gamma_trap(x)
        Iq, nq = gamma_adapt(x)
        print(f'{x:3d} {ex:10.0f} {abs(It-ex)/ex:13.2e} {nt:6d} {abs(Iq-ex)/ex:13.2e} {nq:6d}')
 
    print('\n(c) Gauss-Laguerre: relative error vs number of nodes n')
    ns = [2, 3, 4, 5, 10, 20]
    print(f'{"x":>3} ' + ' '.join(f'{"n="+str(n):>9}' for n in ns))
    for x in xs:
        ex = gamma(x)
        print(f'{x:3d} ' + ' '.join(f'{abs(gamma_laguerre(x, n)-ex)/ex:9.1e}' for n in ns))
 
 
def rombergTrap(a, b, nrefine, f):
    ntest = 2**np.arange(0, nrefine+1)
    nn = len(ntest)
    Rom = np.zeros((nn, nn))
    for j in range(nn):
        Rom[j, 0] = CompTrap(a, b, ntest[j], f)[0]   # CompTrap returns (integral, # evals)
    # error expansion h^2, h^4, ...  ->  divide by 4^j - 1
    for j in range(1, nn):
        for i in range(j, nn):
            Rom[i, j] = Rom[i, j-1] + (Rom[i, j-1] - Rom[i-1, j-1])/(4**j - 1)
    return Rom
 
 
def rombergSimp(a, b, nrefine, f):
    # Simpson needs an even number of panels: 2, 4, ..., 2^(nrefine+1)
    ntest = 2**np.arange(1, nrefine+2)
    nn = len(ntest)
    Rom = np.zeros((nn, nn))
    for j in range(nn):
        Rom[j, 0] = CompSimp(a, b, ntest[j], f)
    # error expansion h^4, h^6, ...  ->  divide by 4^(j+1) - 1
    for j in range(1, nn):
        for i in range(j, nn):
            Rom[i, j] = Rom[i, j-1] + (Rom[i, j-1] - Rom[i-1, j-1])/(4**(j+1) - 1)
    return Rom

def gamma_laguerre(x, n):
    # int_0^inf e^{-t} g(t) dt  ~  sum_i w_i g(t_i),   with g(t) = t^(x-1)
    t, w = laggauss(n)
    return np.sum(w*t**(x-1))

def gamma_trap(x):
    f = lambda t: t**(x-1)*np.exp(-t)
    return CompTrap(0, L, int(round(L/h)), f)
 
 
def gamma_adapt(x, tol=1e-10):
    f = lambda t: t**(x-1)*np.exp(-t)
    I, _, info = quad(f, 0, L, epsabs=0, epsrel=tol, full_output=1)
    return I, info['neval']


def CompTrap(a, b, n, f):
    h = (b-a)/n
    x = a + np.arange(0, n+1)*h
    return h*(0.5*f(x[0]) + np.sum(f(x[1:n])) + 0.5*f(x[n])), n+1
 
 
def CompSimp(a, b, n, f):
    h = (b-a)/n
    x = a + np.arange(0, n+1)*h
    return h/3*(f(x[0]) + 4*np.sum(f(x[1:n:2])) + 2*np.sum(f(x[2:n-1:2])) + f(x[n]))
 
 
if __name__ == '__main__':
    driver()