"""
 This script provides you the opprotunity play with different linear algebra solvers.
You will learn about the performance of the different solvers and how to build
test some test matrices to challenge the different methods. 
"""

############################################# 
"""
Copyright (C) 2025  Adrianna M. Gillman

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
############################################# 


import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as scila
from scipy.linalg import lu
import time

def driver():

     ''' create  matrix for testing different ways of solving a square 
     linear system'''

     '''' N = size of system'''
     N = [100, 500, 1000, 2000, 4000, 5000]
     for i in range(len(N)):
          ''' Right hand side'''
          b = np.random.rand(N[i],1)
          A = np.random.rand(N[i],N[i])

          start_time = time.perf_counter()
          x = scila.solve(A,b)
          end_time = time.perf_counter()
          elapsed = end_time - start_time
          print(f"Elapsed time: {elapsed:.4f} seconds")


          test = np.matmul(A,x)
          r = la.norm(test-b)

          start_time = time.perf_counter()
          P, L, U = lu(A)

          x_lu = la.solve( P @ L @ U, b)
          end_time = time.perf_counter()
          elapsed = end_time - start_time
          print(f"Elapsed time: {elapsed:.4f} seconds")

          print(x_lu)

          test_lu = np.matmul(A,x_lu)
          r_lu = la.norm(test_lu-b)



          
          print(r)
          print(r_lu)


     a = 1000
     count = 0
     for i in range(a):
          ''' Right hand side'''
          b = np.random.rand(i,1)
          A = np.random.rand(i,i)

          start_time = time.perf_counter()
          x = scila.solve(A,b)
          end_time = time.perf_counter()
          elapsed1 = end_time - start_time
          print(f"Elapsed time: {elapsed1:.8f} seconds")


          test = np.matmul(A,x)
          r = la.norm(test-b)

          start_time = time.perf_counter()
          P, L, U = lu(A)

          x_lu = la.solve( P @ L @ U, b)
          end_time = time.perf_counter()
          elapsed2 = end_time - start_time
          print(f"Elapsed time: {elapsed2:.8f} seconds")

          # print(x_lu)

          test_lu = np.matmul(A,x_lu)
          r_lu = la.norm(test_lu-b)

          if (elapsed1<elapsed2):
               print("reg is faster")
          else:
               count += 1

     print(count)

     ''' Create an ill-conditioned rectangular matrix '''
     N = 10
     M = 5
     A = create_rect(N,M)     
     b = np.random.rand(N,1)

     # # write the Q,R factorization code

     # Q, R = la.qr(A, mode='economic')
     # c = Q.T @ b
     # x_qr = la.solve(R, c)

     # # now the solution for the other way
     # x_ne = la.solve(A.T @ A, A.T @ b)


     # print("this is the solution for qr = ", x_qr)
     # print("this is the solution for the other = ", x_ne)


     
def create_rect(N,M):
     ''' this subroutine creates an ill-conditioned rectangular matrix'''
     a = np.linspace(1,10,M)
     d = 10**(-a)
     
     D2 = np.zeros((N,M))
     for j in range(0,M):
        D2[j,j] = d[j]
     
     '''' create matrices needed to manufacture the low rank matrix'''
     A = np.random.rand(N,N)
     Q1, R = la.qr(A)
     test = np.matmul(Q1,R)
     A =    np.random.rand(M,M)
     Q2,R = la.qr(A)
     test = np.matmul(Q2,R)
     
     B = np.matmul(Q1,D2)
     B = np.matmul(B,Q2)
     return B     
          
  
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()       
