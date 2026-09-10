"""
This program is a warm up for coding. You get used to the coding
format and practice some coding skills.
"""
#############################################
"""
Copyright (C) 2025 Adrianna M. Gillman
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
#############################################
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math
def driver():
    n = 100
    x = np.linspace(0,np.pi,n)
# this is a function handle. You can use it to define
# functions instead of using a subroutine like you
# have to in a true low level language.
    f = lambda x: x**2 + 4*x + 2*np.exp(x)
    g = lambda x: 6*x**3 + 2*np.sin(x)
    y = f(x)
    w = g(x)
# evaluate the dot product of y and w
    dp = dotProduct(y,w,n)
# print the output
    print("the dot product is : ", dp)
# 3.2 excersizes
    numpy_array1 = np.linspace(1,10,10)
    numpy_array2 = np.linspace(1,20,10)
    print(numpy_array1, numpy_array2)

    numpy_arrange = np.arange(1,10,1)
    print(numpy_arrange)
# first three entries of x
    print("the first three entires of x are:")
    for i in range(3):
        print(x[i])
# make a weird vector 
    weird = 10**(-np.linspace(1,10,10))
    print(weird)
    time = np.linspace(1, 10, 10)
    plt.plot(time, weird)
    plt.xlabel('time')
    plt.ylabel('decay')
    plt.show()


    return
def dotProduct(x,y,n):
# Computes the dot product of the n x 1 vectors x and y
    dp = 0.
    for j in range(n):
        dp = dp + x[j]*y[j]
    return dp
driver()