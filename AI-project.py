import math
import numpy

N = 10
r = 3

print(f"Ncr = {math.comb(N, r)}")
print(f"Npr = {math.perm(N, r)}")

A = 4
Mod = 11

print(f'The Modular Multiplicative Inverse \
of A under Mod is {pow(A, -1, Mod)}')


x = numpy.array([[1, 2], [4, 5]])
y = numpy.array([[4, 5], [6, 7]])

print(numpy.divide(x, y))
print (numpy.dot(x, y))