import numpy as np
from scipy.integrate import simpson, quad
from numpy.polynomial.legendre import leggauss

def f2(x):
    return x**2 * np.log(x)

def gaussian_quadrature(f, a, b, n):
    x, w = leggauss(n)
    t = 0.5 * (x + 1) * (b - a) + a
    return 0.5 * (b - a) * np.sum(w * f(t))

true_val2, _ = quad(f2, 1, 1.5)
gauss_n3 = gaussian_quadrature(f2, 1, 1.5, 3)
gauss_n4 = gaussian_quadrature(f2, 1, 1.5, 4)

print(f"2) Gaussian Quadrature results:")
print(f"   n=3 approximation: {gauss_n3:.8f}")
print(f"   n=4 approximation: {gauss_n4:.8f}")
print(f"   Exact value:       {true_val2:.8f}")
print(f"   Error (n=3):       {abs(gauss_n3 - true_val2):.2e}")
print(f"   Error (n=4):       {abs(gauss_n4 - true_val2):.2e}\n")