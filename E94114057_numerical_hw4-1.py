import numpy as np
from scipy.integrate import simpson, quad
from numpy.polynomial.legendre import leggauss

# Problem 1: ∫₁² e^x * sin(4x) dx, h = 0.1
f1 = lambda x: np.exp(x) * np.sin(4 * x)
x_vals = np.arange(1, 2.1, 0.1)
y_vals = f1(x_vals)

# a. Trapezoidal Rule
trapezoid_result = np.trapz(y_vals, x_vals)
print(f"1.a) Trapezoidal Rule result: {trapezoid_result:.8f}")

# b. Simpson's Method
simpson_result = simpson(y_vals, x_vals)
print(f"1.b) Simpson's Method result: {simpson_result:.8f}")

# c. Midpoint Rule
midpoints = (x_vals[:-1] + x_vals[1:]) / 2
mid_vals = f1(midpoints)
midpoint_result = np.sum(mid_vals * 0.1)
print(f"1.c) Midpoint Rule result: {midpoint_result:.8f}\n")