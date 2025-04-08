import numpy as np
from scipy.integrate import simpson, quad
from numpy.polynomial.legendre import leggauss

# a) ∫₀¹ x^(-1/4) * sin(x) dx
epsilon = 1e-6
f4a = lambda x: x**(-0.25) * np.sin(x)
x4a = np.linspace(epsilon, 1, 5)
y4a = f4a(x4a)
simpson_4a = simpson(y4a, x4a)

# b) ∫₁^∞ x^(-4) * sin(x) dx → t = 1/x → ∫₀¹ t^2 * sin(1/t) dt
f4b = lambda t: t**2 * np.sin(1/t)
x4b = np.linspace(0.01, 1, 5)
y4b = f4b(x4b)
simpson_4b = simpson(y4b, x4b)

print(f"4) Improper Integral results:")
print(f"   a) ∫₀¹ x^(-1/4)sin(x) dx ≈ {simpson_4a:.8f} (using Simpson's rule)")
print(f"   b) ∫₁^∞ x^(-4)sin(x) dx ≈ {simpson_4b:.8f} (after substitution t=1/x)")