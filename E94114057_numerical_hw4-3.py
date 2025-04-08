import numpy as np
from scipy.integrate import simpson, quad
from numpy.polynomial.legendre import leggauss

def f3(x, y):
    return 2 * y * np.sin(x) + np.cos(x)**2

# a. Simpson's Rule for 2D, n = m = 4
n = m = 4
a, b = 0, np.pi / 4
hx = (b - a) / n
result_simpson_2d = 0

for i in range(n + 1):
    xi = a + i * hx
    ya, yb = np.sin(xi), np.cos(xi)
    hy = (yb - ya) / m
    y_vals = np.linspace(ya, yb, m + 1)
    f_vals = f3(xi, y_vals)
    wx = 1 if i == 0 or i == n else 4 if i % 2 == 1 else 2
    wy = f_vals[0] + f_vals[-1] + 4 * np.sum(f_vals[1:-1:2]) + 2 * np.sum(f_vals[2:-2:2])
    result_simpson_2d += wx * hy / 3 * wy

result_simpson_2d *= hx / 3

# b. Gaussian Quadrature n = 3, m = 3
def double_gauss(f, ax, bx, nx, ny):
    xg, wx = leggauss(nx)
    x_nodes = 0.5 * (xg + 1) * (bx - ax) + ax
    result = 0
    for i in range(nx):
        x = x_nodes[i]
        sinx, cosx = np.sin(x), np.cos(x)
        yg, wy = leggauss(ny)
        y_nodes = 0.5 * (yg + 1) * (cosx - sinx) + sinx
        inner = np.sum(wy * f(x, y_nodes)) * 0.5 * (cosx - sinx)
        result += wx[i] * inner
    return 0.5 * (bx - ax) * result

gauss_result_2d = double_gauss(f3, 0, np.pi / 4, 3, 3)

# c. Exact value for comparison
def exact_inner(x):
    sinx = np.sin(x)
    cosx = np.cos(x)
    return cosx**2 * (cosx - sinx) + np.sin(x) * (cosx**2 - sinx**2)

exact_val_3, _ = quad(exact_inner, 0, np.pi / 4)

print(f"3) Double Integral results:")
print(f"   a) Simpson 2D (4×4):       {result_simpson_2d:.8f}")
print(f"   b) Gaussian 2D (3×3):      {gauss_result_2d:.8f}")
print(f"   c) Exact value:            {exact_val_3:.8f}")
print(f"      Simpson error:          {abs(result_simpson_2d - exact_val_3):.2e}")
print(f"      Gauss error:            {abs(gauss_result_2d - exact_val_3):.2e}\n")