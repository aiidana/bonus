import numpy as np
import matplotlib.pyplot as plt

def function_x(x):
    return x**2 - x**3

def fourier_approximation(x, m):
    approx_values = np.zeros_like(x)
    for n in range(1, m + 1):
        a_n = (-8 * (-1)**n - 4) / (np.pi**3 * n**3)
        approx_values += a_n * np.sin(n * np.pi * x)
    return approx_values

x_domain = np.linspace(0, 1, 100)
f_domain = function_x(x_domain)

approx_5 = fourier_approximation(x_domain, m=5)
approx_10 = fourier_approximation(x_domain, m=10)
approx_20 = fourier_approximation(x_domain, m=20)

plt.figure(figsize=(10, 6))
plt.plot(x_domain, f_domain, label='Initial Function: f(x) = x^2-x^3', color="black")
plt.plot(x_domain, approx_5, label="Fourier Approximation m=5", linestyle='--')
plt.plot(x_domain, approx_10, label="Fourier Approximation m=10", linestyle='-.')
plt.plot(x_domain, approx_20, label="Fourier Approximation m=20", linestyle=':')
plt.title("Fourier Series Approximation of f(x) = x^2-x^3 when m =(5, 10, 20)")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

sqrt_7 = np.sqrt(7)
def u_xt(x, t, m=10):
    sum_result = np.zeros_like(x)
    for n in range(1, m + 1):
        an = (-8 * (-1)**n - 4) / (np.pi**3 * n**3)
        cos_term = np.cos(sqrt_7 * np.pi * n * t)
        sin_term = np.sin(n * np.pi * x)
        sum_result += an * cos_term * sin_term
    return sum_result

time_values = [0, 0.01, 0.05, 0.1]
plt.figure(figsize=(10, 6))
for t in time_values:
    u_values = u_xt(x_domain, t, m=20)
    plt.plot(x_domain, u_values, label=f't={t}')
plt.title("Solution u(x, t)")
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.legend()
plt.grid(True)
plt.show()