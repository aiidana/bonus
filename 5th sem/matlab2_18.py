import numpy as np
import matplotlib.pyplot as plt

def function_x(x):
    return 6*x**2 - 2*x**3

def fourier_approximation(x, m):
    approx_values = np.zeros_like(x)
    a0=4
    for n in range(1, m + 1):
        a_n = 192*((-1)**n -1)/(n**4 * np.pi**4)
        approx_values += a_n * np.cos(n * np.pi * x/2)
    return a0+approx_values

x_domain = np.linspace(0, 2, 100)
f_domain = function_x(x_domain)

approx_5 = fourier_approximation(x_domain, m=5)
approx_10 = fourier_approximation(x_domain, m=10)
approx_20 = fourier_approximation(x_domain, m=20)

plt.figure(figsize=(10, 6))
plt.plot(x_domain, f_domain, label='Initial Function: f(x) = 6x^2-2x^3', color="black")
plt.plot(x_domain, approx_5, label="Fourier Approximation m=5", linestyle='--')
plt.plot(x_domain, approx_10, label="Fourier Approximation m=10", linestyle='-.')
plt.plot(x_domain, approx_20, label="Fourier Approximation m=20", linestyle=':')
plt.title("Fourier Series Approximation of f(x) =6x^2-2x^3, m =(5, 10, 20)")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()


def u_xt(x, t, m=10):
    sum_result = np.zeros_like(x)
    a0=4
    for n in range(1, m + 1):
        an = 192*((-1)**n -1)/(n**4 * np.pi**4)
        cos_term = np.cos(np.sqrt(5) * np.pi * n * t/2)
        cos_term2 = np.cos(n * np.pi * x/2)
        sum_result += an * cos_term * cos_term2
    return a0+sum_result

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