import numpy as np
import matplotlib.pyplot as plt

def function_x(x):
    return 3*x**5 -8*x**4 +6*x**3

def fourier_approximation(x, m):
    approx_values = np.zeros_like(x)
    a0=2/5
    for n in range(1, m + 1):
        a_n = 2*(((-1)**n / (n**2 *np.pi**2)) - (24*(-1)**n /(n**4 * np.pi**4)) +(360*(-1)**n /(n**6 * np.pi**6))+(36/(n**4 * np.pi**4))-(360/(n**6 * np.pi**6)))
        approx_values += a_n * np.cos(n * np.pi * x)
    return a0+approx_values

x_domain = np.linspace(0, 1, 100)
f_domain = function_x(x_domain)


approx_5 = fourier_approximation(x_domain, m=5)
approx_10 = fourier_approximation(x_domain, m=10)
approx_20 = fourier_approximation(x_domain, m=20)

plt.figure(figsize=(10, 6))
plt.plot(x_domain, f_domain, label='Initial Function: f(x) =3x^5-8x^4+6x^3', color="black")
plt.plot(x_domain, approx_5, label="Fourier Approximation m=5", linestyle='--')
plt.plot(x_domain, approx_10, label="Fourier Approximation m=10", linestyle='-.')
plt.plot(x_domain, approx_20, label="Fourier Approximation m=20", linestyle=':')
plt.title("Fourier Series Approximation of f(x) =3x^5-8x^4+6x^3")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

def u_xt(x, t, m=10):
    sum_result = np.zeros_like(x)
    a0=2/5
    for n in range(1, m + 1):
        a_n = 2*(((-1)**n / (n**2 *np.pi**2)) - (24*(-1)**n /(n**4 * np.pi**4)) +(360*(-1)**n /(n**6 * np.pi**6))+(36/(n**4 * np.pi**4))-(360/(n**6 * np.pi**6)))
        cos_term=np.cos(n*np.pi*x)
        sum_result += a_n * np.exp(-n**2 * np.pi**2 *t) * cos_term
    return a0+ sum_result

time_values = [0, 0.01, 0.05, 0.1]
plt.figure(figsize=(10, 6))
for t in time_values:
    u_values = u_xt(x_domain, t, m=10)
    plt.plot(x_domain, u_values, label=f't={t}')
plt.title("Solution to the heat equation u(x, t)")
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.legend()
plt.grid(True)
plt.show()