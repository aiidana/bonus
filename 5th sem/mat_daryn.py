import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return 7 * np.exp(4 * x) - 3 * y

def analytical(x):
    return np.exp(4 * x) + np.exp(-3 * x)


def euler_method(f, x0, y0, h, xn):
    n_steps = int((xn - x0) / h)
    x_values = np.arange(x0, xn + h, h)
    y_values = [y0]
    
    for i in range(n_steps):
        y_next = y0 + h * f(x0, y0)
        y_values.append(y_next)
        x0 += h
        y0 = y_next
    
    return x_values, y_values


# x0, xn, y0, h = 0, 1, 2, 0.1
# x0, xn, y0, h = 0, 1, 2, 0.05
x0, xn, y0, h = 0, 1, 2, 0.025
x_num, y_num = euler_method(f, x0, y0, h, xn)

# Get the analytical solution
x_ana = np.linspace(x0, xn, 100)
y_ana = analytical(x_ana)


plt.figure(figsize=(9, 5)) 
plt.plot(x_num, y_num, 'ms-', label='Euler Method (Numerical)', markersize=7, markerfacecolor='cyan', linewidth=2)
plt.plot(x_ana, y_ana, 'b-.', label='Analytical Solution', linewidth=2.5)


plt.title('comparison of Eulers Method and Analytical solution', fontsize=16, color='black')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)

plt.xticks(np.arange(x0, xn + 0.1, 0.1), fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle=':', linewidth=1, color='black')


plt.legend(loc='upper left', fontsize=10)
plt.show()
