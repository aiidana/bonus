

import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return (y**2 + x*y - x**2)/x**2

def analytical(x):
    return x * (1 + x**2 / 3) / (1 - x**2 / 3)


def eulermethod(f, x0, y0, h, xi):
    x_values = [x0]
    y_values = [y0]
    
    n = int((xi - x0) / h)  # Number of steps
    for i in range(n):
        slope = f(x0, y0)
        yn = y0 + h * slope
        y0 = yn
        x0 = x0 + h
        y_values.append(y0)
        x_values.append(x0)
    
    return x_values, y_values


# x0, xi, y0, h = 1, 1.5, 2, 0.05
# x0, xi, y0, h = 1, 1.5, 2, 0.025
x0, xi, y0, h = 1, 1.5, 2, 0.0125
x_num, y_num = eulermethod(f, x0, y0, h, xi)


x_ana = np.linspace(x0, xi, 100)
y_ana = analytical(x_ana)


plt.plot(x_num, y_num, 'bo-', label='Euler method (numerical)', markersize=4)
plt.plot(x_ana, y_ana, 'r--', label='Analytical solution', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Numerical vs Analytical Solution')
plt.legend()
plt.grid(True)
plt.show()



