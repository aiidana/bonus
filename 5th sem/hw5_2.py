import numpy as np
import matplotlib.pyplot as plt

# Constants
pi = 3.14159265359
n = 51
dx = 1.0 / (n - 1)
dt = 0.5 * dx * dx
eps = 1e-9

# Arrays
oldP = np.zeros(n)
newP = np.zeros(n)
f = np.zeros(n)
d = np.zeros(n)
alfa = np.zeros(n)
betta = np.zeros(n)

# Initial condition
for i in range(n):
    oldP[i] = np.sin(pi * i * dx)

# Boundary conditions
for i in range(n):
    f[i] = 0.0

a = -1.0 / (dx * dx)
b = 1.0 / dt + 2.0 / (dx * dx)
c = -1.0 / (dx * dx)


iter = 0
max_diff = 0.0

while True:
    
    for i in range(1, n-1):
        d[i] = oldP[i] / dt + f[i]

    alfa[0] = 0.0
    betta[0] = 0.0

    for i in range(1, n-1):
        alfa[i] = -c / (b + a * alfa[i - 1])
        betta[i] = (d[i] - a * betta[i - 1]) / (b + a * alfa[i - 1])

    
    newP[n-1] = 0.0
    for i in range(n-2, 0, -1):
        newP[i] = alfa[i] * newP[i + 1] + betta[i]

    
    max_diff = 0.0
    for i in range(n):
        max_diff = max(max_diff, abs(newP[i] - oldP[i]))

    for i in range(n):
        oldP[i] = newP[i]

    iter += 1

    
    if max_diff < eps:
        break


print(f"Number of iterations: {iter}")
with open("out.dat", "w") as fout:
    fout.write("VARIABLES = \"X\",\"P\",\"True\"\n")
    fout.write(f"ZONE I={n}, F=POINT\n")
    for i in range(n):
        fout.write(f"{i*dx}\t{newP[i]}\t{np.exp(-pi * pi * dt * iter) * np.sin(pi * i * dx)}\n")


x_values = np.linspace(0, 1, n)
analytical_solution = np.exp(-pi * pi * dt * iter) * np.sin(pi * x_values)

plt.plot(x_values, newP, label='Numerical Solution')
plt.plot(x_values, analytical_solution, label='Analytical Solution', linestyle='--')
plt.xlabel('x')
plt.ylabel('u')
plt.title('Numerical vs Analytical Solution')
plt.legend()
plt.grid(True)
plt.show()
