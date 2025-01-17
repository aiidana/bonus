import numpy as np
import matplotlib.pyplot as plt

def analytical(x: float):
    return (-x**4 /12 - 5*x**3 /6 + 17*x/6 +1)

def initial(x: float):
    return - x**2 - 5 * x

n = 100  # number of steps
dx = 1 / n  
xlist = [i * dx for i in range(n + 1)]
A = 1 / dx / dx  
B = -2 / dx / dx  
C = 1 / dx / dx  
P = np.zeros(n + 1)  

alphas = [0]  
betas = [1] 

# Sweep method from left to right
for i in range(n):
    alphas.append(-A / (B + C * alphas[i]))
    betas.append((initial(i * dx) - C * betas[i]) / (B + C * alphas[i]))


P[n] = betas[n] / (1 - alphas[n])


for i in range(n - 1, -1, -1):
    P[i] = alphas[i + 1] * P[i + 1] + betas[i + 1]


analytical_solution = []
max_error = 0 

for i in range(len(xlist)):
    analytical_solution.append(analytical(xlist[i]))
    max_error = max(max_error, abs(P[i] - analytical_solution[i]))  # Update max_error


print(f"Maximal error is {max_error}")

# Plot the results
plt.plot(xlist, analytical_solution, label="Analytical", color="BLUE")
plt.plot(xlist, P, label="Numerical", color="YELLOW")
plt.grid()
plt.legend()
plt.show()