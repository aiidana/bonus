# import numpy as np
# import matplotlib.pyplot as plt

# def analytical(x: float):
#     return (1/np.pi)*((4*x**2 +12*x+8)*np.sin(np.pi*x/2)/np.pi +(16*x+24))
# def initial(x: float):
#     return -np.exp(x) *np.cos( x)

# n=100
# dx=1/n
# xlist=[i*dx for i in range(n+1)]
# A=1 / dx / dx 
# B=-2 / dx / dx 
# C=1 / dx / dx 
# P=np.zeros(n+1)
# alphas=[0]
# betas=[1]

# for i in range(n):
#     alphas.append(-A/(B+C*alphas[i]))
#     betas.append((initial(i*dx)-C*betas[i])/(B+C*alphas[i]))
# P[n]=betas[n] / (1 - alphas[n])
# for i in range(n-1,-1,-1):
#     P[i]=alphas[i+1]*P[i+1] +betas[i+1]

# analytical_solution=[]
# max_error=0
# for i in range(len(xlist)):
#     analytical_solution.append(analytical(i*dx))
#     max_error=max(max_error,abs(P[i]-analytical_solution[i]))
# print(f'Maximal error is {max_error}')
# plt.plot(xlist,analytical_solution,label="Analytical",color='BLACK')
# plt.plot(xlist,P,label="Numerical",color="YELLOW")
# plt.grid()
# plt.legend()
# plt.show()
# import numpy as np
# import matplotlib.pyplot as plt

# def analytical(x: float):
#     return (1 / np.pi) * (
#         ((4 * x**2 + 12 * x + 8) * np.sin((np.pi * x) / 2)) / np.pi
#         + ((16 * x + 24) * np.cos((np.pi * x) / 2)) / np.pi**2
#         - (32 * np.sin((np.pi * x) / 2)) / np.pi**3 
#     ) + (1 / np.pi**2) * ( 
#         ((16 * x + 24) * np.cos((np.pi * x) / 2)) / np.pi
#         - (32 * np.sin((np.pi * x) / 2)) / np.pi**2
#     ) - (32 * np.sin((np.pi * x) / 2)) / np.pi**4 + x * (20 / np.pi**2) + ((np.pi**3 - 48) / np.pi**3)

# def initial(x: float):
#     return (-x**2 - 3 * x - 2) * np.sin((np.pi * x) / 2)


# n = 100
# dx = 1 / n
# xlist = [i * dx for i in range(n + 1)]

# A = 1 / dx / dx
# B = -2 / dx / dx
# C = 1 / dx / dx

# P = np.zeros(n + 1)
# alphas = [0]
# betas = [1]

# for i in range(n):
#     alphas.append(-A / (B + C * alphas[i]))
#     betas.append((initial(i * dx) - C * betas[i]) / (B + C * alphas[i]))

# P[n] = betas[n] / (1 - alphas[n])
# for i in range(n - 1, -1, -1):
#     P[i] = alphas[i + 1] * P[i + 1] + betas[i + 1]

# analytical_solution = []
# max_error = 0

# for i in range(len(xlist)):
#     analytical_solution.append(analytical(i * dx))
#     max_error = max(max_error, abs(P[i] - analytical_solution[i]))

# print(f'Maximal error is {max_error}')

# plt.plot(xlist, analytical_solution, label="Analytical", color='black')
# plt.plot(xlist, P, label="Numerical", color="yellow")
# plt.grid()
# plt.legend()
# plt.show()
import numpy as np
import matplotlib.pyplot as plt

def analytical(x: float):
    return (1/2)*np.exp(x)*np.cos(x) +(np.exp(1)*np.sin(1) +np.exp(1)*np.cos(1))*x/2 +1/2

def f(x: float):
    return -np.exp(x) * np.cos(x)

n = 100
dx = 1 / n
xlist = [i * dx for i in range(n + 1)]

A = 1 / dx / dx
B = -2 / dx / dx
C = 1 / dx / dx

P = np.zeros(n + 1)
alphas = [0]
betas = [1]

for i in range(n):
    alphas.append(-A / (B + C * alphas[i]))
    betas.append((f(i * dx) - C * betas[i]) / (B + C * alphas[i]))

P[n] = betas[n] / (1 - alphas[n])
for i in range(n - 1, -1, -1):
    P[i] = alphas[i + 1] * P[i + 1] + betas[i + 1]

analytical_solution = [analytical(i * dx) for i in range(len(xlist))]
max_error = max(abs(P[i] - analytical_solution[i]) for i in range(len(xlist)))

print(f'Maximal error is {max_error}')

plt.plot(xlist, analytical_solution, label="Analytical", color='black')
plt.plot(xlist, P, label="Numerical", color='yellow')
plt.grid()
plt.legend()
plt.show()