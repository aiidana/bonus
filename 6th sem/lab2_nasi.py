# import numpy as np
# import matplotlib.pyplot as plt
# #Analytical solution - formal solution fourier series
# def analytical(x:float, t:float, m: int):
#    analytical = 0
#    for i in range (1, m+1):
       
#        analytical += (324/((i**3)*np.pi**3))*(-1)**i
#    return analytical
# # Function for the initial condition
# def initial(x):
#    return x*(9-x**2)
# #number of steps
# n = 100
# m = 10
# #Step sizes for the space and time
# dx =  2 / n
# dt = 0.0001
# # to save all values of un and u^n+1
# xlist = [i * dx for i in range(n)]
# u_n = [initial(x) for x in xlist]
# u_n1 = np.zeros(n)
# iter = 0 #counter of iterations
# analytical_solution = []
# j = 0
# while iter <= 10:
#    diff = 0
#    for i in range(1, n-1):
#    #your approximation
#        u_n1[i] = u_n[i] + 4*(dt/dx/dx) * (u_n[i+1] - 2* u_n[i] + u_n[i-1])
#    u_n1[0] = 0 #Dirichlet condition
#    u_n1[n-1] = 0 #Dirichlet condition
#    u_n = u_n1.copy()
#    iter += 1
# for i in range(n):
#    analytical_solution.append(analytical(i * dx, (j - 1) * dt, m))
# plt.plot(xlist, u_n, label="Numerical solution")
# plt.plot(xlist, analytical_solution, color='pink', label="Analytical solution")
# plt.grid()
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# # Аналитическое решение
# def analytical(x: float, t: float, m: int):
#     analytical_value = 0
#     for i in range(1, m + 1):
#         an = (128 / (((2 * i - 1)**3) * (np.pi**3))) * np.exp(-(2
# * i - 1)**2 * np.pi**2 *5*t/16) *np.sin((2 * i - 1) * np.pi* x / 4)

#         analytical_value += an
#     return analytical_value

# # Начальное условие
# def initial(x):
#     return x * (4 - x)

# # Параметры
# n = 100
# m = 10
# dx = 2 / n
# dt = 0.00001
# xlist = [i * dx for i in range(n)]

# # Начальная инициализация
# u_n = [initial(x) for x in xlist]
# u_n1 = np.zeros(n)

# # Итерации
# iter = 0
# max_iter = 1000

# # Вычисление численного решения
# while iter <= 1000:
#     for i in range(1, n - 1):
#         u_n1[i] = u_n[i] + (5 * dt / dx / dx) * (u_n[i + 1] - 2 * u_n[i] + u_n[i - 1])  # исправлено i1 на i - 1
    
#     # Граничные условия
#     u_n1[0] = 0
#     u_n1[n - 1] = u_n1[n - 2]

#     u_n = u_n1.copy()
#     iter += 1

# # Аналитическое решение
# anall_solution = [analytical(xi, iter * dt, m) for xi in xlist]

# # Построение графика
# plt.plot(xlist, u_n, label="Numerical solution")
# plt.plot(xlist, anall_solution, color='pink', label="Analytical solution")
# plt.xlabel('x')
# plt.ylabel('u(x,t)')
# plt.legend()
# plt.grid(True)
# plt.title("Numerical vs Analytical Solution")
# plt.show()
import numpy as np
import matplotlib.pyplot as plt

def analytical(x: float):
    return (1/(np.pi**2+1)**2)*(-np.exp(x)*np.sin(np.pi*x) +2*np.exp(x)*np.pi*np.cos(np.pi*x) +np.exp(x)*np.pi**2 *np.sin(np.pi *x)) -(np.pi*x/(np.pi**2 +1)) +1 +((2*np.exp(1)*np.pi +np.pi**3 +np.pi)/(np.pi**2 +1)**2)
def initial(x: float):
    return np.exp(x) *np.sin(np.pi * x)

n = 100  
dx = 1/n

x_list = np.linspace(0, 1, n+1)  

A = -1 / (12 * dx ** 2)
B = 4 / (3 * dx**2)
C = -5 / (2 * dx ** 2)
D = 4 / (3 * dx**2)
E = -1 / (12 * dx ** 2)

u = np.zeros(n+1) 

alpha_2 = - B / (C+D)
beta_2 = -A / (C+D)
gamma_2 = initial(dx) / (C+D)

alphas = [1, alpha_2]
betas = [0, beta_2]
gammas = [0, gamma_2]

H = [-initial(x) for x in x_list]

for i in range(1, n):
    denominator = C + D * alphas[i] + alphas[i-1] * E * alphas[i] + E * betas[i-1]
    alphas.append(-(B + D * betas[i] + E * alphas[i-1] * betas[i]) / denominator)
    betas.append(-A / denominator)
    gammas.append((H[i] - D * gammas[i] - E * alphas[i-1] * gammas[i] - E * gammas[i-1]) / denominator)


u[n] = 1
u[n-1] = alphas[n] + gammas[n]


for i in range(n-2, 1, -1):
    u[i] = alphas[i+1] * u[i+1] + betas[i+1] * u[i+2] + gammas[i+1]


analytical_solution = [analytical(x) for x in x_list]


plt.plot(x_list, analytical_solution, label="Analytical", color="blue")
plt.plot(x_list, u, label="Numerical", color="red", linestyle="dashed")


plt.grid()
plt.legend()
plt.show()

#lab3
import numpy as np
import matplotlib.pyplot as plt

def analytical(x: float):
    return -np.exp(x)*np.sin(x)/2 +(np.exp(1)*np.sin(1)/2 -1)*x +1
def initial(x: float):
    return -np.exp(x) *np.cos( x)


n = 200
dx = 1 / n
xlist = [i * dx for i in range(n+1)]  

A = -1 / (12 * dx**2)  
B = 4 / (3 * dx**2) 
C = -5 / (2 * dx**2)  
D = 4 / (3 * dx**2)
E = -1 / (12 * dx**2)
P = np.zeros(n+1)  

alphas = [0]  
betas = [0]  
gamma = [1] 

alphas.append(-(B + D*betas[0])/(C + D*alphas[0]))
betas.append(-(A)/(C + D*alphas[0]))
gamma.append((initial(0) - D*gamma[0])/(C + D*alphas[0]))

for i in range(1, n):  # sweep method from left to right
    coef = C + D*alphas[i] + E*alphas[i-1]*alphas[i] + E*betas[i-1]
    alphas.append(-(B + D*betas[i] + E*alphas[i-1]*betas[i])/coef)
    betas.append(-A / coef)
    gamma.append((initial(i * dx) - D*gamma[i] - E*alphas[i-1]*gamma[i] - E*gamma[i-1])/coef)

P[n] = 0
P[n-1] =gamma[n] 

for i in range(n-2, -1, -1):
    P[i] = alphas[i+1] * P[i+1] + betas[i+1]*P[i+2] + gamma[i+1]  

analytical_solution = []
max_error = 0
for i in range(len(xlist)):
    analytical_solution.append(analytical(i * dx))  
    max_error = max(max_error, abs(P[i] - analytical_solution[i]))
print(f"Maximal error is {max_error}")

plt.plot(xlist, analytical_solution, label="Analytical", color="black")
plt.plot(xlist, P, label="Numerical", color="yellow")
plt.grid()
plt.legend()
plt.show()