# import numpy as np 
# import matplotlib.pyplot as plt 
 
# # Аналитическое решение 
# def analytical(x: float, t: float, m: int): 
#     analytical = 0 
#     for i in range(1, m + 1): 
#         an = 32 * ((12 * (-1)**(i + 1) / ((2 * i - 1)**4 * np.pi**4)) - (3 / ((2 * i - 1)**3 * np.pi**3))) * np.exp(-(2 * i - 1)**2 * t * np.pi**2 / 4) * np.sin((2 * i - 1) * np.pi * x / 2) 
#         analytical += an 
#     return analytical 
 
# def initial(x): 
#     return 3 * x**2 - 2 * x**3 
 
# n = 100  
# m = 10    
# dx = 1 / n   
# dt = 0.00001   
 
# iter_limit = 100  # Количество итераций 
 
# # Сеточная область 
# xlist = [i * dx for i in range(n + 1)] 
# u_n = [initial(x) for x in xlist] 
# u_n1 = np.zeros(n + 1)  # Инициализируем u_n1 размером n+1 
# itt = 0 
 
# # Основной цикл 
# while itt < iter_limit: 
#     # Векторные коэффициенты для системы 
#     A, B, C, D = [], [], [], [] 
#     for i in range(n): 
#         A.append(-1 / dx / dx) 
#         B.append(1 / dt + 2 / dx / dx) 
#         C.append(-1 / dx / dx) 
#         D.append(u_n[i] / dt) 
     
#     # Массивы для расчетов в схеме 
#     alphan = [0] 
#     betan = [0] 
     
#     for i in range(n): 
#         alphan.append(-A[i] / (B[i] + C[i] * alphan[i])) 
#         betan.append(D[i] - C[i] * betan[i] / (B[i] + C[i] * alphan[i])) 
     
#     # Граничные условия 
#     u_n1[n] = (betan[n] / (1 - alphan[n]))  # Задаем значение в правой границе 
 
#     # Итерации для численного решения 
#     for i in range(n-1, -1, -1): 
#         u_n1[i] = alphan[i+1] * u_n1[i+1] + betan[i+1] 
     
#     # Обновление старого решения на новое 
#     u_n = u_n1.copy()  # Чтобы не перезаписать данные в процессе вычислений 
     
#     itt += 1 
 
# # Численное решение после заданного числа итераций 
# anall_solution = [analytical(x, dt * iter_limit, m) for x in xlist] 
 
# # Построение графика 
# plt.plot(xlist, u_n, label=f"Numerical solution (iter={iter_limit})") 
# plt.plot(xlist, anall_solution, '--', label=f"Analytical solution (iter={iter_limit})") 
# plt.xlabel("x") 
# plt.ylabel("u(x)") 
# plt.legend() 
# plt.grid() 
# plt.title(f"Comparison of Numerical and Analytical Solutions at Iteration {iter_limit}") 
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Analytical solution function
def analytical(x: float, t: float, m: int):
    analytical = 0
    for i in range(1, m + 1):
        an = 32 * ((12 * (-1)**(i + 1) / ((2 * i - 1)**4 * np.pi**4)) - (3 / ((2 * i - 1)**3 * np.pi**3))) \
             * np.exp(-(2 * i - 1)**2 * t * np.pi**2 / 4) * np.sin((2 * i - 1) * np.pi * x / 2)
        analytical += an
    return analytical

def initial(x):
    return 3 * x**2 - 2 * x**3


n = 100  
m = 10    
dx = 1 / n   
dt = 0.00001   

iter_limit = 1000  # Number of iterations
# iter_limit = 100 
# iter_limit = 1000

xlist = [i * dx for i in range(n + 1)]
u_n = [initial(x) for x in xlist]
u_n1 = np.zeros(n + 1)  # Initialize u_n1 with size n+1
itt = 0


while itt < iter_limit:
    # System coefficients for the Thomas algorithm
    A, B, C, D = [], [], [], []
    for i in range(1, n):  
        A.append(-1 / dx / dx)
        B.append(1 / dt + 2 / dx / dx)
        C.append(-1 / dx / dx)
        D.append(u_n[i] / dt)
    
    # Thomas algorithm coefficients
    alphan = [0]
    betan = [0]
    
    for i in range(1, n):
        denom = B[i-1] + C[i-1] * alphan[i-1]
        alphan.append(-A[i-1] / denom)
        betan.append((D[i-1] - C[i-1] * betan[i-1]) / denom)

    # Boundary conditions at u(0, t) and u(1, t) for each time step
    u_n1[n] = 0  
    u_n1[0] = 0  

   
    for i in range(n-1, 0, -1):
        u_n1[i] = alphan[i] * u_n1[i + 1] + betan[i]

    u_n = u_n1.copy()
    
    itt += 1


anall_solution = [analytical(x, dt * iter_limit, m) for x in xlist]

# Plotting the results
plt.plot(xlist, u_n, label=f"Numerical solution (iter={iter_limit})")
plt.plot(xlist, anall_solution, '--', label=f"Analytical solution (iter={iter_limit})")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.grid()
plt.title(f"Comparison of Numerical and Analytical Solutions at Iteration {iter_limit}")
plt.show()
