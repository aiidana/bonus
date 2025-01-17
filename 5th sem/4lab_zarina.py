 
# import numpy as np 
# import matplotlib.pyplot as plt 
 
# # Аналитическое решение 
# def analytical(x: float, t: float, m: int): 
#     analytical = 0 
#     a0 = 3 / 5 
#     for i in range(1, m + 1): 
#         an = (-48 * ((-1)**i + 2) /(i ** 4 * np.pi **4)) * np.exp(-i ** 2 * np.pi ** 2 * t) * np.cos(i * np.pi * x) 
#         analytical += an 
#     return a0 + analytical 
 
# # Начальное условие 
# def initial(x): 
#     return 3 * x ** 4 - 8 * x ** 3 + 6 * x ** 2 
 
# # Параметры задачи 
# n = 100  # Число точек сетки 
# m = 10   # Количество членов в аналитическом решении 
# dx = 1 / n 
# dt = 0.0001 
# iter_limit = 10  # Количество итераций 
 
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

# import numpy as np
# import matplotlib.pyplot as plt


# def analytical(x: float, t: float, m: int):
#     analytical = 0
#     a0 = 3 / 5
#     for i in range(1, m + 1):
#         an = (-48 * ((-1)**i + 2) / (i ** 4 * np.pi ** 4)) * np.exp(-i ** 2 * np.pi ** 2 * t) * np.cos(i * np.pi * x)
#         analytical += an
#     return a0 + analytical


# def initial(x):
#     return 3 * x ** 4 - 8 * x ** 3 + 6 * x ** 2

# # Parameters
# n = 100 
# m = 10   
# dx = 1 / n
# dt = 0.00001
# iter_limit = 10 #100,1000


# xlist = [i * dx for i in range(n + 1)]
# u_n = [initial(x) for x in xlist]
# u_n1 = np.zeros(n + 1)  
# itt = 0

# # Main iteration loop
# while itt < iter_limit:
    
#     A, B, C, D = [], [], [], []
#     for i in range(1, n):  
#         A.append(-1 / dx / dx)
#         B.append(1 / dt + 2 / dx / dx)
#         C.append(-1 / dx / dx)
#         D.append(u_n[i] / dt)
    
    
#     alphan = [0]
#     betan = [0]
    
#     for i in range(1, n):
#         denom = B[i-1] + C[i-1] * alphan[i-1]
#         alphan.append(-A[i-1] / denom)
#         betan.append((D[i-1] - C[i-1] * betan[i-1]) / denom)

#     # u_n1[n] = (betan[n] / (1 - alphan[n]))
#     u_n1[0] = 0  # Left boundary
#     u_n1[n] = 0  # Right boundary

#     # Back-substitution for numerical solution at interior points
#     for i in range(n-1, 0, -1):
#         u_n1[i] = alphan[i] * u_n1[i + 1] + betan[i]

#     # Update old solution with new solution
#     u_n = u_n1.copy()
    
#     itt += 1

# # Analytical solution after final time iteration
# anall_solution = [analytical(x, dt * iter_limit, m) for x in xlist]

# # Plotting the results
# plt.plot(xlist, u_n, label=f"Numerical solution (iter={iter_limit})")
# plt.plot(xlist, anall_solution, '--', label=f"Analytical solution (iter={iter_limit})")
# plt.xlabel("x")
# plt.ylabel("u(x)")
# plt.legend()
# plt.grid()
# plt.title(f" {iter_limit}")
# plt.show()
