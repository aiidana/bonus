# import numpy as np
# import matplotlib.pyplot as plt

# K = 0.001
# b = 10
# T = 10          
# dt = 0.01       # шаг 
# N = 1000  # кол-во шаов

# t = np.linspace(0, T, N)

# # аналитикал
# y_analytical = (b/K) * t - (b/K**2) * (1 - np.exp(-K*t))

# # нумерикал
# y_num = np.zeros(N)
# y_num[0] = 0.0     
# y_num[1] = 0.0 

# #y_num[1] = 0.5 * b * dt**2 


# for n in range(1, N-1):
#     y_num[n+1] = 2*y_num[n] - y_num[n-1] - K*dt*(y_num[n] - y_num[n-1]) + b*dt**2


# plt.figure(figsize=(8,5))
# plt.plot(t, y_analytical, 'r-', label='Аналитическое решение')
# plt.plot(t, y_num, 'b--', label='Численное решение (FDM)')
# plt.xlabel('t')
# plt.ylabel('y(t)')
# plt.legend()
# plt.grid(True)
# plt.title("Сравнение аналитического и численного решения")
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
#аналитическое решение логистического уравнения
def logistic_solution(t, c, T0, lmbd):
    return (c * T0) / (np.exp(-lmbd * t) + c)
#параметры
t_max = 100
dt = 0.0001
N = int(t_max / dt) + 1

t = np.linspace(0, t_max, N) 
T = np.zeros(N)
lmbd = 0.1
T0 = 5000
T_init = 1e3
c = T_init / (T0 - T_init)

#аналитика
Y = logistic_solution(t, c, T0, lmbd)

#численное решение (Эйлер)
T[0] = T_init
for n in range(N - 1):
    T[n+1] = T[n] + dt * lmbd * T[n] * (1 - T[n] / T0)

#сравнение в начальном моменте и ошибка
print("Analytical at t=0:", Y[0])
print("Numerical at t=0:", T[0])
print(f"Max error: {np.max(np.abs(Y - T)):.6f}")


plt.figure(figsize=(8, 5))
plt.plot(t, T, "r--", linewidth=1.5, label="Numerical (Euler)")
plt.plot(t, Y, "b", linewidth=2, label="Analytical")
plt.title("Tumor Cell Proliferation Model", fontsize=14)
plt.xlabel("Time t", fontsize=12)
plt.ylabel("Tumor cells T(t)", fontsize=12)
plt.legend()
plt.grid(alpha=0.4, linestyle="--")
plt.show()