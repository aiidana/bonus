import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def calculate_Am(n):
    an = 8 / (np.pi * ((2 * n - 1) ** 2))
    return an

def calculate_u(x, t, m):
    Am = calculate_Am(m)
    return Am * (np.e**(-((2*m - 1)**2)*t)/2) * np.cos((2*m -1)*x/2)

x_values = np.linspace(0, np.pi, 100)
t_values = [0.1, 0.5, 1.0]  
m_values = [1,2,3,4,5,6,7,8,9]     

plt.figure(figsize=(12, 8))

for t in t_values:
    for m in m_values:
        um = calculate_u(x_values, t, m)
        plt.plot(x_values, um, label=f"")

plt.xlabel('x')
plt.ylabel('u_m(x, t)')
plt.title('')
plt.legend()
plt.grid(True)
plt.show()
# x_values = np.arange(0.01, np.pi, 0.05)
# t_values = np.arange(0, np.pi, 0.05)
# T, X = np.meshgrid(t_values, x_values)


# m = 2
# um_values = calculate_u(X, T, m)

# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, T, um_values, cmap='viridis')
# ax.set_title('Mesh Plot of u_m(x, t) ')
# ax.set_xlabel('x')
# ax.set_ylabel('t')
# ax.set_zlabel('u_m(x, t)')
# ax.set_xlim([0, np.pi])
# ax.set_ylim([0, 3])


# ax.view_init(elev=30, azim=210)  

# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# def f(x: float, t: float, iteration: int) -> float:
#     summ = 0
#     for n in range(1, iteration + 1):
#         summ += (8/(np.pi*(2*n -1)**2)) * (np.e**(-((2*n - 1)**2)*t)/2) * np.cos((2*n -1)*x/2)
#     return summ

# t = np.arange(0.01, np.pi, 0.05)
# x = np.arange(0, np.pi, 0.05)
# X, T = np.meshgrid(x, t)
# U = f(X, T, 2)

# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")
# ax.plot_surface(X, T, U, cmap='viridis')
# ax.set_xlabel('x')
# ax.set_ylabel('t')
# ax.set_zlabel('u(x,t) ')
# ax.set_title('Heat equation')
# plt.show()



# import numpy as np
# import matplotlib.pyplot as plt

# def f(x: float, t: float, iteration: int) -> float:
#     summ = 0
#     for n in range(1, iteration + 1):
#         summ += (8/(np.pi*(2*n -1)**2)) * (np.e**(-((2*n - 1)**2)*t)/2) * np.cos((2*n -1)*x/2)
#     return summ

# x = np.arange(0, np.pi, 0.05)
# for t in np.arange(0.1, 0.6, 0.1):
#     plt.plot(x, f(x, t, 20),
#              label=f"u(x,t) for t = {round(t,1)}", linewidth=1.4)

# x_mid = (np.min(x) + np.max(x)) / 2
# y_mid = (np.min(f(x, 0.1, 20)) + np.max(f(x, 0.1, 20))) / 2

# plt.legend()
# plt.axvline(x_mid, color="black", linewidth=0.6, linestyle="--")
# plt.axhline(y_mid, color="black", linewidth=0.6, linestyle="--")
# plt.text(x_mid, np.min(f(x, 0.1, 20)) - 0.2, 'X axis', ha='center', va='top')
# plt.text(np.min(x) - 0.5, y_mid, 'u(x,t) axis', ha='right', va='center',rotation = 90)

# plt.show()

#13 for 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x: float, t: float, iteration: int) -> float:
    summ = 0
    for n in range(1, iteration + 1):
        summ += (8 / (np.pi * (2 * n - 1) ** 2)) * (np.e ** (-((2 * n - 1) ** 2) * t) / 2) * np.cos((2 * n - 1) * x / 2)
    return summ

x_values = np.linspace(0, np.pi, 100)
t_values = [0.1, 0.5, 1.0]  
m_values = [1,2,3,4,5,6,7,8,9]     

plt.figure(figsize=(12, 8))

for t in t_values:
    for m in m_values:
        um = f(x_values, t, m)
        plt.plot(x_values, um, label=f"")

plt.xlabel('x')
plt.ylabel('u_m(x, t)')
plt.title(' ')
plt.legend()
plt.grid(True)
plt.show()
# x_values = np.linspace(0, np.pi, 100)
# t_values = np.linspace(0, 2, 100) 
# m_value = 1     

# X, T = np.meshgrid(x_values, t_values)  

# um = f(X, T, m_value)

# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, T, um, cmap='viridis')
# ax.set_title(f'u_m(x, t) for m={m_value}')
# ax.set_xlabel('x')
# ax.set_ylabel('t')
# ax.set_zlabel('u_m(x, t)')

# plt.show()
# t = np.arange(0.01, np.pi, 0.05)
# x = np.arange(0, np.pi, 0.05)
# X, T = np.meshgrid(x, t)

# fig = plt.figure(figsize=(12, 8))

# for i, iteration in enumerate([1, 2, 3, 4]):
#     ax = fig.add_subplot(2, 2, i + 1, projection="3d")
#     U = f(X, T, 2)
#     surf = ax.plot_surface(X, T, U, cmap='viridis')
#     ax.set_xlabel('x')
#     ax.set_ylabel('t')
#     ax.set_zlabel('u(x,t)')
#     ax.set_title(f'')

# fig.tight_layout()
# plt.show()
