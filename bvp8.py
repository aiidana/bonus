# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def u(x, t):
#     return (32 / np.pi) * (((np.pi * (-1)**(n+1)) / (2*n - 1)) - (2 / (2*n - 1)**2)) * (np.e**(-(2*n - 1)**2 * 9 * t / 16)) * np.cos((2*n - 1) * x / 4)

# x_values = np.linspace(0, np.pi, 100)
# t_values = np.linspace(0, 2, 100)
# n_values = [1, 2, 3, 4, 5]  # Различные значения n

# X, T = np.meshgrid(x_values, t_values)

# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# for n in n_values:
#     Z = u(X, T)
#     ax.plot_surface(X, T, Z, label=f"n={n}")

# ax.set_xlabel('x')
# ax.set_ylabel('t')
# ax.set_zlabel('u(x, t)')
# ax.set_title('Graph of u(x, t) for Various Values of n')
# ax.legend()

# plt.show()
import numpy as np
import matplotlib.pyplot as plt
def f(x: float, t: float, iteration: int) -> float:
    summ = 0
    for n in range(1, iteration + 1):
        summ +=(32 / np.pi) * (((np.pi * (-1)**(n+1)) / (2*n - 1)) - (2 / (2*n - 1)**2)) * (np.e**(-((2*n - 1)**2) * 9 * t / 16)) * np.cos((2*n - 1) * x / 4)
    return summ

t = np.arange(0.01, np.pi, 0.05)
x = np.arange(0, np.pi, 0.05)
X, T = np.meshgrid(x, t)
U = f(X, T, 2)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(X, T, U, cmap='viridis')
ax.set_xlabel('X axis')
ax.set_ylabel('T axis')
ax.set_zlabel('u(x,t) axis')
ax.set_title('')
fig.colorbar(ax.plot_surface(X, T, U, cmap='viridis'), pad=0.2)
plt.show()
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
plt.title('Partial Sum u_m(x, t) for Various Values of t and m')
plt.legend()
plt.grid(True)
plt.show()