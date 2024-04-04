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

# # Удаляем шкалу температуры (цветовую шкалу)
# ax.grid(False)

# plt.show()
import numpy as np
import matplotlib.pyplot as plt

def f(x: float, t: float, iteration: int) -> float:
    summ = 0
    for n in range(1, iteration + 1):
        summ += (8/(np.pi*(2*n -1)**2)) * (np.e**(-((2*n - 1)**2)*t)/2) * np.cos((2*n -1)*x/2)
    return summ

x = np.arange(0, np.pi, 0.05)
for t in np.arange(0.1, 0.6, 0.1):
    plt.plot(x, f(x, t, 20),
             label=f"u(x,t) for t = {round(t,1)}", linewidth=1.4)

x_mid = (np.min(x) + np.max(x)) / 2
y_mid = (np.min(f(x, 0.1, 20)) + np.max(f(x, 0.1, 20))) / 2

plt.legend()
plt.axvline(x_mid, color="black", linewidth=0.6, linestyle="--")
plt.axhline(y_mid, color="black", linewidth=0.6, linestyle="--")
plt.text(x_mid, np.min(f(x, 0.1, 20)) - 0.2, 'X axis', ha='center', va='top')
plt.text(np.min(x) - 0.5, y_mid, 'u(x,t) axis', ha='right', va='center',rotation = 90)

plt.show()