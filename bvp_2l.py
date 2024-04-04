# import numpy as np
# import matplotlib.pyplot as plt


# # Function to calculate Am
# def calculate_Am(n):
#     an=8/(np.pi * (2*n -1)**2)
#     return an

# # Function to calculate u(x, t) for given m and t
# def calculate_um(x, t, n):
#     Am = calculate_Am(n)
#     return Am *  (np.e**(-((2*n - 1)**2)*t)/2) * np.cos((2*n -1)*x/2)

# # Grid points for x and t
# x_values = np.linspace(0, np.pi, 100)
# t_values = np.linspace(0, 1, 100)

# # Plot u_m(x, t) for m = 5, 10, 20
# plt.figure(figsize=(12, 6))
# for m in [5, 10, 20]:
#     um_values = np.zeros_like(x_values)
#     for i, x in enumerate(x_values):
#         um_values[i] = calculate_um(x, 0.1, m)  # Plotting for t = 0.1
    
#     plt.plot(x_values, um_values, label=f'm={m}')

# plt.title('Plot of u_m(x, t) for m=5, 10, 20')
# plt.xlabel('x')
# plt.ylabel('u_m(x, t)')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Create mesh grid for x and t
# X, T = np.meshgrid(x_values, t_values)

# # Calculate um(x, t) for m = 20
# m = 20
# um_values = calculate_um(X, T, m)

# # Plot mesh plot of um(x, t)
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, T, um_values, cmap='viridis')
# ax.set_title('Mesh Plot of u_m(x, t) for m=20')
# ax.set_xlabel('x')
# ax.set_ylabel('t')
# ax.set_zlabel('u_m(x, t)')
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Function to calculate Am
# def calculate_Am(n):
#     an = 8 / (np.pi * ((2 * n - 1) ** 2))
#     return an

# # Function to calculate u(x, t) for given m and t
# def calculate_um(x, t, m):
#     Am = calculate_Am(m)
#     return Am * (np.e**(-((2*m - 1)**2)*t)/2) * np.cos((2*m -1)*x/2)

# # Grid points for x and t
# x_values = np.arange(0.01, np.pi, 0.05)
# t_values = np.arange(0, np.pi, 0.05)

# # Create mesh grid for x and t
# T, X = np.meshgrid(t_values, x_values)


# m = 2
# um_values = calculate_um(X, T, m)

# # Plot mesh plot of um(x, t) from a different angle
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, T, um_values, cmap='viridis')
# ax.set_title('Mesh Plot of u_m(x, t) ')
# ax.set_xlabel('x')
# ax.set_ylabel('t')
# ax.set_zlabel('u_m(x, t)')
# ax.set_xlim([0, np.pi])
# ax.set_ylim([0, 3])

# # Set the elevation and azimuth angles to view from a different side
# ax.view_init(elev=30, azim=210)  # Change the viewing angles here as needed

# plt.show()



##3
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
# fig.colorbar(ax.plot_surface(X, T, U, cmap='viridis'), pad=0.2)
# plt.show()





