import numpy as np
import matplotlib.pyplot as plt

def calculate_um(x, t, m):
    um = (8 / (((2*m - 1)**2) * np.pi**2)) * np.e**(-((2*m - 1)**2 * np.pi**2 * t) / 2) * np.cos((2*m - 1) * np.pi * x / 2)
    return um

x_values = np.linspace(0, np.pi, 100)
t_values = [0.1, 0.5, 1.0]  
m_values = [1,2,3,4,5,6,7,8,9]     

plt.figure(figsize=(12, 8))

for t in t_values:
    for m in m_values:
        um = calculate_um(x_values, t, m)
        plt.plot(x_values, um, label=f"")

plt.xlabel('x')
plt.ylabel('u_m(x, t)')
plt.title('Partial Sum u_m(x, t) for Various Values of t and m')
plt.legend()
plt.grid(True)
plt.show()
from mpl_toolkits.mplot3d import Axes3D
x_values = np.linspace(0, np.pi, 100)
t_values = np.linspace(0, 2, 100) 
m_value = 1     

X, T = np.meshgrid(x_values, t_values)  

um = calculate_um(X, T, m_value)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, um, cmap='viridis')
ax.set_title(f'u_m(x, t) for m={m_value}')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u_m(x, t)')

plt.show()
