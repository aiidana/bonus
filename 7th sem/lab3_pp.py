import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("result.dat", skiprows=2)
x = data[:, 0]
u_explicit = data[:, 1]
u_implicit = data[:, 2]
u_analytic = data[:, 3]


fig, axs = plt.subplots(2, 2, figsize=(10, 10))


axs[0, 0].plot(x, u_explicit, color='blue', linestyle='--', linewidth=2, label='Explicit')
axs[0, 0].set_title("Explicit Scheme", fontsize=14)
axs[0, 0].grid(True)
axs[0, 0].set_xlabel("x", fontsize=12)
axs[0, 0].set_ylabel("u(x, T)", fontsize=12)
axs[0, 0].legend()


axs[0, 1].plot(x, u_implicit, color='orange', linestyle='-', linewidth=2, label='Implicit')
axs[0, 1].set_title("Implicit Scheme", fontsize=14)
axs[0, 1].grid(True)
axs[0, 1].set_xlabel("x", fontsize=12)
axs[0, 1].set_ylabel("u(x, T)", fontsize=12)
axs[0, 1].legend()


axs[1, 0].plot(x, u_analytic, color='black', linestyle='-', linewidth=2, label='Analytic')
axs[1, 0].set_title("Analytic Solution", fontsize=14)
axs[1, 0].grid(True)
axs[1, 0].set_xlabel("x", fontsize=12)
axs[1, 0].set_ylabel("u(x, T)", fontsize=12)
axs[1, 0].legend()


fig.delaxes(axs[1,1])

plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 6))
plt.plot(x, u_explicit, color='blue', linestyle='--', linewidth=2, label='Explicit')
plt.plot(x, u_implicit, color='orange', linestyle='-', linewidth=2, label='Implicit')
plt.plot(x, u_analytic, color='black', linestyle='-', linewidth=2, label='Analytic')

plt.xlabel("x", fontsize=14)
plt.ylabel("u(x, T)", fontsize=14)
plt.title("Comparison of schemes", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()