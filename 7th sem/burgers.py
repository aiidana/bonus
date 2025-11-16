import numpy as np
import matplotlib.pyplot as plt


data = np.genfromtxt("burgers_result.csv", delimiter=",", skip_header=1)
x = data[:, 0]
num = data[:, 1]
ana = data[:, 2]
err = data[:, 3]

plt.figure(figsize=(10, 5))


plt.subplot(1, 2, 1)
plt.plot(x, ana, 'k--', label='Analytical')
plt.plot(x, num, 'b-', label='Numerical')
plt.xlabel("x")
plt.ylabel("u(x, t)")
plt.title("Burgers Equation Solution at t = 0.3")
plt.legend()
plt.grid(True)


plt.subplot(1, 2, 2)
plt.plot(x, err, 'r')
plt.xlabel("x")
plt.ylabel("Error |u_num - u_ana|")
plt.title("Error Distribution")
plt.grid(True)

plt.tight_layout()
plt.show()
