import numpy as np
import matplotlib.pyplot as plt

num = np.loadtxt("num.txt")
ana = np.loadtxt("analytical.txt")

idx = np.argsort(num[:,0])
x = num[idx,0]
p_num = num[idx,1]
p_ana = ana[idx,1]

plt.figure(figsize=(8,5))
plt.plot(x, p_num, 'bo-', label='Численное ', markersize=4)
plt.plot(x, p_ana, 'r--', label='Аналитическое', linewidth=2)
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title('релаксация (MPI)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

