import numpy as np
import matplotlib.pyplot as plt

def analytical(x: float):
    return -np.exp(x)*np.sin(x)/2 +(np.exp(1)*np.sin(1)/2 -1)*x +1
def initial(x: float):
    return -np.exp(x) *np.cos( x)

n=100
dx=1/n
xlist=[i*dx for i in range(n+1)]
A=1 / dx / dx 
B=-2 / dx / dx 
C=1 / dx / dx 
P=np.zeros(n+1)
alphas=[0]
betas=[1]

for i in range(n):
    alphas.append(-A/(B+C*alphas[i]))
    betas.append((initial(i*dx)-C*betas[i])/(B+C*alphas[i]))
P[n]=0
for i in range(n-1,-1,-1):
    P[i]=alphas[i+1]*P[i+1] +betas[i+1]

analytical_solution=[]
max_error=0
for i in range(len(xlist)):
    analytical_solution.append(analytical(i*dx))
    max_error=max(max_error,abs(P[i]-analytical_solution[i]))
print(f'Maximal error is {max_error}')
plt.plot(xlist,analytical_solution,label="Analytical",color='BLACK')
plt.plot(xlist,P,label="Numerical",color="YELLOW")
plt.grid()
plt.legend()
plt.show()