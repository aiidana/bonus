import numpy as np
import matplotlib.pyplot as plt

def analytical(x: float):
    return (1/(np.pi**2+1)**2)*(-np.exp(x)*np.sin(np.pi*x) +2*np.exp(x)*np.pi*np.cos(np.pi*x) +np.exp(x)*np.pi**2 *np.sin(np.pi *x)) -(np.pi*x/(np.pi**2 +1)) +1 +((2*np.exp(1)*np.pi +np.pi**3 +np.pi)/(np.pi**2 +1)**2)
def initial(x: float):
    return -np.exp(x) *np.sin(np.pi * x)

n=100
dx=1/n
xlist=[i*dx for i in range(n+1)]
A=1 / dx / dx 
B=-2 / dx / dx 
C=1 / dx / dx 
P=np.zeros(n+1)
alphas=[1]
betas=[0]

for i in range(n):
    alphas.append(-A/(B+C*alphas[i]))
    betas.append((initial(i*dx)-C*betas[i])/(B+C*alphas[i]))
P[n]=1
for i in range(n-1,-1,-1):
    P[i]=alphas[i+1]*P[i+1] +betas[i+1]

analytical_solution=[]
max_error=0
for i in range(len(xlist)):
    analytical_solution.append(analytical(i*dx))
    max_error=max(max_error,abs(P[i]-analytical_solution[i]))
print(f'Maximal error is {max_error}')
plt.plot(xlist,analytical_solution,label="Analytical",color='BLACK')
plt.plot(xlist,P,label="Numerical",color="red")
plt.grid()
plt.legend()
plt.show()