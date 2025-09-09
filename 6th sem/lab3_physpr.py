# import numpy as np
# import matplotlib.pyplot as plt

# def analytical(x: float):
#     return (1/(np.pi**2+1)**2)*(-np.exp(x)*np.sin(np.pi*x) +2*np.exp(x)*np.pi*np.cos(np.pi*x) +np.exp(x)*np.pi**2 *np.sin(np.pi *x)) -(np.pi*x/(np.pi**2 +1)) +1 +((2*np.exp(1)*np.pi +np.pi**3 +np.pi)/(np.pi**2 +1)**2)
# def initial(x: float):
#     return -np.exp(x) *np.sin(np.pi * x)

# n=100
# dx=1/n
# xlist=[i*dx for i in range(n+1)]
# A=1 /12/ dx / dx 
# B=16/12 / dx / dx 
# C=-30/12 / dx / dx 
# D=16/12/dx/dx
# E=-1/12/dx/dx
# P=np.zeros(n+2)
# alphas=[1]
# betas=[0]
# gammas=[0]

# alphas2=[-(B+D*betas[0])/(C+D*alphas[0])]
# betas2=[-A/(C+D*alphas[0])]
# gammas2=[-initial(0)-D*alphas[0]/(C+D*alphas[0])]

# for i in range(n):
    
#     alphas.append(-(B+D*betas[i]+E*alphas[i-1]*betas[i])/(C +D*alphas[i]+E*alphas[i-1]*alphas[i]+E*betas[i-1]))
#     betas.append(-A/(C +D*alphas[i]+E*alphas[i-1]*alphas[i]+E*betas[i-1]))
#     gammas.append((-initial(i*dx)-D*gammas[i]-E*gammas[i]*alphas[i-1]-E*gammas[i-1])/(C +D*alphas[i]+E*alphas[i-1]*alphas[i]+E*betas[i-1]))
# P[n]=1
# P[n-1]=P[n]*alphas[n]+gammas[n]
# for i in range(n-1,-1,-1):
#     P[i]=alphas[i+1]*P[i+1] +betas[i+1]*P[i+2]+gammas[i+1]

# analytical_solution=[]
# max_error=0
# for i in range(len(xlist)):
#     analytical_solution.append(analytical(i*dx))
#     max_error=max(max_error,abs(P[i]-analytical_solution[i]))
# print(f'Maximal error is {max_error}')
# plt.plot(xlist,analytical_solution,label="Analytical",color='BLACK')
# plt.plot(xlist,P,label="Numerical",color="YELLOW")
# plt.grid()
# plt.legend()
# plt.show()



import numpy as np
import matplotlib.pyplot as plt

def analytical(x: float):
    return (1/(np.pi**2+1)**2)*(-np.exp(x)*np.sin(np.pi*x) +2*np.exp(x)*np.pi*np.cos(np.pi*x) +np.exp(x)*np.pi**2 *np.sin(np.pi *x)) -(np.pi*x/(np.pi**2 +1)) +1 +((2*np.exp(1)*np.pi +np.pi**3 +np.pi)/(np.pi**2 +1)**2)
def initial(x: float):
    return -np.exp(x) *np.sin(np.pi * x)


n = 200
dx = 1 / n
xlist = [i * dx for i in range(n+1)]  

A = -1 / (12 * dx**2)  
B = 4 / (3 * dx**2) 
C = -5 / (2 * dx**2)  
D = 4 / (3 * dx**2)
E = -1 / (12 * dx**2)
P = np.zeros(n+1)  

alphas = [1]  
betas = [0]  
gamma = [0] 

alphas.append(-(B )/(C + D*alphas[0]))
betas.append(-(A)/(C + D*alphas[0]))
gamma.append((initial(0) )/(C + D*alphas[0]))

for i in range(1, n):  # sweep method from left to right
    coef = C + D*alphas[i] + E*alphas[i-1]*alphas[i] + E*betas[i-1]
    alphas.append(-(B + D*betas[i] + E*alphas[i-1]*betas[i])/coef)
    betas.append(-A / coef)
    gamma.append((initial(i * dx) - D*gamma[i] - E*alphas[i-1]*gamma[i] - E*gamma[i-1])/coef)

P[n] = 1
P[n-1] = alphas[n]  + gamma[n] 

for i in range(n-2, -1, -1):
    P[i] = alphas[i+1] * P[i+1] + betas[i+1]*P[i+2] + gamma[i+1]  

analytical_solution = []
max_error = 0
for i in range(len(xlist)):
    analytical_solution.append(analytical(i * dx))  
    max_error = max(max_error, abs(P[i] - analytical_solution[i]))
print(f"Maximal error is {max_error}")

plt.plot(xlist, analytical_solution, label="Analytical", color="black")
plt.plot(xlist, P, label="Numerical", color="red")
plt.grid()
plt.legend()
plt.show()
