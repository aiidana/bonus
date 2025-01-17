import numpy as np
import matplotlib.pyplot as plt

def func(x:float):
    return x*(x**3 - 2 *x**2 +1)

n=100
dx=1/n
dt=0.01
# Re=5
Re=500
# Re=500

xlist=[i * dx for i in range(n+1)]
A=[]
B=[]
C=[]
D=[]
u=[func(x) for x in xlist]
itt=0

while itt<=10:
    for i in range(n):
        A.append(u[i]/dx - 1/(Re*dx**2))
        B.append(-u[i]/dx + 1/dt + 2/(Re*dx**2))
        C.append(-1/(Re*dx**2))
        D.append(u[i]/dt)
    un=[0 for i in range(len(u))]
    alphan=[1]
    betan=[-dx]
    for i in range(n):
        alphan.append(-A[i]/(B[i]+ C[i]*alphan[i]))
        betan.append((D[i]-C[i]*betan[i])/(B[i]+ C[i]*alphan[i]))
    un[n]=(betan[n]/(1-alphan[n]))
    for i in range(n-1,-1,-1):
        un[i]=alphan[i+1]* un[i+1] + betan[i+1]
    u=un
    itt+=1

plt.plot(xlist,u,label="N")
plt.grid()
plt.legend()
plt.show()
