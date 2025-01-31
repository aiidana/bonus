# import matplotlib.pyplot as plt

# n=100
# dt=0.001
# dx=dy=1/n
# itt=0

# P=[]
# for i in range(n):
#     Pr=[]
#     for j in range(n):
#         if(j==0 and (40<=i<=60)) or(i==n-1 and (80<=j<=99)):
#             Pr.append(1)
#         else:
#             Pr.append(0)
#     P.append(Pr)
# xlist=[i*dx for i in range(n)]
# ylist=[j*dx for j in range(n)]
# while True:
#     Pn=[]
#     diff=0
#     for i in range(n):
#         Pr=[]
#         for j in range(n):
#             if(j ==0 and (40<=i<=60))or (i==n-1 and (80<=j<=99)):
#                 Pr.append(1)
#             else:
#                 Pr.append(0)
#         Pn.append(Pr)
#     for i in range(1,n-1):
#         for j in range(1,n-1):
#             Pn[i][j]=1/4*(P[i+1][j]+P[i-1][j]+P[i][j+1]+P[i][j-1])
#             diff=max(diff,abs(Pn[i][j]-P[i][j]))
#     itt+=1
#     P=Pn
#     if itt%300==0:
#         print(itt)
#         plt.contourf(xlist,ylist,P)
#         plt.show()
#     if diff<0.00001:
#         break
# print(itt)
# plt.contourf(xlist,ylist,P)
# plt.show()




import matplotlib.pyplot as plt

n = 100
dt = 0.001
dx = dy = 1 / n
itt = 0


P = []
for i in range(n):
    Pr = []
    for j in range(n):
        
        if i == 0 and (0 <= j <= 20):
            Pr.append(1)
        
        elif j == n - 1 and (40 <= i <= 60):
            Pr.append(1)
        else:
            Pr.append(0)
    P.append(Pr)

xlist = [i * dx for i in range(n)]
ylist = [j * dx for j in range(n)]


while True:
    Pn = []
    diff = 0
    for i in range(n):
        Pr = []
        for j in range(n):
            
            if i == 0 and (0 <= j <= 20):
                Pr.append(1)
            elif j == n - 1 and (40 <= i <= 60):
                Pr.append(1)
            else:
                Pr.append(0)
        Pn.append(Pr)
    
    
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            Pn[i][j] = 1 / 4 * (P[i + 1][j] + P[i - 1][j] + P[i][j + 1] + P[i][j - 1])
            diff = max(diff, abs(Pn[i][j] - P[i][j]))
    
    itt += 1
    P = Pn
    
    
    if itt % 300 == 0:
        print(itt)
        plt.title(f"itt= {itt}")
        plt.contourf(xlist, ylist, P)
        plt.show()
    
    if diff < 0.00001:
        break

print("Количество итераций:", itt)
plt.contourf(xlist, ylist, P)
plt.show()
















