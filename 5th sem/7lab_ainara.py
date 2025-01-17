import matplotlib.pyplot as plt


n = 100  
dt = 0.001
dx = dy = 1 / n  
itt = 0  

P = []
for i in range(n):
    Pr = []
    for j in range(n):
        
        if i == n - 1 and (40 <= j <= 60):
            Pr.append(1)  
        else:
            Pr.append(0)  
    P.append(Pr)


xlist = [i * dx for i in range(n)]
ylist = [j * dy for j in range(n)]

# Итерационный процесс
while True:
    Pn = []
    diff = 0  
    for i in range(n):
        Pr = []
        for j in range(n):
           
            if i == n - 1 and (40 <= j <= 60):
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
        print(f"Итерация: {itt}")
        plt.title(f"Итерация: {itt}")
        plt.contourf(xlist, ylist, P, levels=20, cmap="hot")
        plt.colorbar()
        plt.show()

    
    if diff < 0.00001:
        break


print("Количество итераций:", itt)
plt.contourf(xlist, ylist, P, levels=20, cmap="hot")
plt.colorbar()
plt.show()