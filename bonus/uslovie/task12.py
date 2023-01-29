m,n,k=int(input()), int(input()), int(input())

if(k%m==0 or k%n==0) and k<m*n:
    print("YES")
else:
    print("NO")