a,b= int(input()), int(input())
c,d=int(input()), int(input())

if (a-c==-1 or a-c==1) and(b-d==2 or b-d == -2):
    print('YES')
elif (a-c==-2 or a-c==2) and (b-d==1 or b-d==-1):
    print("YES")
else:
    print("NO")