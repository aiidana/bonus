cnt=0
max=0
n=-1
while True:
    n=int(input())
    if n==0:
        break
    if max<n:
        max=n
        cnt=1
    elif max==n:
        cnt+=1

print(cnt)