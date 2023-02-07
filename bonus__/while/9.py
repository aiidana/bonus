max=0
i=0
ind=-1
while True:
    n=int(input())
    if n==0:
        break
    if max<n:
        max=n
        ind=i
    i+=1
print(ind)