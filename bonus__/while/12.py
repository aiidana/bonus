max1=0
max2=0
while True:
    n=int(input())
    if n==0:
        break
    if max1<n:
        max2 = max1
        max1 = n
    elif max2<n:
       max2=n
print(max2)

