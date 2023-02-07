max=0
while True:
    n=int(input())
    if n==0:
        break
    if max<n:
        max=n
print(max)