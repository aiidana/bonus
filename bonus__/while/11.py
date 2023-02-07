n=int(input())
a=n
i=0
while n!=0:
    n=int(input())
    if n>a:
        i+=1
    a=n
print(i)