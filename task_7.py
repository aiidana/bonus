a=int(input())
b=int(input())
c=int(input())
if a%2==1:
    a+=1
if b%2==1:
    b+=1
if c%2==1:
    c+=1

a/=2
b/=2
c/=2

z=a+b+c
print(z)