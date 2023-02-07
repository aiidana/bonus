a,b=int(input()), int(input())
if a%2==1:
 for i in range(a,b-1,-2):
    print(i)
else:
    for i in range(a-1, b-1,-2):
        print(i)