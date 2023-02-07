def fibonachi(n):
    if n==1 or n==2:
        return 1
    elif n==0:
        return 0
    else:
        return fibonachi(n-1)+fibonachi(n-2)

print(fibonachi(int(input())))