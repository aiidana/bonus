# def fibonachi(n):
#     if n==1 or n==2:
#         return 1
#     elif n==0:
#         return 0
#     else:
#         return fibonachi(n-1)+fibonachi(n-2)

# fibonachi(int(input()))

# print(fibonachi(int(input())))

a = int(input())
if a == 0:
    print(0)
else:
    fib_prev, fib_next = 0, 1
    n = 1
    while fib_next <= a:
        if fib_next == a:
            print(n)
            break
        fib_prev, fib_next = fib_next, fib_prev + fib_next
        n += 1
    else:
        print(-1)

