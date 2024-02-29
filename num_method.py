# import matplotlib.pyplot as plt
# import numpy as np
# def f(x):
#     fx=x**3+2*x-4
#     return  fx
# def g(x):
#     return 4- 2*x
# def func(x):
#     return x**3
# xx=np.linspace(-3,3,100) #100 points between -3 and 3
# y1=func(xx)
# y2=g(xx)
# plt.figure(figsize=(8, 6))
# plt.plot(xx, y1, label='f(x) = x^3', color='b')
# plt.plot(xx, y2, label='g(x) = 4 - 2x', color='r') 
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Graphs of f(x) and g(x)')
# plt.grid(True)
# plt.legend()

# # Display the plot
# plt.show()

# def bisection(a,b):
#     if f(a)*f(b)>=0:
#         return "interval does not have opposite signs"
#     while (b-a)>=0.01:
#         x=(a+b)/2
#         if f(x)==0:
#             return x
#         if f(a)*f(x)<0:
#             b=x
#         else:
#             a=x
#     return (a+b)/2
# a=1
# b=2
# root=bisection(a,b)
# print(root)

# import math
# error=[]
# rel_error=[]
# p_values=[]
# def f(x):
#     fx=x**3+(2*x)-4
#     return fx
# def fp(x):
#     fprime=3*(x**2)+2
#     return fprime
# def fixed(x0):
#     iter=0
#     max_iter=100
#     p=x0
#     step=1
#     while iter<max_iter:
#         p_values.append(p)
#         p_nxt=p-(f(p)/fp(p))
#         err=abs(p_nxt-p)
#         r_err=err/p_nxt
#         error.append(err)
#         rel_error.append(r_err)
        
#         if err<0.01:
#             break
#         p=p_nxt
#     return p_values, error, rel_error
# x0=1
# print(fixed(x0),end=' ')
# root=p_values[-1]
# print("root is:",root)




p_values = []
error = []
rel_err = []
def g(x):
    gx = x**3 + (2 * x) - 4
    return gx
def f(x):
    fx = x**3 + x - 4
    return fx
def gprime(x):
    gp = 3 * (x**2) + 2
    return gp
def phi(x):
    phix = x - (f(x) / 5)
    return phix
def check(a, b):
    x0 = a
    iter_count = 0
    max_iter = 100
    if (f(a) * f(b) < 0) and abs(gprime(a)) < 1:
        while iter_count < max_iter:
            p_values.append(x0)
            a_nxt = g(x0)
            err = abs(a_nxt - x0)
            rel_error = err / abs(a_nxt)
            error.append(err)
            rel_err.append(rel_error)
            x0 = a_nxt
            if err < 0.01:
                break
            iter_count += 1
    else:
        while iter_count < max_iter:
            p_values.append(x0)
            a_nxt = phi(x0)
            err = abs(a_nxt - x0)
            rel_error = err / abs(a_nxt)
            error.append(err)
            rel_err.append(rel_error)
            x0 = a_nxt
            if err < 0.01:
                break
            iter_count += 1

    return p_values, error, rel_err
a = 1
b = 2
result = check(a, b)
print("Approximate root:", result[0][-1])  # Print the last value in p_values as the approximate root



