import numpy as np
import time
import matplotlib.pyplot as plt
import random

start = time.time()
total_iteration = 0
while_iteration = 0

# Reading txt file and finding pairwise distance for each pair of cities
data = open('qa194.txt','r')

cities_x = np.array([],dtype = float)
cities_y = np.array([],dtype = float)

for lines in data:
    if lines[0] in '1234567890':
        x_y  = lines.split()
        cities_x = np.append(cities_x,float(x_y[1]))
        cities_y = np.append(cities_y,float(x_y[2][:-1]))

distance = np.array([[0 for i in range(len(cities_x))] for j in range(len(cities_y))],dtype = float)

for i in range(len(cities_x)):
    for j in range(len(cities_y)):
        distance[i][j] = np.sqrt((cities_x[i]-cities_x[j])**2+(cities_y[i]-cities_y[j])**2) 

# Function for calculating length of the route
def length(arr):
    output = 0
    for i in range(len(arr)-1):
        output+=distance[arr[i]-1][arr[i+1]-1]  # Расстояние между соседними городами
    output+=distance[arr[-1]-1][arr[0]-1]
    return output

# Function for reversing
def reversing(route):
    new_route = route[:]
    i, j = sorted(random.sample(range(len(route)), 2))
    new_route[i:j + 1] = list(reversed(new_route[i:j + 1]))
    return new_route

# Initializing variables
initial_route = list(range(1, len(cities_x) + 1))
random.shuffle(initial_route) 

Temperature = 100
Rate_of_change = 0.9
Num_of_iteration = len(cities_x)**2 # Количество итераций на каждой температуре
epsilon = 0.1

old_route = initial_route[:]
cur_score = length(old_route)

new_route = initial_route[:]
new_score = length(new_route)

best_route_ever = initial_route[:]
best_score_ever = length(best_route_ever)
print(length(initial_route))


Xx = []
Yy = []
for i in range(len(cities_x)):
    Xx.append(cities_x[initial_route[i]-1])
    Yy.append(cities_y[initial_route[i]-1])
Xx.append(cities_x[0])
Yy.append(cities_y[0])

plt.plot(Yy,Xx)
plt.show()


# Main body of fucntion
while Temperature > epsilon:
    for iter in range(Num_of_iteration):
        if iter % 10000 == 0:
            print(iter," ",best_score_ever," ",cur_score, " ", Temperature)
        total_iteration+=1
        new_route = reversing(old_route)        
        new_score = length(new_route)
        
        if new_score <= cur_score or np.exp(-np.abs(new_score-cur_score)/Temperature)>np.random.random():
            old_route = new_route
            cur_score = new_score

        if cur_score<=best_score_ever:
            best_route_ever = old_route
            best_score_ever = length(old_route)        

    Temperature*=Rate_of_change

print(time.time()/60-start/60)
print(Temperature)
print()
print(old_route)
print(cur_score)
print()
print(new_route)
print(new_score)
print()
print(best_route_ever)
print(best_score_ever)
print()
