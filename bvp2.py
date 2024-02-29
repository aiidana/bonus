# def multiplication_table(n):
#     for i in range(1, n+1):
#         for j in range(1, n+1):
#             print(f"{i} * {j} = {i*j}")
#         print() 


# table_size = 10
# multiplication_table(table_size)
def multiplication_table(n):
    for i in range(1, n+1):
        row = [str(i*j) for j in range(1, n+1)]
        print(" ".join(row))


table_size = 10
multiplication_table(table_size)

