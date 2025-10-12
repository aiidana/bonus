import hashlib
import time


def hash_brute(name, leading_zeros):
    iteration = 1
    prefix = '0' * leading_zeros
    while not (hash_ := hashlib.sha256(f'{name}{iteration}'.encode()).hexdigest()).startswith(prefix):
        iteration += 1
    return name, hash_, iteration


if __name__ == '__main__':
    for zeroes in [5, 6, 7, 8]:
        start_time = time.perf_counter()

        namey, hashe, itera = hash_brute('Kebab', zeroes)

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        print(f'Task: {zeroes}')
        print(f'Nonce: {namey}{itera}\nFound: {hashe}\nIteration: {itera}\nTime: {execution_time}\n')

"""
Nonce: Kebab45528394
Found: 0000000172436cbad24a87853c07614197a5a4d89681c44b762015331fd0e23f
Iteration: 45528394
"""