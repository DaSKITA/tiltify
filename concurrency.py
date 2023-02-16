import concurrent.futures
import math
import random
import time

PRIMES = [
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419]

def is_prime(n):
    chosen = random.choice([1, 5])
    print(f"Wait {chosen} seconds")
    time.sleep(chosen)
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True


def main():
    futures = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for number in PRIMES:
            prime_future = executor.submit(is_prime, number)
            futures.append(prime_future)
        print("Whay!")

        for fut in futures:
            if fut.done():
                print(f"Outcome is: {fut.result()}")
            else:
                futures.append(fut)


if __name__ == '__main__':
    main()
