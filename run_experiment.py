import multiprocessing as mp
import time
import os

# TODO: Create permutation of cli parameters

if __name__ == "__main__":
    with mp.Pool(processes=mp.cpu_count()//2, maxtasksperchild=1) as p:
        result = p.map(os.system, ["echo 'test_1'; sleep 5", "echo 'test_2'; sleep 5", "echo 'test_3'; sleep 5"])
        try:
            print(result)
        except TimeoutError:
            print("We lacked patience and got a multiprocessing.TimeoutError")