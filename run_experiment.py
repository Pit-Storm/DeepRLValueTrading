import multiprocessing as mp
import time
import os
import sys
import itertools

def gen_args(baseline: bool) -> list:
    # switches intersting to manipulate
    # algo = str
    # cagr = bool         # def: False
    # cash = int          # def: 1e6
    # deterministic = bool# def: False
    # episodic = bool     # def: False
    # fee = float         # def: 1e-3
    # learn_steps = int   # def: 15e3
    # policy = str        # def: MlpLstmPolicy
    # scaling = int       # def: 1e2
    # test_eps = int      # def: 100
    # trainsampling = bool# def: False
    # val_eps = int      # def: 10
    # val_freq = int      # def: 1e3
    # yearrange = int     # def: 4

    if baseline:
        baseline = ["--algo=random","--algo=buyhold"]
        lsts = [baseline, ["--episodic"], ["--test_eps=100"]]
    else:
        cagr = ["--cagr", ""]
        episodic = ["--episodic", ""]
        learn_steps = ["--learn_steps=500000"]
        # num_stack = ["--num_stack=3", "--num_stack=0"]
        test_eps = ["--test_eps=100"]
        trainsampling = ["--trainsampling", ""]
        val_freq = ["--val_freq=2500"]

        drl = ["--algo=DDPG", "--algo=PPO", "--algo=A2C"]
        lsts = [drl, learn_steps, test_eps, val_freq, episodic, cagr, trainsampling]
    
    params = list(itertools.product(*lsts))
    return params

if __name__ == "__main__":
    # Set baseline parameter of gen_args to False to run the actual experiment
    # If it is true you generate data for baseline comparison
    param_lists = gen_args(baseline = False)
    param_strings = [" ".join(lst) for lst in param_lists]
    cmd_strings = ["python main.py " + item for item in param_strings]
    
    processes = mp.cpu_count()//2
    with mp.Pool(processes=processes, maxtasksperchild=1) as p:
        results = p.map(os.system, cmd_strings)

        broken = [str(results[i]) + ": " + cmd_strings[i] for i in range(len(results)) if results[i] != 0]
        print("\n" + "="*30 + "\n" + "Following commands returned a non-zero statuscode:\n"+ "="*30)
        for item in broken:
            print(item)
