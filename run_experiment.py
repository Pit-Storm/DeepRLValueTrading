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

    cagr = ["--cagr", ""]
    episodic = ["--episodic", ""]
    learn_steps = ["--learn_steps=500000"]
    trainsampling = ["--trainsampling", ""]
    val_freq = ["--val_freq=2500"]
    yearrange = ["--yearrange=2","--yearrange=4"]

    if baseline:
        baseline = ["--algo=random","--algo=buyhold"]
        lsts = [baseline, cagr, yearrange]
    else:
        drl = ["--algo=DDPG", "--algo=A2C", "--algo=PPO"]
        lsts = [drl, cagr, episodic, learn_steps, trainsampling, val_freq, yearrange]
    
    params = list(itertools.product(*lsts))
    return params

if __name__ == "__main__":
    # Set baseline parameter of gen_args to False to run the actual experiment
    # If it is true you generate data for baseline comparison
    param_lists = gen_args(baseline = True)
    param_strings = [" ".join(lst) for lst in param_lists]
    cmd_strings = ["python main.py " + item for item in param_strings]
    
    processes = mp.cpu_count()//2
    with mp.Pool(processes=processes, maxtasksperchild=1) as p:
        results = p.map(os.system, cmd_strings)

        broken = [str(results[i]) + ": " + cmd_strings[i] for i in range(len(results)) if results[i] != 0]
        print("\n" + "="*30 + "\n" + "Following commands returned a non-zero statuscode:\n"+ "="*30)
        for item in broken:
            print(item)
