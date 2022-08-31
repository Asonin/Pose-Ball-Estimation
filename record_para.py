import subprocess
import signal
import os
import sys
import multiprocessing
from joblib import Parallel, delayed

num_cores = multiprocessing.cpu_count()

p_list = []

def launch(addr_list, name_list, i):
    addr, name = addr_list[i], name_list[i]
    p = subprocess.Popen(["./job %s %s &" % (addr, name)], shell=True)
    p_list.append(p)
    print(i)
    

addr_list = ["162.105.162.228", "162.105.162.144", "162.105.162.150", "162.105.162.103"]
name_list = ["228", "144", "150", "103"]

num_cams = len(addr_list)

processed_list = Parallel(n_jobs=num_cores)(delayed(launch)(addr_list, name_list, i) for i in range(num_cams))

while True:
    try:
        pass
    except KeyboardInterrupt:
        for p in p_list:
            p.kill()
        print('All subprocesses killed')
        sys.exit()