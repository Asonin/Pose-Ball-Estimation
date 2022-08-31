import subprocess
import signal
import os
import yaml
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--prefix', type=str, default='test')
parser.add_argument('--scene', type=str, default='wusi')
args = parser.parse_args()

# addr_list = ["222.29.70.11",
#              "222.29.70.12",
#              "222.29.70.13",
#              "222.29.70.14",
#              "222.29.70.15",
#              "222.29.70.16",
#              "222.29.70.17",
#              "222.29.70.18"]
# name_list = [str(x+1) for x in range(8)]


with open("configs/%s.yaml" % args.scene) as f:
    cfg_data = yaml.load(f, Loader=yaml.FullLoader)

addr_list = cfg_data['ips']
name_list = [str(x) for x in cfg_data['names']]
    
p_list = []

os.makedirs('out/%s' % args.prefix, exist_ok=True)

for addr, name in zip(addr_list, name_list):
    cmd = "./job %s %s/%s &" % (addr, args.prefix, name)
    print(cmd)
    p = subprocess.Popen([cmd], shell=True)
    p_list.append(p)
    
# print(p_list)

while True:
    try:
        pass
    except KeyboardInterrupt:
        for p in p_list:
            p.kill()
        print('All subprocesses killed')
        sys.exit()
