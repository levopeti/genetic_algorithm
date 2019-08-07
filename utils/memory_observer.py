import os
import time
import argparse

parser = argparse.ArgumentParser(description='Prevention of OOM.')
parser.add_argument('--script_name', type=str)
args = parser.parse_args()

print('The script use the pkill -9 -f', args.script_name, 'command, if the available memory less than 0.05 * total memory.')

m = list(map(int, os.popen('free -t -m').readlines()[1].split()[1:]))
total = m[0]

while True:
    m = list(map(int, os.popen('free -t -m').readlines()[1].split()[1:]))

    available = m[-1]

    if available < 0.001 * total:
        print(args.script_name, 'is killed.')
        os.system('pkill -9 -f ' + args.script_name)

    time.sleep(1)

