#!/usr/bin/env python
"""
Produces load on all available CPU cores
"""
from multiprocessing import Pool
from multiprocessing import cpu_count

def f(x):
    while True:
        x*x

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--percent', type=float, default = 1.0, help='Percent cpu load')
    args = parser.parse_args()

    processes = int(args.percent*cpu_count())
    pool = Pool(processes)
    pool.map(f, range(processes))

