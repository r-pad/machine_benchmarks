# machine_benchmarks

Basic self-contained torch benchmarking tool for forward, backward, and dataloading speeds. 

## Requirements
* numpy
* six
* torch
* torchvision


## Basic Usage
Change test-dir to test speeds of loading from different disks

``` 
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python speedtest.py --eval --train --load --dynamic-input --times 10000 --test-dir=.
```

## Load Testing
To test usage under high CPU load run load_cpu.py. Will generate mindless workers for given precent of cpu calculating the square of thier process number.

``` 
python load_cpu.py --percent .8
```
