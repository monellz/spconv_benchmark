#!/bin/bash
set -x
srun -n 1 -w gorgon3 python bench_spconv.py $1
