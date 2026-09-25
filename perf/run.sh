#!/bin/zsh
# usage: run.sh <name> <threads> [ENV=VAL ...]
P=${0:A:h}; name=$1; threads=$2; shift 2
rm -rf $P/$name.requests
env "$@" PYTHONPATH=$P/../src ~/Desktop/Claude/Boundarylab/boundary-lab/.venv/bin/python $P/../scripts/benchmark_worker.py \
  --request $P/sawmod.json --backend metal --threads $threads --repeats 2 --label "$name $*" --out $P/$name.json > $P/$name.log 2>&1
rc=$?; echo "$name exit $rc"; exit $rc
