#!/bin/zsh
# usage: job.sh <name> '<json job>' [timeout_s=900]
# Streams progress; gives up if the server dies or the timeout passes.
Q=${0:A:h}/queue; name=$1; limit=${3:-900}
pgrep -f perf/quick.py >/dev/null || { echo "quick.py server is not running"; exit 1; }
echo "$2" > $Q/$name.job.json
shown=0; start=$SECONDS
until [ -f $Q/$name.job.done ]; do
  if [ -f $Q/$name.out ]; then n=$(wc -l < $Q/$name.out); [ $n -gt $shown ] && { tail -n +$((shown+1)) $Q/$name.out; shown=$n; }; fi
  pgrep -f perf/quick.py >/dev/null || { echo "server died; see perf/quick.log"; tail -5 ${Q:h}/quick.log; exit 1; }
  (( SECONDS - start > limit )) && { echo "timeout after ${limit}s"; exit 1; }
  sleep 1
done
tail -n +$((shown+1)) $Q/$name.out
