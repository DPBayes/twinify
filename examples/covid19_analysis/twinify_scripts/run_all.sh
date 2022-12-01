#!/bin/bash

while read ARGS; do
    ./run_twinify.sh $ARGS
done < 'seeds_and_eps.txt'

echo "Skipping non-private inference - it is currently not possible to run this.."
# for n in {1..10}; do
#     ARGS=`sed "${n}q;d" seeds_and_eps.txt`
#     ./run_twinify_nonprivate.sh $ARGS
# done
